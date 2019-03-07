#!/usr/bin/env python

from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
import argparse
import logging
import math
import pickle
import random

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.metrics import Loss, MetricsLambda
from text2array import Batch, BatchIterator, ShuffleIterator, Vocab
import torch

from nnlibbench import Sample
from nnlibbench.torch import LMLoss, create_lm
from nnlibbench.torch.serialization import dump


def train(
        trn_path: Path,
        vocab_path: Path,
        save_dir: Path,
        dev_path: Optional[Path] = None,
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 16,
        patience: int = 5,
        device: Optional[str] = None,
) -> None:
    logging.info('Creating save directory if not exist in %s', save_dir)
    save_dir.mkdir()

    ### Read/create/load samples and vocab

    trn_samples = load_samples(trn_path)
    vocab = load_vocab(vocab_path)
    dev_samples = None
    if dev_path is not None:
        dev_samples = load_samples(dev_path, name='dev')

    ### Create model, optimizer, and loss fn

    logging.info('Creating language model')
    padding_idx = vocab['words']['<pad>']
    max_width = get_max_filter_width([trn_samples, dev_samples])
    model = create_lm(
        len(vocab['words']),
        len(vocab['chars']),
        padding_idx=padding_idx,
        filter_widths=list(range(1, max_width)),
    )
    logging.info('Model created with %d parameters', sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = LMLoss(padding_idx=padding_idx)

    ### Save model metadata

    metadata_path = save_dir / 'metadata.yml'
    logging.info('Saving model metadata to %s', metadata_path)
    metadata_path.write_text(dump(model), encoding='utf8')

    ### Prepare engines

    def batch2tensors(
            batch: Batch,
            device: Optional[str] = None,
            non_blocking: Optional[bool] = None,
    ) -> Tuple[dict, torch.LongTensor]:
        arr = batch.to_array(pad_with=padding_idx)
        tsr = {k: torch.from_numpy(v).to(device=device) for k, v in arr.items()}
        words = tsr['words'][:, :-1].contiguous()
        chars = tsr['chars'][:, :-1, :].contiguous()
        targets = tsr['words'][:, 1:].contiguous()
        return {'words': words, 'chars': chars}, targets

    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=device, prepare_batch=batch2tensors)
    trn_evaluator = create_supervised_evaluator(
        model, device=device, prepare_batch=batch2tensors)
    dev_evaluator = create_supervised_evaluator(
        model, device=device, prepare_batch=batch2tensors)

    ### Attach metrics

    loss = Loss(loss_fn, batch_size=lambda tgt: (tgt != padding_idx).long().sum().item())
    ppl = MetricsLambda(math.exp, loss)
    loss.attach(trn_evaluator, 'loss')
    loss.attach(dev_evaluator, 'loss')
    ppl.attach(trn_evaluator, 'ppl')
    ppl.attach(dev_evaluator, 'ppl')

    ### Attach timers

    epoch_timer = Timer()
    epoch_timer.attach(trainer, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED)

    ### Attach progress bars

    trn_pbar = ProgressBar(bar_format=None, unit='batch', desc='Training')
    trn_pbar.attach(
        trainer, output_transform=lambda loss: {
            'loss': loss,
            'ppl': math.exp(loss)
        })
    eval_pbar = ProgressBar(bar_format=None, unit='sent', desc='Evaluating')
    eval_pbar.attach(trn_evaluator)
    eval_pbar.attach(dev_evaluator)

    ### Attach checkpointers

    if dev_samples is None:
        ckptr_kwargs: dict = {'save_interval': 1, 'n_saved': 5}
        ckptr_engine = trainer
    else:
        ckptr_kwargs = {
            'score_function': lambda eng: -eng.state.metrics['ppl'],
            'score_name': 'dev_ppl'
        }
        ckptr_engine = dev_evaluator
    ckptr = ModelCheckpoint(
        str(save_dir / 'checkpoints'), 'ckpt', save_as_state_dict=True, **ckptr_kwargs)
    ckptr_engine.add_event_handler(
        Events.EPOCH_COMPLETED, ckptr, {
            'model': model,
            'optimizer': optimizer
        })

    ### Attach early stopper

    if dev_samples is not None:
        early_stopper = EarlyStopping(patience, lambda eng: -eng.state.metrics['ppl'], trainer)
        dev_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

    ### Attach custom handlers

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch(engine: Engine) -> None:
        logging.info('[Epoch %d/%d] Starting', engine.state.epoch, engine.state.max_epochs)

    @trainer.on(Events.EPOCH_COMPLETED)
    def complete_epoch(engine: Engine) -> None:
        epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs
        logging.info(
            '[Epoch %d/%d] Done in %s', epoch, max_epochs,
            timedelta(seconds=epoch_timer.value()))
        logging.info('[Epoch %d/%d] Evaluating on train corpus', epoch, max_epochs)
        trn_evaluator.run(BatchIterator(trn_samples))
        if dev_samples is not None:
            logging.info('[Epoch %d/%d] Evaluating on dev corpus', epoch, max_epochs)
            dev_evaluator.run(BatchIterator(dev_samples))

    @trn_evaluator.on(Events.COMPLETED)
    @dev_evaluator.on(Events.COMPLETED)
    def print_metrics(engine: Engine) -> None:
        loss = engine.state.metrics['loss']
        ppl = engine.state.metrics['ppl']
        logging.info('||| loss %.4f | ppl %.4f', loss, ppl)

    ### Start training

    iterator = ShuffleIterator(trn_samples, key=lambda s: len(s['words']))
    iterator = BatchIterator(iterator, batch_size=batch_size)
    try:
        trainer.run(iterator, max_epochs=max_epochs)
    except KeyboardInterrupt:
        logging.info('Interrupt detected, aborting training')
        trainer.terminate()


def load_samples(path: Path, name: str = 'train') -> Sequence[Sample]:
    logging.info('Loading %s samples from %s', name, path)
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    logging.info('Found %d %s samples', len(samples), name)
    return samples


def load_vocab(path: Path) -> Vocab:
    logging.info('Loading vocab from %s', path)
    with open(path, 'rb') as fb:
        vocab = pickle.load(fb)
    for name in vocab:
        logging.info('Found %d %s', len(vocab[name]), name)
    return vocab


def get_max_filter_width(samples_iter: Iterable[Optional[Sequence[Sample]]]) -> int:
    max_width = 8
    for samples in samples_iter:
        if samples is None:
            continue
        for s in samples:
            max_width = min(max_width, len(s['words']))
    return max_width


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Run LM model built with PyTorch.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('train_path', type=Path, help='path to train .pkl file')
    p.add_argument('vocab_path', type=Path, help='path to vocab .pkl file')
    p.add_argument('save_dir', type=Path, help='save training artifacts here')
    p.add_argument('-d', '--dev-path', type=Path, help='path to dev .pkl file')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('-e', '--max-epochs', type=int, default=50, help='max number of train epochs')
    p.add_argument('-b', '--batch-size', type=int, default=16, help='train batch size')
    p.add_argument('-p', '--patience', type=int, default=5, help='patience for early stopping')
    p.add_argument('--device', help='run on this device (e.g. "cpu", "cuda", "cuda:0")')
    p.add_argument('-l', '--log-level', default='info', help='logging level')
    p.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    args = p.parse_args()

    logging.basicConfig(
        format='%(levelname)s (%(name)s) %(message)s',
        level=getattr(logging, args.log_level.upper()),
    )
    # Turn off ignite.engine's logging so tqdm pbar can actually disappears
    logging.getLogger('ignite.engine').setLevel('CRITICAL')

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(
        args.train_path,
        args.vocab_path,
        args.save_dir,
        dev_path=args.dev_path,
        lr=args.lr,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
    )
