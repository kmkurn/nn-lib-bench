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
from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda
from text2array import Batch, BatchIterator, ShuffleIterator, Vocab
import torch

from nnlibbench import Sample, make_sample
from nnlibbench.torch import LMLoss, create_lm


def train(
        trn_path: Path,
        save_dir: Path,
        dev_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
        encoding: str = 'utf8',
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 16,
        overwrite: bool = False,
        numeric: bool = False,
) -> None:
    logging.info('Creating save directory if not exist in %s', save_dir)
    save_dir.mkdir(exist_ok=overwrite)

    ### Read/create/load datasets and vocab

    trn_samples = read_or_load_samples(trn_path, encoding=encoding)
    vocab = create_or_load_vocab(trn_samples, path=vocab_path)
    dev_samples = None
    if dev_path is not None:
        dev_samples = read_or_load_samples(dev_path, encoding=encoding, name='dev')

    ### Numericalize datasets

    if not numeric:
        logging.info('Numericalizing train samples')
        trn_samples = list(vocab.apply_to(trn_samples))
        if dev_samples is not None:
            logging.info('Numericalizing dev samples')
            dev_samples = list(vocab.apply_to(dev_samples))

    ### Save vocab and datasets

    fnames = ['vocab.pkl', 'train-samples.pkl', 'dev-samples.pkl']
    objs = [vocab, trn_samples]
    if dev_samples is not None:
        objs.append(dev_samples)
    for fname, obj in zip(fnames, objs):
        save_path = save_dir / fname
        logging.info('Saving to %s', save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f)

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

    ### Prepare engines

    def batch2tensors(
            batch: Batch) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        arr = batch.to_array(pad_with=padding_idx)
        tsr = {k: torch.from_numpy(v) for k, v in arr.items()}
        words = tsr['words'][:, :-1].contiguous()
        chars = tsr['chars'][:, :-1, :].contiguous()
        targets = tsr['words'][:, 1:].contiguous()
        return words, chars, targets

    def train_process(engine: Engine, batch: Batch) -> float:
        model.train()
        words, chars, targets = batch2tensors(batch)
        outputs = model(words, chars)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_process(engine: Engine, batch: Batch) -> Tuple[torch.Tensor, torch.LongTensor]:
        model.eval()
        with torch.no_grad():
            words, chars, targets = batch2tensors(batch)
            return model(words, chars), targets

    trainer = Engine(train_process)
    trn_evaluator = Engine(eval_process)
    dev_evaluator = Engine(eval_process)

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
            'score_function': lambda engine: -engine.state.metrics['ppl'],
            'score_name': 'dev_ppl'
        }
        ckptr_engine = dev_evaluator
    ckptr = ModelCheckpoint(
        str(save_dir / 'ckpts'), 'ckpt', save_as_state_dict=True, **ckptr_kwargs)
    ckptr_engine.add_event_handler(
        Events.EPOCH_COMPLETED, ckptr, {
            'model': model,
            'optimizer': optimizer
        })

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


def read_or_load_samples(
        path: Path,
        encoding: str = 'utf8',
        name: str = 'train',
) -> Sequence[Sample]:
    if path.name.endswith('.pkl'):
        logging.info('Loading %s dataset from %s', name, path)
        with open(path, 'rb') as fb:
            samples = pickle.load(fb)
    else:
        logging.info('Reading %s samples from %s', name, path)
        samples = []
        with open(path, encoding=encoding) as f:
            for line in f:
                text = line.rstrip()
                if text:
                    samples.append(make_sample(text))

    logging.info('Found %d %s samples', len(samples), name)
    return samples


def create_or_load_vocab(samples: Sequence[Sample], path: Optional[Path] = None) -> Vocab:
    if path is None:
        logging.info('Creating vocab from %d samples', len(samples))
        vocab = Vocab.from_samples(samples)
    else:
        logging.info('Reading vocab from %s', path)
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
    p.add_argument('train_path', type=Path, help='path to train corpus/dataset file')
    p.add_argument('save_dir', type=Path, help='save training artifacts here')
    p.add_argument('-d', '--dev-path', type=Path, help='path to dev corpus/dataset file')
    p.add_argument(
        '-v', '--vocab-path', type=Path, help='path to vocab file (implies --numeric)')
    p.add_argument('--encoding', default='utf8', help='file encoding to use')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--max-epochs', type=int, default=50, help='max number of train epochs')
    p.add_argument('-b', '--batch-size', type=int, default=16, help='train batch size')
    p.add_argument('-w', '--overwrite', action='store_true', help='overwrite save directory')
    p.add_argument(
        '-n', '--numeric', action='store_true', help='treat datasets as already numericalized')
    p.add_argument('-l', '--log-level', default='info', help='logging level')
    p.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    args = p.parse_args()

    logging.basicConfig(
        format='%(levelname)s - %(message)s',
        level=getattr(logging, args.log_level.upper()),
    )
    # Turn off ignite's logging so tqdm pbar can actually disappears
    logging.getLogger('ignite').setLevel('CRITICAL')

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(
        args.train_path,
        args.save_dir,
        dev_path=args.dev_path,
        vocab_path=args.vocab_path,
        encoding=args.encoding,
        lr=args.lr,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        numeric=args.numeric or args.vocab_path is not None,
    )
