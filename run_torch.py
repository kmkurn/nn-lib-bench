#!/usr/bin/env python

from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple
import argparse
import logging
import math
import pickle

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import Loss, MetricsLambda
from text2array import Batch, Dataset, Vocab
import torch

from nnlibbench import make_sample
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

    trn_dataset = read_or_load_dataset(trn_path, encoding=encoding)
    vocab = create_or_load_vocab(trn_dataset, path=vocab_path)
    dev_dataset = None
    if dev_path is not None:
        dev_dataset = read_or_load_dataset(dev_path, encoding=encoding, name='dev')

    ### Numericalize datasets

    if not numeric:
        logging.info('Numericalizing train dataset')
        trn_dataset.apply_vocab(vocab)
        if dev_dataset is not None:
            logging.info('Numericalizing dev dataset')
            dev_dataset.apply_vocab(vocab)

    ### Save vocab and datasets

    fnames = ['vocab.pkl', 'train-dataset.pkl', 'dev-dataset.pkl']
    objs = [vocab, trn_dataset]
    if dev_dataset is not None:
        objs.append(dev_dataset)
    for fname, obj in zip(fnames, objs):
        save_path = save_dir / fname
        logging.info('Saving to %s', save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f)

    ### Create model, optimizer, and loss fn

    logging.info('Creating language model')
    padding_idx = vocab['words']['<pad>']
    max_width = get_max_filter_width([trn_dataset, dev_dataset])
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
    evaluator = Engine(eval_process)

    ### Attach handlers and metrics

    epoch_timer = Timer()
    epoch_timer.attach(trainer, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED)
    trn_pbar = ProgressBar(bar_format=None, unit='batch')
    trn_pbar.attach(
        trainer, output_transform=lambda loss: {
            'loss': loss,
            'ppl': math.exp(loss)
        })
    eval_pbar = ProgressBar(bar_format=None, unit='sent')
    eval_pbar.attach(evaluator)

    loss = Loss(loss_fn, batch_size=lambda tgt: (tgt != padding_idx).long().sum().item())
    ppl = MetricsLambda(math.exp, loss)
    loss.attach(evaluator, 'loss')
    ppl.attach(evaluator, 'ppl')

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch(engine: Engine) -> None:
        logging.info('[Epoch %d/%d] Starting', engine.state.epoch, engine.state.max_epochs)

    @trainer.on(Events.EPOCH_COMPLETED)
    def complete_epoch(engine: Engine) -> None:
        logging.info(
            '[Epoch %d/%d] Done in %s', engine.state.epoch, engine.state.max_epochs,
            timedelta(seconds=epoch_timer.value()))
        logging.info(
            '[Epoch %d/%d] Evaluating on train corpus', engine.state.epoch,
            engine.state.max_epochs)
        evaluator.run(list(trn_dataset.batch(1)))
        if dev_dataset is not None:
            logging.info(
                '[Epoch %d/%d] Evaluating on dev corpus', engine.state.epoch,
                engine.state.max_epochs)
            evaluator.run(list(dev_dataset.batch(1)))

    @evaluator.on(Events.COMPLETED)
    def print_metrics(engine: Engine) -> None:
        loss = engine.state.metrics['loss']
        ppl = engine.state.metrics['ppl']
        logging.info('|| loss %.4f | ppl %.4f', loss, ppl)

    ### Start training

    try:
        trainer.run(list(trn_dataset.batch(batch_size)), max_epochs=max_epochs)
    except KeyboardInterrupt:
        trainer.terminate()


def read_or_load_dataset(path: Path, encoding: str = 'utf8', name: str = 'train') -> Dataset:
    if path.name.endswith('.pkl'):
        logging.info('Loading %s dataset from %s', name, path)
        with open(path, 'rb') as fb:
            dataset = pickle.load(fb)
    else:
        logging.info('Reading %s corpus from %s', name, path)
        samples = []
        with open(path, encoding=encoding) as f:
            for line in f:
                text = line.rstrip()
                if text:
                    samples.append(make_sample(text))
        dataset = Dataset(samples)

    logging.info('Found %d %s samples', len(dataset), name)
    return dataset


def create_or_load_vocab(dataset: Dataset, path: Optional[Path] = None) -> Vocab:
    if path is None:
        logging.info('Creating vocab from dataset with %d samples', len(dataset))
        vocab = Vocab.from_samples(dataset)
    else:
        logging.info('Reading vocab from %s', path)
        with open(path, 'rb') as fb:
            vocab = pickle.load(fb)

    for name in vocab:
        logging.info('Found %d %s', len(vocab[name]), name)
    return vocab


def get_max_filter_width(datasets: Iterable[Dataset]) -> int:
    max_width = 8
    for dat in datasets:
        if dat is None:
            continue
        for s in dat:
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
    args = p.parse_args()

    logging.basicConfig(
        format='%(levelname)s - %(message)s',
        level=getattr(logging, args.log_level.upper()),
    )
    # Turn off ignite's logging so tqdm pbar can actually disappears
    logging.getLogger('ignite').setLevel('CRITICAL')

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
