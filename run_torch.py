#!/usr/bin/env python

from pathlib import Path
from typing import Iterator, Optional
import argparse
import logging
import math

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from text2array import Batch, Dataset, Vocab
import torch
import torch.optim as optim

from nnlibbench import make_sample
from nnlibbench.torch import create_lm


def train(
        trn_path: Path,
        encoding: str = 'utf8',
        dev_path: Optional[Path] = None,
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 16,
        test_batch_size: int = 128,
) -> None:
    logging.info('Reading train corpus from %s', trn_path)
    trn_dataset = read_corpus(trn_path, encoding=encoding)
    trn_batches = Batches(trn_dataset, batch_size)
    dev_dataset = None
    dev_batches = None
    if dev_path is not None:
        logging.info('Reading dev corpus from %s', dev_path)
        dev_dataset = read_corpus(dev_path, encoding=encoding)
        dev_batches = Batches(dev_dataset, test_batch_size, shuffle=False)

    logging.info('Create vocab and numericalize dataset(s)')
    vocab = Vocab.from_samples(trn_dataset)
    trn_dataset.apply_vocab(vocab)
    if dev_dataset is not None:
        dev_dataset.apply_vocab(vocab)

    logging.info('Create language model')
    padding_idx = vocab['words']['<pad>']
    model = create_lm(len(vocab['words']), len(vocab['chars']), padding_idx=padding_idx)
    logging.info('Model created with %d parameters', sum(p.numel() for p in model.parameters()))

    logging.info('Create optimizer')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def _update(trainer: Engine, batch: Batch) -> dict:
        ts = to_tensors(batch, pad_with=model.word_emb.padding_idx)
        model.train()
        loss = model(ts['words'], ts['chars'], ts['targets'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            'loss': loss.item(),
            'n_tokens': torch.sum(ts['words'] != padding_idx).item(),
        }

    def _evaluate(engine: Engine, batch: Batch) -> dict:
        ts = to_tensors(batch, pad_with=model.word_emb.padding_idx)
        model.eval()
        loss = model(ts['words'], ts['chars'], ts['targets'])
        return {
            'loss': loss.item(),
            'n_tokens': torch.sum(ts['words'] != padding_idx).item(),
        }

    # TODO manual progress bar to control what gets displayed
    trainer = Engine(_update)
    ProgressBar(persist=True, bar_format=None).attach(
        trainer, output_transform=lambda out: {'loss': out['loss'] / out['n_tokens']})

    evaluator = Engine(_evaluate)
    ProgressBar(bar_format=None, desc='Evaluation').attach(evaluator)
    Perplexity().attach(evaluator, 'ppl')

    @trainer.on(Events.EPOCH_STARTED)
    def on_epoch_started(engine: Engine) -> None:
        logging.info('Starting epoch %d', engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluation(engine: Engine) -> None:
        logging.info('Epoch %d completed', engine.state.epoch)
        logging.info('Evaluating on train corpus')
        evaluator.run(trn_batches)
        logging.info('Result on TRAIN: ppl %.4f', evaluator.state.metrics['ppl'])
        if dev_batches is not None:
            logging.info('Evaluating on dev corpus')
            evaluator.run(dev_batches)
            logging.info('Result on DEV: ppl %.4f', evaluator.state.metrics['ppl'])

    try:
        trainer.run(trn_batches, max_epochs=max_epochs)
    except KeyboardInterrupt:
        logging.info('Interrupt detected, terminating')
        trainer.terminate()


def read_corpus(path: Path, encoding: str = 'utf8') -> Dataset:
    samples = []
    with open(path, encoding=encoding) as f:
        for line in f:
            samples.append(make_sample(line.rstrip()))
    return Dataset(samples)


class Batches:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Batch]:
        if self.shuffle:
            self.dataset.shuffle_by(lambda s: len(s['words']))
        return self.dataset.batch(self.batch_size)

    def __len__(self) -> int:
        n = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size != 0:
            n += 1
        return n


def to_tensors(batch: Batch, pad_with: int = 0) -> dict:
    arr = batch.to_array(pad_with=pad_with)
    return {k: torch.from_numpy(v) for k, v in arr.items()}


class Perplexity(Metric):
    def reset(self) -> None:
        self.loss = 0
        self.n_tokens = 0

    def compute(self) -> float:
        return math.exp(self.loss / self.n_tokens)

    def update(self, output: dict) -> None:
        self.loss += output['loss']
        self.n_tokens += output['n_tokens']


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Run LM model built with PyTorch.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('path', type=Path, help='path to train corpus file')
    p.add_argument('--encoding', default='utf8', help='file encoding to use')
    p.add_argument('--dev', type=Path, help='path to dev corpus file')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--max-epochs', type=int, default=50, help='max number of train epochs')
    p.add_argument('--bsz', type=int, default=16, help='train batch size')
    p.add_argument('--test-bsz', type=int, default=128, help='test batch size')
    args = p.parse_args()

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.getLogger('ignite.engine').setLevel(logging.ERROR)
    train(
        args.path,
        encoding=args.encoding,
        dev_path=args.dev,
        lr=args.lr,
        max_epochs=args.max_epochs,
        batch_size=args.bsz,
        test_batch_size=args.test_bsz,
    )
