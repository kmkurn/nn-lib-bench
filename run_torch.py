#!/usr/bin/env python

from datetime import timedelta
from pathlib import Path
from typing import Optional
import argparse
import logging
import math
import time

from text2array import Dataset, Vocab
from tqdm import tqdm
import torch

from nnlibbench import make_sample
from nnlibbench.torch import create_lm


def train(
        trn_path: Path,
        encoding: str = 'utf8',
        dev_path: Optional[Path] = None,
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 16,
) -> None:
    logging.info('Reading train corpus from %s', trn_path)
    trn_dataset = read_corpus(trn_path, encoding=encoding)
    dev_dataset = None
    if dev_path is not None:
        logging.info('Reading dev corpus from %s', dev_path)
        dev_dataset = read_corpus(dev_path, encoding=encoding)

    logging.info('Creating vocab and numericalizing dataset(s)')
    vocab = Vocab.from_samples(trn_dataset)
    trn_dataset.apply_vocab(vocab)
    if dev_dataset is not None:
        dev_dataset.apply_vocab(vocab)

    logging.info('Creating language model')
    padding_idx = vocab['words']['<pad>']
    # Find max possible filter width
    max_width = 8
    for dat in (trn_dataset, dev_dataset):
        for s in dat:
            max_width = min(max_width, len(s['words']))

    model = create_lm(
        len(vocab['words']),
        len(vocab['chars']),
        padding_idx=padding_idx,
        filter_widths=list(range(1, max_width)),
    )
    logging.info('Model created with %d parameters', sum(p.numel() for p in model.parameters()))

    logging.info('Creating optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logging.info('Starting training')
    trn_start = time.time()

    for epoch in range(1, max_epochs + 1):
        logging.info('Starting epoch %d/%d', epoch, max_epochs)
        train_epoch(model, optimizer, trn_dataset, batch_size, padding_idx=padding_idx)
        logging.info('Epoch %d/%d completed', epoch, max_epochs)

        logging.info('Evaluating on train corpus')
        ppl = evaluate(model, trn_dataset, padding_idx=padding_idx)
        logging.info('Result on TRAIN: ppl %.4f', ppl)

        if dev_dataset is not None:
            logging.info('Evaluating on dev corpus')
            ppl = evaluate(model, dev_dataset, padding_idx=padding_idx)
            logging.info('Result on DEV: ppl %.4f', ppl)

    logging.info('Training completed in %s', timedelta(seconds=time.time() - trn_start))


def read_corpus(path: Path, encoding: str = 'utf8') -> Dataset:
    samples = []
    with open(path, encoding=encoding) as f:
        for line in f:
            samples.append(make_sample(line.rstrip()))
    return Dataset(samples)


def train_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        batch_size: int,
        padding_idx: int = 0,
) -> None:
    model.train()
    pbar = tqdm(total=len(dataset), unit='sent')

    for batch in dataset.shuffle_by(lambda s: len(s['words'])).batch(batch_size):
        arr = batch.to_array(pad_with=padding_idx)
        tsr = {k: torch.from_numpy(v) for k, v in arr.items()}
        words = tsr['words'][:, :-1].contiguous()
        chars = tsr['chars'][:, :-1, :].contiguous()
        targets = tsr['words'][:, 1:].contiguous()

        loss = model(words, chars, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_tokens = (words != padding_idx).long().sum()
        mean_loss = loss.item() / n_tokens.item()
        pbar.set_postfix(mean_loss=mean_loss)
        pbar.update(words.size(0))

    pbar.close()


def evaluate(
        model: torch.nn.Module,
        dataset: Dataset,
        padding_idx: int = 0,
) -> float:
    model.eval()
    pbar = tqdm(total=len(dataset), unit='sent')
    tot_loss, tot_tokens = 0, 0

    for batch in dataset.batch(1):
        arr = batch.to_array(pad_with=padding_idx)
        tsr = {k: torch.from_numpy(v) for k, v in arr.items()}
        words = tsr['words'][:, :-1].contiguous()
        chars = tsr['chars'][:, :-1, :].contiguous()
        targets = tsr['words'][:, 1:].contiguous()

        loss = model(words, chars, targets)
        tot_loss += loss.item()
        tot_tokens += (words != padding_idx).long().sum().item()
        pbar.update(words.size(0))

    pbar.close()
    return math.exp(tot_loss / tot_tokens)


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
    args = p.parse_args()

    logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
    train(
        args.path,
        encoding=args.encoding,
        dev_path=args.dev,
        lr=args.lr,
        max_epochs=args.max_epochs,
        batch_size=args.bsz,
    )
