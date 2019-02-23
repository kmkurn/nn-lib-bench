#!/usr/bin/env python

from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional
import argparse
import logging
import math
import pickle
import time

from text2array import Dataset, Vocab
from tqdm import tqdm
import torch

from nnlibbench import make_sample
from nnlibbench.torch import create_lm


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

    trn_dataset = read_or_load_dataset(trn_path, encoding=encoding)
    dev_dataset = None
    if dev_path is not None:
        dev_dataset = read_or_load_dataset(dev_path, encoding=encoding, name='dev')

    vocab = create_or_load_vocab(trn_dataset, path=vocab_path)

    if not numeric:
        logging.info('Numericalizing train dataset')
        trn_dataset.apply_vocab(vocab)
        if dev_dataset is not None:
            logging.info('Numericalizing dev dataset')
            dev_dataset.apply_vocab(vocab)

    fnames = ['vocab.pkl', 'train-dataset.pkl', 'dev-dataset.pkl']
    objs = [vocab, trn_dataset]
    if dev_dataset is not None:
        objs.append(dev_dataset)
    for fname, obj in zip(fnames, objs):
        save_path = save_dir / fname
        logging.info('Saving to %s', save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f)

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

    logging.info('Creating optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logging.info('Starting training')
    trn_start = time.time()

    for epoch in range(1, max_epochs + 1):
        logging.info('Starting epoch %d/%d', epoch, max_epochs)
        ep_start = time.time()
        train_epoch(model, optimizer, trn_dataset, batch_size, padding_idx=padding_idx)
        logging.info(
            'Epoch %d/%d completed in %s', epoch, max_epochs,
            timedelta(seconds=time.time() - ep_start))

        logging.info('Evaluating on train corpus')
        ppl = evaluate(model, trn_dataset, padding_idx=padding_idx)
        logging.info('Result on TRAIN: ppl %.4f', ppl)

        if dev_dataset is not None:
            logging.info('Evaluating on dev corpus')
            ppl = evaluate(model, dev_dataset, padding_idx=padding_idx)
            logging.info('Result on DEV: ppl %.4f', ppl)

    logging.info('Training completed in %s', timedelta(seconds=time.time() - trn_start))


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


def train_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        batch_size: int,
        padding_idx: int = 0,
) -> None:
    model.train()
    pbar = tqdm(total=len(dataset), unit='sent', leave=False)

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
    pbar = tqdm(total=len(dataset), unit='sent', leave=False)
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
    logging.debug('Total loss: %.4f', tot_loss)
    logging.debug('Total tokens: %d', tot_tokens)
    return math.exp(tot_loss / tot_tokens)


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
