#!/usr/bin/env python

from pathlib import Path
import argparse
import pickle

from text2array import Vocab

from nnlibbench import read_corpus


def load_vocab(path: Path) -> Vocab:
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Apply vocab to samples in a corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('vocab', type=Path, help='path to vocab file')
    p.add_argument('corpus', type=Path, help='path to corpus file')
    p.add_argument('save_to', help='path to save the resulting samples')
    p.add_argument('--encoding', default='utf8', help='encoding to use')
    args = p.parse_args()

    vocab = load_vocab(args.vocab)
    samples = read_corpus(args.corpus, encoding=args.encoding)
    samples = vocab.apply_to(samples)
    with open(args.save_to, 'wb') as f:
        pickle.dump(list(samples), f)
