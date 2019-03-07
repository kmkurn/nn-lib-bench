#!/usr/bin/env python

from pathlib import Path
import argparse
import pickle
import sys

from text2array import Vocab

from nnlibbench import read_corpus


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Make vocabulary from train corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('path', type=Path, help='path to train corpus')
    p.add_argument('save_to', help='path to save vocab to')
    p.add_argument('--encoding', default='utf8', help='encoding to use')
    p.add_argument('-c', '--min-count', default=2, type=int, help='min word count')
    args = p.parse_args()

    samples = list(read_corpus(args.path, encoding=args.encoding))
    vocab = Vocab.from_samples(samples, options={'words': {'min_count': args.min_count}})
    for name in vocab:
        print('Found', len(vocab[name]), name, 'in vocab', file=sys.stderr)
    with open(args.save_to, 'wb') as f:
        pickle.dump(vocab, f)
