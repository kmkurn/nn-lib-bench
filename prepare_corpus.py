#!/usr/bin/env python

from pathlib import Path
import argparse
import json


def main(path: Path, encoding: str = 'utf-8') -> None:
    with open(path, encoding=encoding) as f:
        for line in f:
            obj = json.loads(line.rstrip())
            for para in obj['paragraphs']:
                for sent in para:
                    print(' '.join(sent))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare corpus for language modeling.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=Path, help='path to the JSONL corpus file to prepare')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    args = parser.parse_args()
    main(args.path, encoding=args.encoding)
