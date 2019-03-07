from pathlib import Path
from typing import Dict, Iterator, List, Union

from tqdm import tqdm

Sample = Dict[str, Union[List[str], List[List[str]]]]


def read_corpus(path: Path, encoding: str = 'utf-8') -> Iterator[Sample]:
    lines = path.read_text(encoding=encoding).split('\n')
    for line in tqdm(lines, desc='Reading corpus', unit='line'):
        text = line.rstrip()
        if text:
            yield _make_sample(text)


def _make_sample(text: str) -> Sample:
    toks = text.split()

    words = ['<s>']
    chars = [['<s>']]
    for tok in toks:
        words.append(tok)
        chars.append(list(tok))
    words.append('</s>')
    chars.append(['</s>'])

    return {'words': words, 'chars': chars}
