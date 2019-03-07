from pathlib import Path
from unittest.mock import MagicMock

from nnlibbench import read_corpus


def test_ok():
    path = MagicMock(spec=Path)
    path.read_text.return_value = 'aa aa\n\nbbb bb\n  \nc cc\n'
    samples = list(read_corpus(path))
    assert len(samples) == 3
    assert samples[0] == {
        'words': ['<s>', 'aa', 'aa', '</s>'],
        'chars': [['<s>'], ['a', 'a'], ['a', 'a'], ['</s>']]
    }
    assert samples[1] == {
        'words': ['<s>', 'bbb', 'bb', '</s>'],
        'chars': [['<s>'], ['b', 'b', 'b'], ['b', 'b'], ['</s>']]
    }
    assert samples[2] == {
        'words': ['<s>', 'c', 'cc', '</s>'],
        'chars': [['<s>'], ['c'], ['c', 'c'], ['</s>']]
    }
