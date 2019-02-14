from typing import Dict, List, Union


def make_sample(text: str) -> Dict[str, Union[List[str], List[List[str]]]]:
    toks = text.split()

    words = ['<s>']
    chars = [['<s>']]
    targets = []
    for tok in toks:
        words.append(tok)
        chars.append(list(tok))
        targets.append(tok)
    targets.append('</s>')

    return {'words': words, 'chars': chars, 'targets': targets}
