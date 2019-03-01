from typing import Dict, List, Union

Sample = Dict[str, Union[List[str], List[List[str]]]]


def make_sample(text: str) -> Sample:
    toks = text.split()

    words = ['<s>']
    chars = [['<s>']]
    for tok in toks:
        words.append(tok)
        chars.append(list(tok))
    words.append('</s>')
    chars.append(['</s>'])

    return {'words': words, 'chars': chars}
