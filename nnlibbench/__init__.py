from typing import Dict, List, Union


def make_sample(text: str) -> Dict[str, Union[List[str], List[List[str]]]]:
    toks = text.split()

    words = ['<s>']
    chars = [['<s>']]
    for tok in toks:
        words.append(tok)
        chars.append(list(tok))
    words.append('</s>')
    chars.append(['</s>'])

    return {'words': words, 'chars': chars}
