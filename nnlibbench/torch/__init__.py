__all__ = [
    'LMLoss',
    'create_lm',
]

from typing import List, Optional

import torch.nn as nn

from .losses import LMLoss
from .modules import Highway, LanguageModel


def create_lm(
        num_words: int,
        num_chars: int,
        word_emb_size: int = 300,
        char_emb_size: int = 15,
        padding_idx: int = 0,
        filter_widths: Optional[List[int]] = None,
        num_filters: Optional[List[int]] = None,
        highway_layers: int = 2,
        lstm_size: int = 650,
) -> nn.Module:
    if filter_widths is None:
        filter_widths = list(range(1, 8))
    if num_filters is None:
        num_filters = [min(200, w * 50) for w in filter_widths]

    word_emb = nn.Embedding(num_words, word_emb_size, padding_idx=padding_idx)
    input_size = word_emb_size

    char_emb = nn.Embedding(num_chars, char_emb_size, padding_idx=padding_idx)
    char_convs = nn.ModuleList()
    for n, w in zip(num_filters, filter_widths):
        char_convs.append(nn.Conv1d(char_emb_size, n, w))
        input_size += n

    highway = Highway(input_size, highway_layers)
    lstm = nn.LSTM(input_size, lstm_size, num_layers=2, dropout=0.5, batch_first=True)
    output_layer = nn.Linear(lstm.hidden_size, num_words)

    return LanguageModel(word_emb, char_emb, char_convs, highway, lstm, output_layer)
