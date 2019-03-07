from typing import List, Optional

import dynet as dy


class LMBuilder:
    def __init__(
            self,
            m: dy.ParameterCollection,
            num_words: int,
            num_chars: int,
            word_emb_size: int = 300,
            char_emb_size: int = 15,
            padding_idx: int = 0,
            filter_widths: Optional[List[int]] = None,
            num_filters: Optional[List[int]] = None,
            highway_layers: int = 2,
            lstm_size: int = 650,
    ) -> None:
        if filter_widths is None:
            filter_widths = list(range(1, 8))
        if num_filters is None:
            num_filters = [min(50 * w, 200) for w in filter_widths]

        self.word_emb = m.add_lookup_parameters((num_words, word_emb_size))
        self.word_emb.init_row(padding_idx, [0] * word_emb_size)
        self.char_emb = m.add_lookup_parameters((num_chars, char_emb_size))
        self.char_emb.init_row(padding_idx, [0] * char_emb_size)
        self.cconv_filters = [
            m.add_parameters((w, n)) for w, n in zip(filter_widths, num_filters)
        ]
        input_dim = sum(p.shape()[1] for p in self.cconv_filters)
        self.highway = HighwayBuilder(m, input_dim, num_layers=highway_layers)
        self.lstm = dy.VanillaLSTMBuilder(2, input_dim, lstm_size, m)
        self.output_layer = LinearBuilder(m, lstm_size, num_words)

    @property
    def num_words(self) -> int:
        return self.word_emb.shape()[0]

    @property
    def num_chars(self) -> int:
        return self.char_emb.shape()[0]


class HighwayBuilder:
    def __init__(self, m: dy.ParameterCollection, size: int, num_layers: int = 2) -> None:
        self.linears = [LinearBuilder(m, size, size) for _ in range(num_layers)]
        self.gates = [LinearBuilder(m, size, size) for _ in range(num_layers)]


class LinearBuilder:
    def __init__(self, m: dy.ParameterCollection, in_size: int, out_size: int) -> None:
        self.weight = m.add_parameters((in_size, out_size))
        self.bias = m.add_parameters((out_size, ))
