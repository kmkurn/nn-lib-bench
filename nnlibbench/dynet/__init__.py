from typing import List, Optional

import dynet as dy


class LMBuilder:
    def __init__(
            self,
            num_words: int,
            num_chars: int,
            m: dy.ParameterCollection,
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
        # TODO add bias for conv
        self.cconv_filters = [
            m.add_parameters((w, char_emb_size, n)) for w, n in zip(filter_widths, num_filters)
        ]
        input_dim = word_emb_size + sum(p.shape()[2] for p in self.cconv_filters)
        self.highway = HighwayBuilder(input_dim, m, num_layers=highway_layers)
        self.lstm = dy.VanillaLSTMBuilder(2, input_dim, lstm_size, m)
        self.output_layer = LinearBuilder(lstm_size, num_words, m)

    @property
    def num_words(self) -> int:
        return self.word_emb.shape()[0]

    @property
    def num_chars(self) -> int:
        return self.char_emb.shape()[0]

    def __call__(self, words: List[int], chars: List[List[int]]) -> dy.Expression:
        assert len(words) == len(chars)

        char_emb_sz = self.char_emb.shape()[1]

        inputs = []
        for w, cs in zip(words, chars):
            # shape: (len(cs), char_emb_sz)
            e_cs = dy.concatenate([dy.reshape(self.char_emb[c], (1, char_emb_sz)) for c in cs])
            # shape: (len(cs), char_emb_sz, 1)
            e_cs = dy.reshape(e_cs, (len(cs), char_emb_sz, 1))

            res = []
            for f in self.cconv_filters:
                w, _, n = f.shape()
                # shape: (w, char_emb_sz, 1, n)
                e_f = dy.reshape(f, (w, char_emb_sz, 1, n))
                # shape: (len(cs)-w+1, 1, n)
                e_res = dy.conv2d(e_cs, e_f, [1, 1])
                # shape: (1, n)
                e_res = dy.max_dim(e_res)
                # shape: (n,)
                e_res = dy.reshape(e_res, (n, ))
                res.append(e_res)

            # shape: (sum(num_filters),)
            e_cinp = dy.concatenate(res)
            # shape: (word_emb_sz,)
            e_w = self.word_emb[w]
            # shape: (sum(num_filters) + word_emb_sz)
            e_inp = dy.concatenate([e_cinp, e_w])
            # shape: (sum(num_filters) + word_emb_sz)
            e_inp = self.highway(e_inp)
            inputs.append(e_inp)

        states = self.lstm.initial_state().add_inputs(inputs)
        # each shape: (lstm_size,)
        outputs = [s.output() for s in states]
        # each shape: (num_words,)
        outputs = [dy.log_softmax(self.output_layer(out)) for out in outputs]

        return outputs


class HighwayBuilder:
    def __init__(self, size: int, m: dy.ParameterCollection, num_layers: int = 2) -> None:
        self.linears = [LinearBuilder(size, size, m) for _ in range(num_layers)]
        self.gates = [LinearBuilder(size, size, m) for _ in range(num_layers)]

    def __call__(self, inp: dy.Expression) -> dy.Expression:
        for linear, gate in zip(self.linears, self.gates):
            t = dy.logistic(gate(inp))
            inp = dy.cmult(t, dy.rectify(linear(inp))) + dy.cmult((1 - t), inp)
        return inp


class LinearBuilder:
    def __init__(self, in_size: int, out_size: int, m: dy.ParameterCollection) -> None:
        self.weight = m.add_parameters((out_size, in_size))
        self.bias = m.add_parameters((out_size, ))

    def __call__(self, inp: dy.Expression) -> dy.Expression:
        return self.weight * inp + self.bias
