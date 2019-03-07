import dynet_config
dynet_config.set(random_seed=0)
import dynet as dy
import numpy as np
import pytest
import random

from nnlibbench.dynet import LMBuilder


class TestInit:
    def test_ok(self):
        num_words, num_chars = 10, 5
        m = dy.ParameterCollection()
        model = LMBuilder(m, num_words, num_chars)
        assert model.num_words == num_words
        assert model.num_chars == num_chars
        # Word embedding layer
        assert isinstance(model.word_emb, dy.LookupParameters)
        assert model.word_emb.shape() == (num_words, 300)
        assert np.linalg.norm(model.word_emb[0].value()) == pytest.approx(0, abs=1e-6)
        # Char embedding layer
        assert isinstance(model.char_emb, dy.LookupParameters)
        assert model.char_emb.shape() == (num_chars, 15)
        assert np.linalg.norm(model.char_emb[0].value()) == pytest.approx(0, abs=1e-6)
        # Char convolution filters
        assert len(model.cconv_filters) == 7
        assert all(isinstance(p, dy.Parameters) for p in model.cconv_filters)
        assert all(len(p.shape()) == 3 for p in model.cconv_filters)
        assert [p.shape()[0] for p in model.cconv_filters] == list(range(1, 8))
        assert all(p.shape()[1] == 15 for p in model.cconv_filters)
        assert [p.shape()[2] for p in model.cconv_filters] == [50, 100, 150, 200, 200, 200, 200]
        # Highway layers
        size = model.word_emb.shape()[1] + sum(p.shape()[2] for p in model.cconv_filters)
        assert len(model.highway.linears) == 2
        assert all(lin.weight.shape() == (size, size) for lin in model.highway.linears)
        assert all(lin.bias.shape() == (size, ) for lin in model.highway.linears)
        assert len(model.highway.gates) == 2
        assert all(g.weight.shape() == (size, size) for g in model.highway.gates)
        assert all(g.bias.shape() == (size, ) for g in model.highway.gates)
        # Word LSTM
        assert len(model.lstm.get_parameters()) == 2
        hsize = 650
        assert all(
            p[0].shape() == (4 * hsize, hsize if i else size)
            for i, p in enumerate(model.lstm.get_parameters()))
        assert all(p[1].shape() == (4 * hsize, hsize) for p in model.lstm.get_parameters())
        assert all(p[2].shape() == (4 * hsize, ) for p in model.lstm.get_parameters())
        # Output layer
        assert model.output_layer.weight.shape() == (num_words, hsize)
        assert model.output_layer.bias.shape() == (num_words, )

    def test_kwargs(self):
        num_words, num_chars = 10, 5
        kwargs = {
            'word_emb_size': 200,
            'char_emb_size': 30,
            'padding_idx': 1,
            'filter_widths': list(range(1, 7)),
            'num_filters': [25 * i for i in range(1, 7)],
            'highway_layers': 1,
            'lstm_size': 300,
        }
        m = dy.ParameterCollection()
        model = LMBuilder(m, num_words, num_chars, **kwargs)
        assert model.word_emb.shape()[1] == kwargs['word_emb_size']
        assert model.char_emb.shape()[1] == kwargs['char_emb_size']
        assert np.linalg.norm(model.word_emb[kwargs['padding_idx']].value()) == pytest.approx(
            0, abs=1e-6)
        assert np.linalg.norm(model.char_emb[kwargs['padding_idx']].value()) == pytest.approx(
            0, abs=1e-6)
        assert [p.shape()[0] for p in model.cconv_filters] == kwargs['filter_widths']
        assert [p.shape()[2] for p in model.cconv_filters] == kwargs['num_filters']
        assert len(model.highway.linears) == kwargs['highway_layers']
        assert len(model.highway.gates) == kwargs['highway_layers']
        size = model.word_emb.shape()[1] + sum(p.shape()[2] for p in model.cconv_filters)
        hsize = kwargs['lstm_size']
        assert all(
            p[0].shape() == (4 * hsize, hsize if i else size)
            for i, p in enumerate(model.lstm.get_parameters()))
        assert all(p[1].shape() == (4 * hsize, hsize) for p in model.lstm.get_parameters())
        assert all(p[2].shape() == (4 * hsize, ) for p in model.lstm.get_parameters())
        assert model.output_layer.weight.shape() == (num_words, hsize)


def test_forward():
    random.seed(0)

    m = dy.ParameterCollection()
    model = LMBuilder(m, 10, 5)
    slen, wlen = 3, 10
    words = [random.randrange(model.num_words) for _ in range(slen)]
    chars = [[random.randrange(model.num_chars) for _ in range(wlen)] for _ in range(slen)]

    dy.renew_cg()
    scores = model(words, chars)
    assert len(scores) == slen
    assert all(isinstance(score, dy.Expression) for score in scores)
    assert all(len(score.value()) == model.num_words for score in scores)
    dy.sum_elems(dy.esum(scores)).backward()
