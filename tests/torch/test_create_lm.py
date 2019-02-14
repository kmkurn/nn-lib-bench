import pytest
import torch.nn as nn

from nnlibbench.torch import create_lm


def test_ok():
    num_words, num_chars = 10, 5
    model = create_lm(num_words, num_chars)
    assert isinstance(model, nn.Module)
    assert model.num_words == num_words
    assert model.num_chars == num_chars
    # Word embedding layer
    assert isinstance(model.word_emb, nn.Embedding)
    assert model.word_emb.num_embeddings == num_words
    assert model.word_emb.embedding_dim == 300
    assert model.word_emb.padding_idx == 0
    # Char embedding layer
    assert isinstance(model.char_emb, nn.Embedding)
    assert model.char_emb.num_embeddings == num_chars
    assert model.char_emb.embedding_dim == 15
    assert model.char_emb.padding_idx == model.word_emb.padding_idx
    # Char convolution layers
    assert isinstance(model.char_convs, nn.ModuleList)
    assert len(model.char_convs) == 7
    assert all(isinstance(m, nn.Conv1d) for m in model.char_convs)
    assert all(m.in_channels == model.char_emb.embedding_dim for m in model.char_convs)
    assert [m.out_channels for m in model.char_convs] == [50, 100, 150, 200, 200, 200, 200]
    assert [m.kernel_size[0] for m in model.char_convs] == list(range(1, 8))
    # Highway layer
    assert isinstance(model.highway, nn.Module)
    assert isinstance(model.highway.linears, nn.ModuleList)
    assert len(model.highway.linears) == 2
    assert all(isinstance(m, nn.Linear) for m in model.highway.linears)
    assert all(
        m.in_features == model.word_emb.embedding_dim + sum(
            c.out_channels for c in model.char_convs) for m in model.highway.linears)
    assert all(m.in_features == m.out_features for m in model.highway.linears)
    assert isinstance(model.highway.gates, nn.ModuleList)
    assert len(model.highway.gates) == len(model.highway.linears)
    assert all(isinstance(m, nn.Linear) for m in model.highway.gates)
    assert all(
        g.in_features == l.in_features
        for g, l in zip(model.highway.gates, model.highway.linears))
    assert all(
        g.out_features == l.out_features
        for g, l in zip(model.highway.gates, model.highway.linears))
    # Word LSTM
    assert isinstance(model.lstm, nn.LSTM)
    assert model.lstm.input_size == model.word_emb.embedding_dim + sum(
        m.out_channels for m in model.char_convs)
    assert model.lstm.hidden_size == 650
    assert model.lstm.num_layers == 2
    assert model.lstm.dropout == pytest.approx(0.5)
    assert not model.lstm.bidirectional
    assert model.lstm.batch_first
    # Output layer
    assert isinstance(model.output_layer, nn.Linear)
    assert model.output_layer.in_features == model.lstm.hidden_size
    assert model.output_layer.out_features == model.num_words


def test_kwargs():
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
    model = create_lm(num_words, num_chars, **kwargs)
    assert model.word_emb.embedding_dim == kwargs['word_emb_size']
    assert model.char_emb.embedding_dim == kwargs['char_emb_size']
    assert model.word_emb.padding_idx == kwargs['padding_idx']
    assert model.char_emb.padding_idx == kwargs['padding_idx']
    assert [m.kernel_size[0] for m in model.char_convs] == kwargs['filter_widths']
    assert [m.out_channels for m in model.char_convs] == kwargs['num_filters']
    assert len(model.highway.linears) == kwargs['highway_layers']
    assert model.lstm.hidden_size == kwargs['lstm_size']
