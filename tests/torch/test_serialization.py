import torch.nn as nn

from nnlibbench.torch import Highway, LanguageModel
from nnlibbench.torch.serialization import dump, load


def test_linear():
    m = nn.Linear(2, 3)
    m_ = load(dump(m))
    assert isinstance(m_, nn.Linear)
    assert m_.in_features == m.in_features
    assert m_.out_features == m.out_features
    assert (m_.bias is None) == (m.bias is None)


def test_embedding():
    m = nn.Embedding(10, 20, padding_idx=5)
    m_ = load(dump(m))
    assert isinstance(m_, nn.Embedding)
    assert m_.num_embeddings == m.num_embeddings
    assert m_.embedding_dim == m.embedding_dim
    assert m_.padding_idx == m.padding_idx


def test_lstm():
    m = nn.LSTM(10, 20, num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
    m_ = load(dump(m))
    assert isinstance(m_, nn.LSTM)
    assert m_.input_size == m.input_size
    assert m_.hidden_size == m.hidden_size
    assert m_.num_layers == m.num_layers
    assert m_.dropout == m.dropout
    assert m_.batch_first == m.batch_first
    assert m_.bidirectional == m.bidirectional


def test_module_list():
    m = nn.ModuleList([nn.Linear(2, 3), nn.Embedding(10, 20)])
    m_ = load(dump(m))
    assert isinstance(m_, nn.ModuleList)
    assert len(m_) == len(m)
    assert isinstance(m_[0], nn.Linear)
    assert isinstance(m_[1], nn.Embedding)


def test_conv1d():
    m = nn.Conv1d(2, 3, 4)
    m_ = load(dump(m))
    assert isinstance(m_, nn.Conv1d)
    assert m_.in_channels == m.in_channels
    assert m_.out_channels == m.out_channels
    assert m_.kernel_size == m.kernel_size


def test_highway():
    m = Highway(10, 20)
    m_ = load(dump(m))
    assert isinstance(m_, Highway)
    assert m_.size == m.size
    assert m_.num_layers == m.num_layers


def test_language_model():
    m = LanguageModel(
        nn.Embedding(2, 3),
        nn.Embedding(4, 5),
        nn.ModuleList([nn.Conv1d(2, 3, 4), nn.Conv1d(3, 4, 5)]),
        Highway(3, 4),
        nn.LSTM(10, 20),
        nn.Linear(4, 5),
    )
    m_ = load(dump(m))
    assert isinstance(m_, LanguageModel)
    assert isinstance(m_.word_emb, nn.Embedding)
    assert isinstance(m_.char_emb, nn.Embedding)
    assert isinstance(m_.char_convs, nn.ModuleList)
    assert isinstance(m_.highway, Highway)
    assert isinstance(m_.lstm, nn.LSTM)
    assert isinstance(m_.output_layer, nn.Linear)
