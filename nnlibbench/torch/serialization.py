from typing import Any

import torch.nn as nn
from camel import Camel, CamelRegistry, PYTHON_TYPES

from . import Highway, LanguageModel

registry = CamelRegistry()


def dump(obj: Any) -> str:
    return Camel([PYTHON_TYPES, registry]).dump(obj)


def load(text: str) -> Any:
    return Camel([PYTHON_TYPES, registry]).load(text)


@registry.dumper(nn.Linear, 'linear', version=1)
def _dump_linear(m: nn.Linear) -> dict:
    return {
        'in': m.in_features,
        'out': m.out_features,
        'bias': m.bias is not None,
    }


@registry.loader('linear', version=1)
def _load_linear(data: dict, version: int) -> nn.Linear:
    return nn.Linear(data['in'], data['out'], bias=data['bias'])


@registry.dumper(nn.Embedding, 'embedding', version=1)
def _dump_embedding(m: nn.Embedding) -> dict:
    return {
        'num': m.num_embeddings,
        'dim': m.embedding_dim,
        'pad_idx': m.padding_idx,
    }


@registry.loader('embedding', version=1)
def _load_embedding(data: dict, version: int) -> nn.Embedding:
    return nn.Embedding(data['num'], data['dim'], padding_idx=data['pad_idx'])


@registry.dumper(nn.LSTM, 'lstm', version=1)
def _dump_lstm(m: nn.LSTM) -> dict:
    return {
        'input': m.input_size,
        'hidden': m.hidden_size,
        'layers': m.num_layers,
        'dropout': m.dropout,
        'batch_first': m.batch_first,
        'bidir': m.bidirectional,
    }


@registry.loader('lstm', version=1)
def _load_lstm(data: dict, version: int) -> nn.LSTM:
    return nn.LSTM(
        data['input'],
        data['hidden'],
        num_layers=data['layers'],
        dropout=data['dropout'],
        batch_first=data['batch_first'],
        bidirectional=data['bidir'],
    )


@registry.dumper(nn.ModuleList, 'module_list', version=1)
def _dump_module_list(ms: nn.ModuleList) -> list:
    return [m for m in ms]


@registry.loader('module_list', version=1)
def _load_module_list(data: list, version: int) -> nn.ModuleList:
    return nn.ModuleList(data)


@registry.dumper(nn.Conv1d, 'conv1d', version=1)
def _dump_conv1d(m: nn.Conv1d) -> dict:
    return {
        'in': m.in_channels,
        'out': m.out_channels,
        'kernel_size': m.kernel_size,
    }


@registry.loader('conv1d', version=1)
def _load_conv1d(data: dict, version: int) -> nn.Conv1d:
    return nn.Conv1d(data['in'], data['out'], data['kernel_size'])


@registry.dumper(Highway, 'highway', version=1)
def _dump_highway(m: Highway) -> dict:
    return {'size': m.size, 'layers': m.num_layers}


@registry.loader('highway', version=1)
def _load_highway(data: dict, version: int) -> Highway:
    return Highway(data['size'], data['layers'])


@registry.dumper(LanguageModel, 'lm', version=1)
def _dump_lm(m: LanguageModel) -> dict:
    return {
        'word_emb': m.word_emb,
        'char_emb': m.char_emb,
        'char_convs': m.char_convs,
        'highway': m.highway,
        'lstm': m.lstm,
        'output_layer': m.output_layer,
    }


@registry.loader('lm', version=1)
def _load_lm(data: dict, version: int) -> LanguageModel:
    return LanguageModel(
        data['word_emb'],
        data['char_emb'],
        data['char_convs'],
        data['highway'],
        data['lstm'],
        data['output_layer'],
    )
