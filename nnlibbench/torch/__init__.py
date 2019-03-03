from typing import List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO separate into modules


class Highway(nn.Module):
    def __init__(self, size: int, num_layers: int) -> None:
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    @property
    def size(self) -> int:
        return self.linears[0].in_features

    @property
    def num_layers(self) -> int:
        return len(self.linears)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for gate, linear in zip(self.gates, self.linears):
            t = torch.sigmoid(gate(inputs))
            inputs = t * F.relu(linear(inputs)) + (1 - t) * inputs
        return inputs


class LanguageModel(nn.Module):
    def __init__(
            self,
            word_emb: nn.Embedding,
            char_emb: nn.Embedding,
            char_convs: nn.ModuleList,
            highway: Highway,
            lstm: nn.LSTM,
            output_layer: nn.Linear,
    ) -> None:
        super().__init__()
        self.word_emb = word_emb
        self.char_emb = char_emb
        self.char_convs = char_convs
        self.highway = highway
        self.lstm = lstm
        self.output_layer = output_layer

    @property
    def num_words(self) -> int:
        return self.word_emb.num_embeddings

    @property
    def num_chars(self) -> int:
        return self.char_emb.num_embeddings

    def forward(self, inputs: Mapping[str, torch.LongTensor]) -> torch.Tensor:
        words = inputs['words']
        chars = inputs['chars']

        assert words.dim() == 2
        assert chars.dim() == 3
        assert chars.shape[:2] == words.shape
        # words shape: (bsz, slen)
        # chars shape: (bsz, slen, wlen)

        bsz, slen, wlen = chars.shape

        # shape: (bsz, slen, wlen, cembsz)
        c_embd = self.char_emb(chars)
        # shape: (bsz * slen, wlen, cembsz)
        c_embd = c_embd.view(bsz * slen, wlen, -1)
        # shape: (bsz * slen, cembsz, wlen)
        c_embd = torch.transpose(c_embd, 1, 2)

        conv_res = []
        for conv in self.char_convs:
            # shape: (bsz * slen, n, wlen + w - 1)
            res = conv(c_embd)
            # shape: (bsz * slen, n)
            res, _ = torch.max(res, -1)
            conv_res.append(torch.tanh(res))

        # shape: (bsz * slen, sum(num_filters))
        c_inputs = torch.cat(conv_res, -1)
        # shape: (bsz, slen, sum(num_filters))
        c_inputs = c_inputs.view(bsz, slen, -1)

        # shape: (bsz, slen, wembsz)
        w_embd = self.word_emb(words)
        # shape: (bsz, slen, inputsz)
        inputs = torch.cat((w_embd, c_inputs), -1)
        # shape: (bsz, slen, inputsz)
        inputs = self.highway(inputs)

        # shape: (bsz, slen, lstmsz)
        outputs, _ = self.lstm(inputs)
        # shape: (bsz, slen, lstmsz)
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        # shape: (bsz, slen, num_words)
        outputs = self.output_layer(outputs)

        return outputs


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


class LMLoss(nn.Module):
    def __init__(self, padding_idx: int = -100) -> None:
        super().__init__()
        self._padding_idx = padding_idx

    def forward(self, scores: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        assert scores.dim() == 3
        assert targets.shape == scores.shape[:2]
        # scores shape: (bsz, slen, num_words)
        # targets shape: (bsz, slen)

        # shape: (bsz * slen, num_words)
        scores = scores.view(-1, scores.size(-1))
        # shape: (bsz * slen,)
        targets = targets.view(-1)

        return F.cross_entropy(scores, targets, ignore_index=self._padding_idx)
