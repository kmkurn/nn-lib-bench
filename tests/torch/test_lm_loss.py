import pytest
import torch

from nnlibbench.torch import LMLoss


def test_forward():
    torch.random.manual_seed(12345)

    bsz, slen, n_words = 2, 3, 4
    loss_fn = LMLoss()
    scores = torch.randn(bsz, slen, n_words, requires_grad=True)
    targets = torch.randint(1, n_words, (bsz, slen), dtype=torch.long)
    loss = loss_fn(scores, targets)
    loss.item()  # assert scalar
    loss.backward()  # assert no error


def test_forward_padded():
    torch.random.manual_seed(12345)

    bsz, slen, n_words = 2, 3, 4
    loss_fn = LMLoss(padding_idx=0)

    scores = torch.randn(bsz, slen, n_words)
    targets = torch.randint(1, n_words, (bsz, slen), dtype=torch.long)
    unpadded_loss = loss_fn(scores, targets)

    scores = torch.cat([scores, torch.randn(bsz, 1, n_words)], dim=1)
    targets = torch.cat([targets, torch.zeros(bsz, 1, dtype=torch.long)], dim=1)
    padded_loss = loss_fn(scores, targets)

    assert unpadded_loss.item() == pytest.approx(padded_loss.item())
