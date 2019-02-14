import pytest
import torch

from nnlibbench.torch import create_lm


@pytest.fixture
def setup_rng():
    seed = 12345
    torch.random.manual_seed(seed)


class TestForward:
    def test_ok(self, setup_rng):
        model = create_lm(10, 5)
        bsz, slen, wlen = 2, 3, 10
        words = torch.randint(1, model.num_words, (bsz, slen), dtype=torch.long)
        chars = torch.randint(1, model.num_chars, (bsz, slen, wlen), dtype=torch.long)
        targets = torch.randint(1, model.num_words, (bsz, slen), dtype=torch.long)

        loss = model(words, chars, targets)
        loss.item()  # no error raised bc loss is a scalar
        loss.backward()  # no error raised when computing gradients

    def test_padded(self, setup_rng):
        model = create_lm(4, 4, padding_idx=0, filter_widths=[1, 2])
        model.eval()

        words = torch.tensor([[1, 2, 3]], dtype=torch.long)
        chars = torch.tensor([[[1, 2], [2, 3], [3, 1]]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 1]], dtype=torch.long)
        unpadded_loss = model(words, chars, targets)

        words = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
        chars = torch.tensor([[[1, 2], [2, 3], [3, 1], [0, 0]]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 1, 0]], dtype=torch.long)
        padded_loss = model(words, chars, targets)

        assert unpadded_loss.item() == pytest.approx(padded_loss.item())
