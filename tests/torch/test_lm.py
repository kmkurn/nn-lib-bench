import torch

from nnlibbench.torch import create_lm


def test_forward():
    torch.random.manual_seed(12345)

    model = create_lm(10, 5)
    bsz, slen, wlen = 2, 3, 10
    words = torch.randint(1, model.num_words, (bsz, slen), dtype=torch.long)
    chars = torch.randint(1, model.num_chars, (bsz, slen, wlen), dtype=torch.long)

    scores = model(words, chars)
    assert torch.is_tensor(scores)
    assert scores.shape == (bsz, slen, model.num_words)
    scores.sum().backward()  # assert no error
