import torch
import torch.nn as nn
import torch.nn.functional as F


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
