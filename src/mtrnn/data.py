"""Helpers for getting recordings into the shape the model expects."""

from __future__ import annotations

import torch
from torch import Tensor


def pad(seqs: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Right-pad variable-length sequences by holding the last frame.

    Returns (B, T, D) targets and a (B, T) float mask of the real steps. Pass the
    mask to :func:`~mtrnn.train.train` so the padding does not enter the loss.
    """
    length = max(len(s) for s in seqs)
    targets = torch.stack(
        [torch.cat([s, s[-1:].expand(length - len(s), -1)]) for s in seqs]
    )
    mask = torch.stack([torch.arange(length) < len(s) for s in seqs]).to(targets.dtype)
    return targets, mask
