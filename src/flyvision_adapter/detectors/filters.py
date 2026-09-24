"""Causal temporal filters on (tiles, steps, ...) tensors."""

from __future__ import annotations

import torch

from ..eye.temporal import DT


def lowpass(x: torch.Tensor, tau: float, dt: float = DT) -> torch.Tensor:
    """First-order low-pass along dim 1 (steps), started at the first sample."""
    a = dt / tau
    out = torch.empty_like(x)
    y = x[:, 0]
    for s in range(x.shape[1]):
        y = y + a * (x[:, s] - y)
        out[:, s] = y
    return out
