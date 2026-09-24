"""Detectors on the shared flyvis photoreceptor signal (tiles, steps, 721)."""

from __future__ import annotations

import torch

from ..eye.lattice import axis_neighbours, ring_neighbours
from .base import HexDetector, HexInput, lag_max
from .filters import lowpass


class ReceptorDiff(HexDetector):
    """|R(frame i) - R(frame i-1)| on the receptors: frame differencing after the fly-eye
    sampling. Separates 'the eye's optics' from 'the eye's circuit'."""

    name = "receptor_diff"

    def score(self, inp: HexInput) -> torch.Tensor:
        s = torch.as_tensor(inp.last_step, device=inp.stim.device)
        cur = inp.stim[:, s]
        prev = inp.stim[:, torch.cat([s[:1], s[:-1]])]
        return (cur - prev).abs().permute(1, 0, 2)


class HexEMD(HexDetector):
    """Lightweight fly-inspired motion detector: temporal high-pass -> centre-surround ->
    Hassenstein-Reichardt correlators along the three lattice axes (opponent by construction),
    combined into direction-invariant motion energy. Hand-set, untuned time constants."""

    name = "emd"

    def __init__(self, tau_hp: float = 0.25, tau_delay: float = 0.05):
        self.tau_hp, self.tau_delay = tau_hp, tau_delay

    def score(self, inp: HexInput) -> torch.Tensor:
        x = inp.stim
        x = x - lowpass(x, self.tau_hp)
        ring = torch.as_tensor(ring_neighbours(inp.tiling.extent), device=x.device)
        c = x - x[..., ring[:, 1:]].mean(dim=-1)
        d = lowpass(c, self.tau_delay)
        energy = torch.zeros_like(c)
        for nb in torch.as_tensor(axis_neighbours(inp.tiling.extent), device=x.device):
            r = d * c[..., nb] - c * d[..., nb]
            energy += r * r
        return lag_max(energy.sqrt(), inp.last_step, self.lag)
