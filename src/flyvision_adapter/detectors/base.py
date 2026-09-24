"""Shared detector interface.

Every detector sees the same clip and returns, per video frame, one location (its strongest
response) and that response's strength. Pixel detectors work on the 1080p frames; hex detectors
work on the shared flyvis photoreceptor signal and their best hexal is mapped back to pixels.
All are causal: the output for frame i only uses frames <= i.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..eye.lattice import ring_neighbours
from ..eye.temporal import last_step_of_frame
from ..eye.tiling import EyeTiling


@dataclass
class Clip:
    key: str  # "dataset3/cam1@9406"
    frames: np.ndarray  # (T, H, W) float32 in [0, 1]
    fps: float
    gt_xy: np.ndarray  # (T, 2) label in 1080p-scaled pixels
    visible: np.ndarray  # (T,) bool


@dataclass
class HexInput:
    tiling: EyeTiling
    stim: torch.Tensor  # (tiles, steps, 721) receptor luminance, on the GPU
    last_step: np.ndarray  # (T,) simulation step read out for each frame
    render_ms: float  # total time to produce `stim`, charged to every hex detector
    # detector name -> (per-frame score, ms): lets a wrapper reuse its inner detector's work
    cache: dict = field(default_factory=dict)


@dataclass
class Detections:
    xy: np.ndarray  # (T, 2) predicted location, pixels
    peak: np.ndarray  # (T,) strength of that location (detector-specific units)
    ms_per_frame: float


def timed(fn):
    """Run fn() and return (result, elapsed ms), synchronising the GPU around it."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, 1000 * (time.perf_counter() - t0)


def lag_max(per_step: torch.Tensor, last_step: np.ndarray, lag: int) -> torch.Tensor:
    """(tiles, steps, H) -> (T, tiles, H): max over the `lag` steps ending at each frame's readout
    step, absorbing response latency without looking ahead."""
    out = []
    for s in last_step:
        s = int(s)
        out.append(per_step[:, max(0, s - lag + 1):s + 1].amax(dim=1))
    return torch.stack(out)


def hex_readout(score: torch.Tensor, tiling: EyeTiling) -> tuple[np.ndarray, np.ndarray]:
    """(T, tiles, 721) per-frame score -> best pixel location and its strength, after averaging
    each hexal with its 6 neighbours: a target between columns (or flanked by direction-selective
    responses) is pooled, and unlike a ring max the pooled peak stays on the centre column."""
    ring = torch.as_tensor(ring_neighbours(tiling.extent), device=score.device)
    score = score[..., ring].mean(dim=-1)
    flat = score.flatten(1)
    peak, idx = flat.max(dim=1)
    idx = idx.cpu().numpy()
    tile, hexal = np.divmod(idx, score.shape[-1])
    yx = tiling.hexal_pixels()[tile, hexal]
    return yx[:, ::-1].astype(np.float64), peak.cpu().numpy()


class HexDetector:
    name = "hex"
    lag = 6  # steps (60 ms at dt = 0.01)

    def score(self, inp: HexInput) -> torch.Tensor:  # (T, tiles, 721)
        raise NotImplementedError

    def cached_score(self, inp: HexInput) -> tuple[torch.Tensor, float]:
        if self.name not in inp.cache:
            inp.cache[self.name] = timed(lambda: self.score(inp))
        return inp.cache[self.name]

    def __call__(self, clip: Clip, inp: HexInput) -> Detections:
        score, ms = self.cached_score(inp)
        xy, peak = hex_readout(score, inp.tiling)
        return Detections(xy, peak, (ms + inp.render_ms) / len(clip.frames))


def make_hex_input(clip: Clip, renderer) -> HexInput:
    stim, ms = timed(lambda: renderer.render_clip(clip.frames, clip.fps)[:, :, 0])
    return HexInput(renderer.tiling, stim, last_step_of_frame(len(clip.frames), clip.fps), ms)
