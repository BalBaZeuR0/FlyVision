"""Small-target stage (STMD / LC-like): centre minus surround on a motion map.

Flies single out small moving objects with lobula circuits (e.g. LC11, STMDs) that are excited by
local motion and inhibited by motion in the surround, so wide-field motion (clouds, swaying
foliage) cancels while a small target survives. flyvis stops at T4/T5 and has no such stage; this
adds the same, parameter-light stage on top of any motion map, pixel or hexal:

    out = relu(centre - surround)
    centre   = mean over the hexal and its 6 neighbours
    surround = mean over rings R_IN..R_OUT (hex) / a wider Gaussian (pixel)
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from ..eye.lattice import annulus_neighbours, ring_neighbours
from .base import Clip, Detections, HexDetector, HexInput, timed
from .pixel import BLUR_SIGMA, FrameDiff

R_IN, R_OUT = 2, 4  # rings: 26-52 px at k=13, i.e. about 1-3 drone widths
SURROUND_SIGMA = 20.0  # px, Gaussian matching the hex annulus (26-52 px) at k=13


def centre_surround(score: torch.Tensor, extent: int) -> torch.Tensor:
    """(..., n_hexals) -> same shape, relu(centre - surround)."""
    ring = torch.as_tensor(ring_neighbours(extent), device=score.device)
    idx, mask = annulus_neighbours(extent, R_IN, R_OUT)
    idx = torch.as_tensor(idx, device=score.device)
    mask = torch.as_tensor(mask, device=score.device, dtype=score.dtype)
    out = torch.empty_like(score)
    for i in range(0, score.shape[0], 16):  # chunked: gathering the annulus is ~50x the input
        s = score[i:i + 16]
        centre = s[..., ring].mean(dim=-1)
        surround = (s[..., idx] * mask).sum(dim=-1) / mask.sum(dim=-1)
        out[i:i + 16] = (centre - surround).clamp(min=0)
    return out


class SmallTarget(HexDetector):
    """Wrap a hex detector: its per-frame motion map goes through `centre_surround`."""

    def __init__(self, inner: HexDetector):
        self.inner = inner
        self.name = f"{inner.name}+st"

    def score(self, inp: HexInput) -> torch.Tensor:
        return centre_surround(self.inner.score(inp), inp.tiling.extent)

    def cached_score(self, inp: HexInput) -> tuple[torch.Tensor, float]:
        # reuse the inner map if it was already computed for this clip, but still charge its time
        inner, inner_ms = self.inner.cached_score(inp)
        out, ms = timed(lambda: centre_surround(inner, inp.tiling.extent))
        return out, inner_ms + ms


class FrameDiffST(FrameDiff):
    """Pixel control: frame difference through a difference-of-Gaussians centre-surround, so any
    gain of the fly models' small-target stage can be told apart from surround suppression alone."""

    name = "framediff+st"

    def __init__(self, blur: float = BLUR_SIGMA, surround: float = SURROUND_SIGMA):
        super().__init__(blur)
        self.surround = surround

    def __call__(self, clip: Clip, _inp=None) -> Detections:
        xy, peak = np.zeros((len(clip.frames), 2)), np.zeros(len(clip.frames))
        t0 = time.perf_counter()
        prev = clip.frames[0]
        for i, f in enumerate(clip.frames):
            d = np.abs(f - prev)
            prev = f
            dog = np.maximum(cv2.GaussianBlur(d, (0, 0), self.blur)
                             - cv2.GaussianBlur(d, (0, 0), self.surround), 0)
            y, x = np.unravel_index(int(np.argmax(dog)), dog.shape)
            xy[i], peak[i] = (x, y), dog[y, x]
        ms = 1000 * (time.perf_counter() - t0) / len(clip.frames)
        return Detections(xy, peak, ms)
