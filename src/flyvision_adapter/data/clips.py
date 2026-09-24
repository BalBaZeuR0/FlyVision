"""Clip sampling with a fixed dev/test split.

Clip starts lie on a grid of `length` frames, so any two distinct starts never overlap. The test
set is the original benchmark sample (`test_starts`, reported in v1/v2); the dev set, used for all
hyper-parameter choices, is drawn from the remaining grid cells of the same videos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .video import VideoInfo, iter_gray_frames
from ..detectors.base import Clip


def _candidates(lab: pd.DataFrame, n_frames: int, length: int, min_visible: float) -> list[int]:
    vis = lab.set_index("frame").visible.reindex(range(n_frames), fill_value=False).to_numpy()
    return [s for s in range(0, n_frames - length, length)
            if vis[s:s + length].mean() >= min_visible]


def _spread(starts: list[int], n: int) -> list[int]:
    if len(starts) <= n:
        return starts
    return [starts[i] for i in np.linspace(0, len(starts) - 1, n).round().astype(int)]


def test_starts(lab: pd.DataFrame, n_frames: int, length: int = 150, n: int = 4,
                min_visible: float = 0.3) -> list[int]:
    """Deterministic starts spread over the whole flight, each with >= min_visible labelled."""
    return _spread(_candidates(lab, n_frames, length, min_visible), n)


def dev_starts(lab: pd.DataFrame, n_frames: int, length: int = 150, n: int = 3,
               n_test: int = 4, min_visible: float = 0.3) -> list[int]:
    """Starts disjoint from `test_starts(..., n=n_test)`, also spread over the flight."""
    held_out = set(test_starts(lab, n_frames, length, n_test, min_visible))
    rest = [s for s in _candidates(lab, n_frames, length, min_visible) if s not in held_out]
    return _spread(rest, n)


def load_clip(v: VideoInfo, lab: pd.DataFrame, start: int, length: int = 150) -> Clip:
    frames = np.stack([f for _, f in iter_gray_frames(v, start, start + length)])
    rows = lab.set_index("frame").reindex(range(start, start + len(frames)))
    visible = rows.visible.astype("boolean").fillna(False).to_numpy(bool)
    gt = rows[["x", "y"]].fillna(0).to_numpy() * v.scale
    return Clip(f"{v.key}@{start}", frames, v.fps, gt, visible)
