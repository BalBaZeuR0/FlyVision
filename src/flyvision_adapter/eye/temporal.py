"""Map flyvis integration steps onto video frames.

flyvis integrates at a fixed `dt` (1/100 s by default) while cameras run at 21-60 fps, so each
video frame is held for however many steps fall inside its display interval (sample-and-hold,
which is what flyvis's own `SequenceDataset` resampling does).
"""

from __future__ import annotations

import numpy as np

DT = 1 / 100


def n_sim_steps(n_frames: int, fps: float, dt: float = DT) -> int:
    return int(np.floor(n_frames / fps / dt + 1e-9))


def sim_frame_indices(n_frames: int, fps: float, dt: float = DT) -> np.ndarray:
    """Video frame index shown at each simulation step."""
    t = np.arange(n_sim_steps(n_frames, fps, dt)) * dt
    return np.minimum(np.floor(t * fps + 1e-9).astype(np.int64), n_frames - 1)


def last_step_of_frame(n_frames: int, fps: float, dt: float = DT) -> np.ndarray:
    """For each video frame, the last simulation step showing it (-1 if none, when fps > 1/dt).

    Detector read-outs are taken at this step, so a detection for frame i uses only input up to
    frame i (causal, same as the classical baselines).
    """
    idx = sim_frame_indices(n_frames, fps, dt)
    out = np.full(n_frames, -1, dtype=np.int64)
    out[idx] = np.arange(len(idx))  # later steps overwrite earlier ones -> last step wins
    return out
