"""Classical pixel-domain baselines on the 1080p frames."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .base import Clip, Detections

BLUR_SIGMA = 3.0  # px; pools a few-pixel drone into one peak, same for every pixel baseline


def _peak(score: np.ndarray) -> tuple[np.ndarray, float]:
    score = cv2.GaussianBlur(score, (0, 0), BLUR_SIGMA)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return np.array([x, y], dtype=np.float64), float(score[y, x])


class FrameDiff:
    """|I_t - I_{t-1}|: the simplest motion detector."""

    name = "framediff"

    def __call__(self, clip: Clip, _inp=None) -> Detections:
        xy, peak = np.zeros((len(clip.frames), 2)), np.zeros(len(clip.frames))
        t0 = time.perf_counter()
        prev = clip.frames[0]
        for i, f in enumerate(clip.frames):
            xy[i], peak[i] = _peak(np.abs(f - prev))
            prev = f
        ms = 1000 * (time.perf_counter() - t0) / len(clip.frames)
        return Detections(xy, peak, ms)


class MOG2:
    """OpenCV Gaussian-mixture background subtraction — the standard static-camera baseline."""

    name = "mog2"

    def __init__(self, history: int = 150, var_threshold: float = 16.0):
        self.history, self.var_threshold = history, var_threshold

    def __call__(self, clip: Clip, _inp=None) -> Detections:
        sub = cv2.createBackgroundSubtractorMOG2(self.history, self.var_threshold,
                                                 detectShadows=False)
        xy, peak = np.zeros((len(clip.frames), 2)), np.zeros(len(clip.frames))
        t0 = time.perf_counter()
        for i, f in enumerate(clip.frames):
            fg = sub.apply((f * 255).astype(np.uint8)).astype(np.float32) / 255.0
            xy[i], peak[i] = _peak(fg)
        ms = 1000 * (time.perf_counter() - t0) / len(clip.frames)
        return Detections(xy, peak, ms)
