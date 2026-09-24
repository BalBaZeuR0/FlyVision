"""Video discovery and grayscale frame reading, with every camera rescaled to a common height.

Resolutions differ across cameras (1440x1080, 1920x1080, 3840x2160); all detectors see frames
rescaled to `TARGET_HEIGHT` so pixel scales — and hence the fly-eye tiling — are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

TARGET_HEIGHT = 1080


@dataclass(frozen=True)
class VideoInfo:
    dataset: str
    cam: str
    path: Path
    labels_path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def scale(self) -> float:
        """Factor mapping native pixel coordinates to the rescaled frame."""
        return TARGET_HEIGHT / self.height

    @property
    def scaled_size(self) -> tuple[int, int]:
        """(height, width) of frames returned by `iter_gray_frames`."""
        return TARGET_HEIGHT, round(self.width * self.scale)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.cam}"


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    try:
        info = VideoInfo(
            dataset=path.parts[-3],
            cam=path.stem,
            path=path,
            labels_path=path.parents[1] / "detections" / f"{path.stem}.txt",
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()
    return info


def list_videos(root: str | Path, datasets: tuple[str, ...] = ("dataset1", "dataset2",
                                                              "dataset3", "dataset4")
                ) -> list[VideoInfo]:
    root = Path(root)
    return [probe(p) for d in datasets for p in sorted((root / d).glob("cam*/cam*.mp4"))]


def iter_gray_frames(info: VideoInfo, start: int = 0, stop: int | None = None
                     ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(frame_index, frame)` with frame float32 in [0, 1], shape `info.scaled_size`."""
    stop = info.n_frames if stop is None else min(stop, info.n_frames)
    h, w = info.scaled_size
    cap = cv2.VideoCapture(str(info.path))
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for idx in range(start, stop):
            ok, bgr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if gray.shape != (h, w):
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
            yield idx, gray.astype(np.float32) / 255.0
    finally:
        cap.release()
