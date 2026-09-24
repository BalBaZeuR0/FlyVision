"""Frames -> flyvis photoreceptor input, via flyvis's own `BoxEye` on every eye of a tiling.

Every detector (EMD and flyvis alike) consumes this same (n_tiles, T, 1, 721) receptor signal,
so the comparison isolates the circuit, not the front-end sampling.
"""

from __future__ import annotations

import numpy as np
import torch

from .temporal import DT, sim_frame_indices
from .tiling import EyeTiling


class HexRenderer:
    def __init__(self, tiling: EyeTiling):
        from flyvis.datasets.rendering import BoxEye

        self.tiling = tiling
        self.eye = BoxEye(extent=tiling.extent, kernel_size=tiling.kernel_size)
        if int(self.eye.min_frame_size.max()) != tiling.window:
            raise ValueError("tiling window does not match BoxEye.min_frame_size")

    @torch.no_grad()
    def render(self, frames: np.ndarray, chunk: int = 8) -> torch.Tensor:
        """(T, H, W) video frames in [0, 1] -> (n_tiles, T, 1, n_hexals) receptor input.

        Frames go through in chunks: the crops (n_tiles x window^2 per frame, ~21 MB at 1080p)
        are the memory hog, the 721-hexal output is tiny.
        """
        dev = torch.get_default_device()
        out = [self.eye(torch.from_numpy(self.tiling.crops(frames[i:i + chunk])).float().to(dev))
               for i in range(0, len(frames), chunk)]
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def render_clip(self, frames: np.ndarray, fps: float, dt: float = DT) -> torch.Tensor:
        """Render a clip and resample it to simulation steps (sample-and-hold).

        Returns (n_tiles, n_steps, 1, n_hexals), ready for `network.simulate(..., dt)`.
        """
        per_frame = self.render(frames)
        idx = torch.as_tensor(sim_frame_indices(len(frames), fps, dt), device=per_frame.device)
        return per_frame[:, idx]
