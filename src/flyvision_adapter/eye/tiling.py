"""Cover a video frame with flyvis compound eyes.

A flyvis eye is 721 receptors on a hexagonal lattice (`extent`=15 rings), `kernel_size` pixels
apart, each averaging a `kernel_size`^2 pixel box (flyvis `BoxEye`). The receptor field is a
hexagon — pointy at top/bottom, flat vertical sides:

    {(dy, dx) : |dx| <= R,  |dy| <= R - |dx| / 2},   R = extent * kernel_size

so frames are tiled with a hexagonal tessellation (column pitch 2R, row pitch 1.5R, odd rows
shifted by R), not a square grid, which would leave the square corners unseen. `margin` receptor
rings are overlapped between neighbours, because receptors on an eye's rim lack the lattice
neighbours that motion detection (T4/T5) needs.

Pure numpy — no torch/flyvis — so the geometry is testable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GREY = 0.5  # flyvis's neutral luminance, used outside the frame


def receptor_offsets(extent: int = 15, kernel_size: int = 13) -> np.ndarray:
    """(n_hexals, 2) integer (dy, dx) offsets from the eye centre, in flyvis `BoxEye` order.

    Mirrors `BoxEye._receptor_centers` including its float->long truncation, so hexal i here is
    hexal i of the network input.
    """
    n, d = extent, kernel_size
    out = []
    for u in range(-n, n + 1):
        for v in range(max(-n, -n - u), min(n, n - u) + 1):
            out.append((d * (u + v / 2), d * v))
    return np.trunc(np.asarray(out)).astype(np.int64)


def hex_contains(dy: np.ndarray, dx: np.ndarray, radius: float) -> np.ndarray:
    """Whether offsets (dy, dx) fall inside the eye hexagon of the given radius."""
    adx = np.abs(dx)
    return (adx <= radius) & (np.abs(dy) <= radius - adx / 2)


@dataclass
class EyeTiling:
    frame_h: int
    frame_w: int
    extent: int = 15
    kernel_size: int = 13
    margin: int = 2
    centers: np.ndarray = field(init=False)  # (n_tiles, 2) int (cy, cx)

    def __post_init__(self) -> None:
        if not 0 <= self.margin < self.extent:
            raise ValueError("margin must be in [0, extent)")
        r = self.pitch_radius
        col, row = 2 * r, 1.5 * r
        cands = []
        for i in range(int(np.ceil(self.frame_h / row)) + 2):
            shift = r if i % 2 else 0.0
            for j in range(-1, int(np.ceil(self.frame_w / col)) + 2):
                cands.append((i * row, j * col + shift))
        cands = np.rint(np.asarray(cands)).astype(np.int64)
        # keep only eyes whose (effective) hexagon covers at least one frame pixel
        ys, xs = np.mgrid[0:self.frame_h:4, 0:self.frame_w:4]
        keep = [hex_contains(ys - cy, xs - cx, r).any() for cy, cx in cands]
        self.centers = cands[np.asarray(keep)]

    @property
    def radius(self) -> int:
        """Full receptor-field radius in pixels."""
        return self.extent * self.kernel_size

    @property
    def pitch_radius(self) -> float:
        """Radius used for the tessellation: the field minus the overlapped rim."""
        return (self.extent - self.margin) * self.kernel_size

    @property
    def window(self) -> int:
        """Side of the square crop handed to `BoxEye` (its `min_frame_size`)."""
        off = receptor_offsets(self.extent, self.kernel_size)
        return int((off.max(0) - off.min(0) + 1).max())

    @property
    def n_tiles(self) -> int:
        return len(self.centers)

    def crops(self, frames: np.ndarray) -> np.ndarray:
        """(T, H, W) frames -> (n_tiles, T, window, window) crops centred on each eye.

        The crop centre sits at index window // 2, where `BoxEye.hex_render` places the eye.
        """
        if frames.ndim == 2:
            frames = frames[None]
        w = self.window
        half = w // 2
        # rim eyes may be centred outside the frame; pad enough for the farthest one
        lo = half + max(0, -int(self.centers.min()))
        hi_y = half + max(0, int(self.centers[:, 0].max()) - (self.frame_h - 1))
        hi_x = half + max(0, int(self.centers[:, 1].max()) - (self.frame_w - 1))
        padded = np.pad(frames, ((0, 0), (lo, hi_y), (lo, hi_x)), constant_values=GREY)
        out = np.empty((self.n_tiles, frames.shape[0], w, w), dtype=frames.dtype)
        for k, (cy, cx) in enumerate(self.centers + lo - half):
            out[k] = padded[:, cy:cy + w, cx:cx + w]
        return out

    def hexal_pixels(self) -> np.ndarray:
        """(n_tiles, n_hexals, 2) frame (y, x) of every receptor of every eye."""
        off = receptor_offsets(self.extent, self.kernel_size)
        return self.centers[:, None, :] + off[None, :, :]

    def locate(self, x: float, y: float) -> tuple[int, int]:
        """(tile, hexal) of the receptor nearest to frame point (x, y), preferring the eye whose
        centre is closest (so the point is away from that eye's rim)."""
        tile = int(np.argmin(np.hypot(self.centers[:, 0] - y, self.centers[:, 1] - x)))
        px = self.hexal_pixels()[tile]
        hexal = int(np.argmin(np.hypot(px[:, 0] - y, px[:, 1] - x)))
        return tile, hexal

    def coverage(self, step: int = 2) -> float:
        """Fraction of frame pixels (sampled every `step`) inside some eye's pitch hexagon."""
        ys, xs = np.mgrid[0:self.frame_h:step, 0:self.frame_w:step]
        seen = np.zeros(ys.shape, dtype=bool)
        for cy, cx in self.centers:
            seen |= hex_contains(ys - cy, xs - cx, self.pitch_radius)
        return float(seen.mean())
