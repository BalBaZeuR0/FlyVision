"""Neighbour tables on the flyvis hexal lattice, in `receptor_offsets` / BoxEye order.

Hexal i sits at axial coordinates (u, v) with pixel offset (dy, dx) = (d(u + v/2), d v). The three
lattice axes (and their negatives) are

    (1, 0)   straight down        (0, 1)   down-right        (-1, 1)   up-right

Missing neighbours at the rim point back at the hexal itself, so rim cells see no contrast /
no motion instead of wrapping around.
"""

from __future__ import annotations

import numpy as np

AXES = ((1, 0), (0, 1), (-1, 1))


def axial_coords(extent: int = 15) -> np.ndarray:
    n = extent
    return np.asarray([(u, v) for u in range(-n, n + 1)
                       for v in range(max(-n, -n - u), min(n, n - u) + 1)], dtype=np.int64)


def axis_neighbours(extent: int = 15) -> np.ndarray:
    """(3, n_hexals) index of the neighbour one step along each axis (self at the rim)."""
    uv = axial_coords(extent)
    where = {tuple(c): i for i, c in enumerate(uv)}
    out = np.empty((len(AXES), len(uv)), dtype=np.int64)
    for a, (du, dv) in enumerate(AXES):
        out[a] = [where.get((u + du, v + dv), i) for i, (u, v) in enumerate(uv)]
    return out


def ring_neighbours(extent: int = 15) -> np.ndarray:
    """(n_hexals, 7) index of each hexal and its six neighbours (self where missing)."""
    uv = axial_coords(extent)
    where = {tuple(c): i for i, c in enumerate(uv)}
    steps = [(0, 0)] + [s for du, dv in AXES for s in ((du, dv), (-du, -dv))]
    return np.asarray([[where.get((u + du, v + dv), i) for du, dv in steps]
                       for i, (u, v) in enumerate(uv)], dtype=np.int64)
