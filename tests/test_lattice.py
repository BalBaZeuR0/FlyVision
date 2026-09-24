import numpy as np

from flyvision_adapter.eye.lattice import axial_coords, axis_neighbours, ring_neighbours
from flyvision_adapter.eye.tiling import receptor_offsets


def test_axial_order_matches_receptor_offsets():
    uv = axial_coords(15)
    d = 13
    expect = np.trunc(np.stack([d * (uv[:, 0] + uv[:, 1] / 2), d * uv[:, 1]], 1))
    assert (expect == receptor_offsets(15, d)).all()


def test_axis_neighbours_are_one_lattice_step_away():
    off = receptor_offsets(15, 13).astype(float)
    nb = axis_neighbours(15)
    for a in range(3):
        moved = nb[a] != np.arange(721)
        dist = np.hypot(*(off[nb[a][moved]] - off[moved]).T)
        assert (dist >= 12).all() and (dist <= 1.2 * 13).all()


def test_centre_has_six_distinct_neighbours_rim_pads_self():
    ring = ring_neighbours(15)
    centre = int(np.flatnonzero((axial_coords(15) == 0).all(1))[0])
    assert len(set(ring[centre])) == 7
    assert (ring == np.arange(721)[:, None]).any(1).all()
