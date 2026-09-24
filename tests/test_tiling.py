import numpy as np
import pytest

from flyvision_adapter.eye.tiling import EyeTiling, hex_contains, receptor_offsets


def test_721_receptors_inside_their_hexagon():
    off = receptor_offsets(15, 13)
    assert off.shape == (721, 2)
    # truncation can move a receptor by <1 px, never out of the field
    assert hex_contains(off[:, 0], off[:, 1], 15 * 13 + 1).all()


def test_window_matches_boxeye_min_frame_size():
    assert EyeTiling(1080, 1920).window == 2 * 15 * 13 + 1


@pytest.mark.parametrize("size", [(1080, 1920), (1080, 1440)])
@pytest.mark.parametrize("k", [5, 9, 13])
def test_tiling_covers_whole_frame(size, k):
    t = EyeTiling(*size, kernel_size=k)
    assert t.coverage(step=4) == 1.0


def test_crop_is_centred_on_eye():
    t = EyeTiling(1080, 1920)
    frames = np.zeros((2, 1080, 1920), np.float32)
    cy, cx = t.centers[t.n_tiles // 2]
    frames[:, cy, cx] = 1.0
    crops = t.crops(frames)
    assert crops.shape == (t.n_tiles, 2, t.window, t.window)
    assert crops[t.n_tiles // 2, :, t.window // 2, t.window // 2].tolist() == [1.0, 1.0]


def test_out_of_frame_is_grey():
    t = EyeTiling(1080, 1920)
    crops = t.crops(np.zeros((1, 1080, 1920), np.float32))
    assert crops.max() == pytest.approx(0.5)  # corner eyes see padding


def test_locate_returns_nearest_receptor():
    t = EyeTiling(1080, 1920)
    tile, hexal = t.locate(x=700.0, y=400.0)
    py, px = t.hexal_pixels()[tile, hexal]
    assert abs(py - 400) <= 13 and abs(px - 700) <= 13
