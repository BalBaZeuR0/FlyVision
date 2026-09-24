"""Synthetic moving dot: every detector must find it. Hex detectors get a box-averaged receptor
input on CPU, so this needs torch but not flyvis."""

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flyvision_adapter.detectors.base import Clip, HexInput  # noqa: E402
from flyvision_adapter.detectors.hexdet import HexEMD, ReceptorDiff  # noqa: E402
from flyvision_adapter.detectors.pixel import MOG2, FrameDiff  # noqa: E402
from flyvision_adapter.detectors.small_target import FrameDiffST, SmallTarget  # noqa: E402
from flyvision_adapter.eval.metrics import WARMUP, frame_table  # noqa: E402
from flyvision_adapter.eye.temporal import last_step_of_frame, sim_frame_indices  # noqa: E402
from flyvision_adapter.eye.tiling import EyeTiling  # noqa: E402

H, W, T, FPS = 400, 600, 60, 30.0


def moving_dot_clip() -> Clip:
    rng = np.random.default_rng(0)
    frames = (np.full((T, H, W), 0.6, np.float32)
              + rng.normal(0, 0.01, (T, H, W)).astype(np.float32))
    xy = np.stack([np.linspace(150, 450, T), np.full(T, 200.0)], 1)
    for i, (x, y) in enumerate(xy):
        frames[i, int(y) - 6:int(y) + 6, int(x) - 6:int(x) + 6] = 0.1
    return Clip("synthetic@0", frames, FPS, xy, np.ones(T, bool))


def fake_hex_input(clip: Clip) -> HexInput:
    tiling = EyeTiling(H, W)
    px = tiling.hexal_pixels()
    ys = np.clip(px[..., 0], 0, H - 1)
    xs = np.clip(px[..., 1], 0, W - 1)
    # 13x13 box average then sample, like BoxEye (plain point samples would miss the dot)
    per_frame = np.stack([cv2.blur(f, (13, 13))[ys, xs] for f in clip.frames], 1)
    idx = sim_frame_indices(T, FPS)
    return HexInput(tiling, torch.as_tensor(per_frame[:, idx]), last_step_of_frame(T, FPS), 0.0)


@pytest.mark.parametrize("det", [FrameDiff(), MOG2(history=20), FrameDiffST()])
def test_pixel_detectors_find_dot(det):
    clip = moving_dot_clip()
    df = frame_table(clip, det(clip), det.name)
    assert (df.err <= 25).mean() > 0.9


@pytest.mark.parametrize("det", [ReceptorDiff(), HexEMD(), SmallTarget(HexEMD())])
def test_hex_detectors_find_dot(det):
    clip = moving_dot_clip()
    df = frame_table(clip, det(clip, fake_hex_input(clip)), det.name)
    assert len(df) == T - WARMUP
    assert (df.err <= 25).mean() > 0.9


def test_small_target_keeps_dot_next_to_wide_field_band():
    """A dot plus a large moving bright band: the centre-surround stage must not lose the dot.
    (Whether it beats plain detectors on real clouds is a benchmark question, not a unit test.)"""
    clip = moving_dot_clip()
    for i in range(T):
        x0 = 20 + 3 * i
        clip.frames[i, 250:390, x0:x0 + 200] += 0.3  # wide-field 'cloud' below the dot
    df = frame_table(clip, SmallTarget(HexEMD())(clip, fake_hex_input(clip)), "emd+st")
    assert (df.err <= 25).mean() > 0.8
