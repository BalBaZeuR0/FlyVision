import numpy as np

from flyvision_adapter.eye.temporal import last_step_of_frame, n_sim_steps, sim_frame_indices


def test_30fps_holds_each_frame_about_3_steps():
    idx = sim_frame_indices(30, 30.0)
    assert len(idx) == n_sim_steps(30, 30.0) == 100
    counts = np.bincount(idx)
    assert set(counts.tolist()) <= {3, 4} and counts.sum() == 100


def test_indices_monotonic_and_cover_all_frames():
    for fps in (21.24, 25.0, 29.97, 50.0, 59.94):
        idx = sim_frame_indices(600, fps)
        assert (np.diff(idx) >= 0).all()
        assert set(idx.tolist()) == set(range(idx.max() + 1))


def test_last_step_is_causal():
    last = last_step_of_frame(60, 59.94)
    idx = sim_frame_indices(60, 59.94)
    for f, s in enumerate(last):
        if s >= 0:
            assert idx[s] == f and (s + 1 == len(idx) or idx[s + 1] > f)
