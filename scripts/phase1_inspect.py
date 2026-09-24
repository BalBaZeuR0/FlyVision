"""Phase 1: validate the data pipeline end to end on noron and pick the eye scale.

1. every video: resolution, fps, frames, labelled fraction          -> outputs/phase1/videos.csv
2. our receptor geometry == flyvis BoxEye's, and hexal i of the input drives column i of the
   network (L1 impulse response)                                     -> printed + summary.json
3. drone crops around the labels, to eyeball apparent size           -> outputs/phase1/crops.png
4. a first signal check per eye scale (kernel_size): does T4/T5 motion energy peak at the
   labelled drone? Compared against raw receptor frame-difference.   -> outputs/phase1/summary.json

Run on noron:  .venv/bin/python scripts/phase1_inspect.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flyvision_adapter.data.labels import load_labels
from flyvision_adapter.data.video import iter_gray_frames, list_videos
from flyvision_adapter.eye.render import HexRenderer
from flyvision_adapter.eye.temporal import DT, last_step_of_frame
from flyvision_adapter.eye.tiling import EyeTiling, receptor_offsets

ROOT = Path("data/drone-tracking-datasets")
OUT = Path("outputs/phase1")
CLIPS = ["dataset3/cam1", "dataset4/cam1", "dataset1/cam0"]  # easy..hard, all 1080p ~30fps
CLIP_FRAMES = 90
KERNEL_SIZES = (13, 9, 5)
TILE_BATCH = 8
MOTION_TYPES = [f"T{a}{b}" for a in (4, 5) for b in "abcd"]


def video_table(videos) -> pd.DataFrame:
    rows = []
    for v in videos:
        lab = load_labels(v.labels_path)
        rows.append({"video": v.key, "w": v.width, "h": v.height, "fps": round(v.fps, 2),
                     "frames": v.n_frames, "label_rows": len(lab),
                     "visible_frac": round(float(lab.visible.mean()), 3)})
    return pd.DataFrame(rows)


def check_geometry(network) -> dict:
    from flyvis.datasets.rendering import BoxEye
    from flyvis.utils.activity_utils import LayerActivity

    eye = BoxEye(15, 13)
    same_order = bool((eye.receptor_centers.cpu().numpy() == receptor_offsets(15, 13)).all())

    # light one receptor at a time; the retinotopic L1 column that reacts most should be the same
    hits = []
    for i in (0, 100, 360, 555, 720):
        movie = torch.full((1, 30, 1, 721), 0.5)
        movie[0, 10:, 0, i] = 1.0
        with torch.no_grad():
            state = network.fade_in_state(1.0, DT, movie[:, 0])
            resp = network.simulate(movie, DT, initial_state=state)
        l1 = LayerActivity(resp.cpu(), network.connectome, keepref=True)["L1"][0]  # (T, 721)
        change = (l1[10:] - l1[:10].mean(0)).abs().max(0).values
        hits.append(int(change.argmax()) == i)
    return {"receptor_order_matches_boxeye": same_order, "impulse_hits_own_column": hits}


def drone_crops(videos, per_video: int = 4, half: int = 32) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(videos), per_video, figsize=(per_video * 1.6, len(videos) * 1.6))
    rng = np.random.default_rng(0)
    for r, v in enumerate(videos):
        lab = load_labels(v.labels_path)
        vis = lab[lab.visible & (lab.frame < v.n_frames)]
        picks = np.sort(rng.choice(vis.frame.to_numpy(), per_video, replace=False))
        for c, f in enumerate(picks):
            _, frame = next(iter_gray_frames(v, f, f + 1))
            row = vis[vis.frame == f].iloc[0]
            x, y = row.x * v.scale, row.y * v.scale
            pad = np.pad(frame, half, constant_values=0.5)
            crop = pad[int(y):int(y) + 2 * half, int(x):int(x) + 2 * half]
            ax = axes[r, c]
            ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(v.key, fontsize=6)
    fig.suptitle(f"{2 * half}px crops centred on the label (1080p scale)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "crops.png", dpi=150)


def first_visible_run(lab: pd.DataFrame, n: int, n_frames: int) -> int:
    vis = lab.set_index("frame").visible.reindex(range(n_frames), fill_value=False).to_numpy()
    run = np.convolve(vis.astype(int), np.ones(n, int), "valid")
    ok = np.flatnonzero(run == n)
    if not len(ok):
        raise ValueError("no fully labelled run")
    return int(ok[len(ok) // 2])  # middle of the flight, not take-off


def percentile_of(score: np.ndarray, value: float) -> float:
    return float((score < value).mean() * 100)


def signal_check(network, v, k: int) -> dict:
    from flyvis.utils.activity_utils import LayerActivity

    lab = load_labels(v.labels_path)
    start = first_visible_run(lab, CLIP_FRAMES, v.n_frames)
    frames = np.stack([f for _, f in iter_gray_frames(v, start, start + CLIP_FRAMES)])
    tiling = EyeTiling(*frames.shape[1:], kernel_size=k)
    renderer = HexRenderer(tiling)

    t0 = time.perf_counter()
    stim = renderer.render_clip(frames, v.fps)  # (tiles, steps, 1, 721)
    energy = []
    for b in range(0, tiling.n_tiles, TILE_BATCH):
        chunk = stim[b:b + TILE_BATCH]
        with torch.no_grad():
            state = network.fade_in_state(1.0, DT, chunk[:, 0])
            resp = network.simulate(chunk, DT, initial_state=state)
        la = LayerActivity(resp.cpu(), network.connectome, keepref=True)
        energy.append(sum(la[t].clamp(min=0) for t in MOTION_TYPES))  # (b, steps, 721)
        del resp, la
    torch.cuda.synchronize()
    sim_s = time.perf_counter() - t0
    energy = torch.cat(energy).numpy()  # (tiles, steps, 721)
    receptors = stim[:, :, 0].cpu().numpy()

    last = last_step_of_frame(len(frames), v.fps)
    pct_t45, pct_diff = [], []
    for i in range(10, len(frames)):  # skip onset transient
        s = last[i]
        row = lab[lab.frame == start + i].iloc[0]
        tile, hexal = tiling.locate(row.x * v.scale, row.y * v.scale)
        e = energy[:, s]
        d = np.abs(receptors[:, s] - receptors[:, last[i - 1]])
        pct_t45.append(percentile_of(e.ravel(), e[tile, hexal]))
        pct_diff.append(percentile_of(d.ravel(), d[tile, hexal]))
    return {
        "video": v.key, "kernel_size": k, "tiles": tiling.n_tiles, "clip_start": start,
        "sim_seconds": round(sim_s, 2),
        "ms_per_video_frame": round(1000 * sim_s / len(frames), 1),
        # percentile of the labelled drone's receptor among all receptors of the frame (100 = top)
        "drone_percentile_T4T5_median": round(float(np.median(pct_t45)), 2),
        "drone_percentile_framediff_median": round(float(np.median(pct_diff)), 2),
    }


def main() -> None:
    import flyvis

    OUT.mkdir(parents=True, exist_ok=True)
    videos = list_videos(ROOT)
    table = video_table(videos)
    table.to_csv(OUT / "videos.csv", index=False)
    print(table.to_string(index=False))

    network = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000").init_network()
    summary = {"geometry": check_geometry(network)}
    print(json.dumps(summary["geometry"]))

    drone_crops(videos)

    by_key = {v.key: v for v in videos}
    summary["signal"] = []
    for key in CLIPS:
        for k in KERNEL_SIZES:
            res = signal_check(network, by_key[key], k)
            print(json.dumps(res))
            summary["signal"].append(res)
            torch.cuda.empty_cache()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
