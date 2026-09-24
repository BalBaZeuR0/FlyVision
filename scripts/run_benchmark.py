"""Run every detector on the same sampled clips and score them.

Run on noron:  .venv/bin/python scripts/run_benchmark.py --out outputs/bench/v1
Writes frames.csv (per frame x detector), summary_by_dataset.csv, summary_overall.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flyvision_adapter.data.labels import load_labels
from flyvision_adapter.data.video import iter_gray_frames, list_videos
from flyvision_adapter.detectors.base import Clip, make_hex_input
from flyvision_adapter.detectors.hexdet import HexEMD, ReceptorDiff
from flyvision_adapter.detectors.pixel import MOG2, FrameDiff
from flyvision_adapter.detectors.small_target import FrameDiffST, SmallTarget
from flyvision_adapter.eval.metrics import frame_table, summarize
from flyvision_adapter.eye.render import HexRenderer
from flyvision_adapter.eye.tiling import EyeTiling

ROOT = Path("data/drone-tracking-datasets")


def sample_clips(lab: pd.DataFrame, n_frames: int, length: int, n: int,
                 min_visible: float = 0.3) -> list[int]:
    """Deterministic clip starts spread over the whole flight, each with >= min_visible of its
    frames labelled (so both drone-present and, usually, drone-absent frames occur)."""
    vis = lab.set_index("frame").visible.reindex(range(n_frames), fill_value=False).to_numpy()
    starts = [s for s in range(0, n_frames - length, length)
              if vis[s:s + length].mean() >= min_visible]
    if len(starts) <= n:
        return starts
    return [starts[i] for i in np.linspace(0, len(starts) - 1, n).round().astype(int)]


def load_clip(v, lab: pd.DataFrame, start: int, length: int) -> Clip:
    frames = np.stack([f for _, f in iter_gray_frames(v, start, start + length)])
    rows = lab.set_index("frame").reindex(range(start, start + len(frames)))
    visible = rows.visible.astype("boolean").fillna(False).to_numpy(bool)
    gt = rows[["x", "y"]].fillna(0).to_numpy() * v.scale
    return Clip(f"{v.key}@{start}", frames, v.fps, gt, visible)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bench/v1")
    ap.add_argument("--clips-per-video", type=int, default=4)
    ap.add_argument("--clip-frames", type=int, default=150)
    ap.add_argument("--videos", nargs="*", help="e.g. dataset3/cam1 (default: all)")
    ap.add_argument("--kernel-size", type=int, default=13)
    ap.add_argument("--no-flyvis", action="store_true")
    ap.add_argument("--no-small-target", dest="small_target", action="store_false")
    ap.add_argument("--only", nargs="*", help="detector names to run (default: all)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    hex_dets = [ReceptorDiff(), HexEMD()]
    if not args.no_flyvis:
        from flyvision_adapter.detectors.flyvis_motion import FlyvisMotion
        hex_dets.append(FlyvisMotion())
    detectors = ([FrameDiff(), FrameDiffST(), MOG2()] + hex_dets
                 + ([SmallTarget(d) for d in hex_dets] if args.small_target else []))
    if args.only:
        detectors = [d for d in detectors if d.name in args.only]

    videos = [v for v in list_videos(ROOT) if not args.videos or v.key in args.videos]
    renderers: dict[tuple[int, int], HexRenderer] = {}
    # every finished clip is appended here, so a crash loses at most one clip and a rerun resumes
    partial = out / "frames_partial.csv"
    tables = [pd.read_csv(partial)] if partial.exists() else []
    done = set(tables[0]["clip"]) if tables else set()
    t_all = time.perf_counter()
    for v in videos:
        lab = load_labels(v.labels_path)
        for start in sample_clips(lab, v.n_frames, args.clip_frames, args.clips_per_video):
            if f"{v.key}@{start}" in done:
                continue
            clip = load_clip(v, lab, start, args.clip_frames)
            size = clip.frames.shape[1:]
            if size not in renderers:
                renderers[size] = HexRenderer(EyeTiling(*size, kernel_size=args.kernel_size))
            inp = make_hex_input(clip, renderers[size])
            for det in detectors:
                tables.append(frame_table(clip, det(clip, inp), det.name))
            del inp
            torch.cuda.empty_cache()
            last = pd.concat(tables[-len(detectors):])
            last.to_csv(partial, mode="a", header=not partial.exists(), index=False)
            hits = last[last.visible].groupby("detector").err.apply(lambda e: (e <= 25).mean())
            print(f"[{time.perf_counter() - t_all:7.0f}s] {clip.key:24s} hit@25 "
                  + " ".join(f"{k}={hits.get(k, float('nan')):.2f}" for k in hits.index),
                  flush=True)

    frames = pd.concat(tables, ignore_index=True)
    frames["dataset"] = frames["clip"].str.split("/").str[0]
    frames.to_csv(out / "frames.csv", index=False)
    by_ds = summarize(frames, ["dataset", "detector"])
    overall = summarize(frames, ["detector"])
    by_ds.to_csv(out / "summary_by_dataset.csv", index=False)
    overall.to_csv(out / "summary_overall.csv", index=False)
    pd.set_option("display.width", 160)
    print(by_ds.round(3).to_string(index=False))
    print(overall.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
