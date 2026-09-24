"""Hyper-parameter sweep for every detector family on DEV clips only (disjoint from the test
clips of the v1/v2 benchmark), then pick each family's best configuration.

Run on noron:  .venv/bin/python scripts/tune_dev.py --out outputs/tune/v1
Writes dev_results.csv (family, config, clip, hits) — resumable per clip — and dev_best.json,
which `run_benchmark.py --config` turns into the test-set detectors.

Families and grids (every family gets a comparable search):
  framediff      blur {1.5, 3, 6} px, with/without DoG surround {10, 20, 40} px
  mog2           history {50, 150, 500} x varThreshold {8, 16, 32}
  receptor_diff  with/without small-target stage
  emd            tau_hp {0.1, 0.25, 0.5} x tau_delay {0.02, 0.05, 0.1} s x lag {1, 6, 12} x st
  flyvis         readout cells (each columnar type alone, T4, T5, T4+T5, ON, OFF, ON+OFF, all)
                 x baseline tau {0.25, 0.5, 1} s x lag {1, 6, 12} x per-type normalisation
                 (groups only) x st
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flyvision_adapter.data.clips import dev_starts, load_clip
from flyvision_adapter.data.labels import load_labels
from flyvision_adapter.data.video import list_videos
from flyvision_adapter.detectors.base import hex_readout, lag_max, make_hex_input
from flyvision_adapter.detectors.flyvis_motion import (MOTION_TYPES, columnar_types, combine,
                                                       load_network, simulate_types,
                                                       type_frame_maps, type_index)
from flyvision_adapter.detectors.hexdet import HexEMD, ReceptorDiff
from flyvision_adapter.detectors.pixel import MOG2, FrameDiff
from flyvision_adapter.detectors.small_target import FrameDiffST, centre_surround
from flyvision_adapter.eval.metrics import WARMUP
from flyvision_adapter.eye.render import HexRenderer
from flyvision_adapter.eye.tiling import EyeTiling

ROOT = Path("data/drone-tracking-datasets")
LAGS = (1, 6, 12)
FLYVIS_TAUS = (0.25, 0.5, 1.0)
GROUPS = {
    "T4": [f"T4{c}" for c in "abcd"],
    "T5": [f"T5{c}" for c in "abcd"],
    "T4T5": list(MOTION_TYPES),
    "ON": ["Mi1", "Tm3", "Mi4", "Mi9"] + [f"T4{c}" for c in "abcd"],
    "OFF": ["Tm1", "Tm2", "Tm4", "Tm9"] + [f"T5{c}" for c in "abcd"],
}
GROUPS["ON+OFF"] = GROUPS["ON"] + GROUPS["OFF"]


def hits(xy: np.ndarray, clip) -> dict:
    keep = np.arange(len(xy)) >= WARMUP
    vis = keep & clip.visible
    err = np.hypot(*(xy[vis] - clip.gt_xy[vis]).T)
    return {"hit25": float((err <= 25).mean()) if len(err) else np.nan,
            "hit50": float((err <= 50).mean()) if len(err) else np.nan,
            "n_visible": int(vis.sum())}


def row(family: str, config: dict, clip, xy: np.ndarray) -> dict:
    return {"family": family, "config": json.dumps(config, sort_keys=True), "clip": clip.key,
            **hits(xy, clip)}


def sweep_pixel(clip) -> list[dict]:
    rows = []
    for blur in (1.5, 3.0, 6.0):
        rows.append(row("framediff", {"blur": blur, "st": False}, clip, FrameDiff(blur)(clip).xy))
        for sur in (10.0, 20.0, 40.0):
            det = FrameDiffST(blur, sur)
            rows.append(row("framediff", {"blur": blur, "surround": sur, "st": True}, clip,
                            det(clip).xy))
    for hist, var in itertools.product((50, 150, 500), (8.0, 16.0, 32.0)):
        rows.append(row("mog2", {"history": hist, "var_threshold": var}, clip,
                        MOG2(hist, var)(clip).xy))
    return rows


def readout_rows(family: str, base: dict, score: torch.Tensor, clip, inp) -> list[dict]:
    """Score map with and without the small-target stage -> two rows."""
    out = []
    for st in (False, True):
        s = centre_surround(score, inp.tiling.extent) if st else score
        xy, _ = hex_readout(s, inp.tiling)
        out.append(row(family, {**base, "st": st}, clip, xy))
    return out


def sweep_hex(clip, inp) -> list[dict]:
    rows = readout_rows("receptor_diff", {}, ReceptorDiff().score(inp), clip, inp)
    for hp, delay in itertools.product((0.1, 0.25, 0.5), (0.02, 0.05, 0.1)):
        energy = HexEMD(hp, delay).energy(inp)
        for lag in LAGS:
            rows += readout_rows("emd", {"tau_hp": hp, "tau_delay": delay, "lag": lag},
                                 lag_max(energy, inp.last_step, lag), clip, inp)
        del energy
    return rows


def sweep_flyvis(clip, inp, network, types: list[str], index: torch.Tensor,
                 tile_batch: int = 8) -> list[dict]:
    # per (tau, lag): (T, tiles, n_types, 721) half-precision maps, filled one tile batch at a time
    maps = {k: [] for k in itertools.product(FLYVIS_TAUS, LAGS)}
    for b in range(0, inp.stim.shape[0], tile_batch):
        act = simulate_types(network, index, inp.stim[b:b + tile_batch])
        for tau, lag in maps:
            maps[(tau, lag)].append(type_frame_maps(act, inp.last_step, tau, lag).half().cpu())
        del act
        torch.cuda.empty_cache()
    pos = {t: i for i, t in enumerate(types)}
    sets = {t: [t] for t in types} | {g: m for g, m in GROUPS.items()}
    sets["all"] = types
    rows = []
    for (tau, lag), parts in maps.items():
        m = torch.cat(parts, dim=1).cuda()
        for name, members in sets.items():
            sub = m[:, :, [pos[t] for t in members]].float()
            for norm in ((False, True) if len(members) > 1 else (False,)):
                cfg = {"cell_types": members, "readout": name, "tau_baseline": tau,
                       "lag": lag, "normalize": norm}
                rows += readout_rows("flyvis", cfg, clip=clip, inp=inp,
                                     score=combine(sub, norm))
            del sub
        del m
        torch.cuda.empty_cache()
    return rows


def pick_best(df: pd.DataFrame) -> dict:
    """Best config per (family, st) by mean dev hit@25 over clips; hit@50 breaks ties."""
    df = df.copy()
    df["st"] = df.config.map(lambda c: json.loads(c)["st"] if "st" in json.loads(c) else False)
    mean = df.groupby(["family", "st", "config"])[["hit25", "hit50"]].mean().reset_index()
    best = {}
    for (fam, st), g in mean.groupby(["family", "st"]):
        top = g.sort_values(["hit25", "hit50"], ascending=False).iloc[0]
        name = fam + ("+st" if st else "")
        cfg = json.loads(top.config)
        cfg.pop("readout", None)
        best[name] = {"family": fam, "params": cfg,
                      "dev_hit25": round(float(top.hit25), 4),
                      "dev_hit50": round(float(top.hit50), 4)}
    # fixed-readout reference: T4/T5 as in v1/v2, only its time constants tuned
    t4t5 = mean[(mean.family == "flyvis")
                & mean.config.str.contains('"readout": "T4T5"')]
    for st, g in t4t5.groupby("st"):
        top = g.sort_values(["hit25", "hit50"], ascending=False).iloc[0]
        cfg = json.loads(top.config)
        cfg.pop("readout", None)
        best["flyvis_t4t5" + ("+st" if st else "")] = {
            "family": "flyvis", "params": cfg,
            "dev_hit25": round(float(top.hit25), 4), "dev_hit50": round(float(top.hit50), 4)}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tune/v1")
    ap.add_argument("--clips-per-video", type=int, default=3)
    ap.add_argument("--kernel-size", type=int, default=13)
    ap.add_argument("--videos", nargs="*", help="e.g. dataset3/cam1 (default: all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "dev_results.csv"
    done = set(pd.read_csv(results)["clip"]) if results.exists() else set()

    network = load_network()
    types = columnar_types(network)
    index = type_index(network, types)
    renderers: dict = {}
    t0 = time.perf_counter()
    for v in list_videos(ROOT):
        if args.videos and v.key not in args.videos:
            continue
        lab = load_labels(v.labels_path)
        for start in dev_starts(lab, v.n_frames, n=args.clips_per_video):
            if f"{v.key}@{start}" in done:
                continue
            clip = load_clip(v, lab, start)
            size = clip.frames.shape[1:]
            if size not in renderers:
                renderers[size] = HexRenderer(EyeTiling(*size, kernel_size=args.kernel_size))
            inp = make_hex_input(clip, renderers[size])
            rows = sweep_pixel(clip) + sweep_hex(clip, inp)
            rows += sweep_flyvis(clip, inp, network, types, index)
            pd.DataFrame(rows).to_csv(results, mode="a", header=not results.exists(), index=False)
            del inp
            torch.cuda.empty_cache()
            print(f"[{time.perf_counter() - t0:7.0f}s] {clip.key:24s} {len(rows)} configs",
                  flush=True)

    best = pick_best(pd.read_csv(results))
    (out / "dev_best.json").write_text(json.dumps(best, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ("dev_hit25", "dev_hit50")} | {"params": v["params"]}
                      for k, v in best.items()}, indent=1))


if __name__ == "__main__":
    main()
