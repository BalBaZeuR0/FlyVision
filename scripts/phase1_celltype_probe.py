"""Phase 1b: which flyvis cell types single out the drone?

phase1_inspect's T4/T5 score (sum of rectified raw activity) left the drone at the ~30-50th
percentile while plain receptor frame-difference put it at ~99.9. Candidate reasons, each tested
here as a separate score variant, for all 65 cell types:
  dev   |activity - that cell's temporal median over the clip|   (removes tonic baseline)
  lag   max of `dev` over the last LAG steps                       (response latency)
  ring  max of `lag` over the hexal and its 6 neighbours           (direction-selective cells
                                                                    fire beside the target)
Score = percentile of the labelled drone's hexal among all hexals of all eyes in that frame,
median over the clip (100 = drone is the single strongest spot).

Run on noron:  .venv/bin/python scripts/phase1_celltype_probe.py [video] [kernel_size]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flyvision_adapter.data.labels import load_labels
from flyvision_adapter.data.video import iter_gray_frames, list_videos
from flyvision_adapter.eye.render import HexRenderer
from flyvision_adapter.eye.temporal import DT, last_step_of_frame
from flyvision_adapter.eye.tiling import EyeTiling, receptor_offsets

sys.path.insert(0, str(Path(__file__).parent))
from phase1_inspect import CLIP_FRAMES, ROOT, TILE_BATCH, first_visible_run  # noqa: E402

OUT = Path("outputs/phase1")
LAG = 6  # steps (60 ms)
SKIP = 10  # onset transient, in video frames


def neighbour_index(extent: int, k: int) -> np.ndarray:
    """(721, 7) hexal indices of each hexal and its lattice neighbours (self-padded at rim)."""
    off = receptor_offsets(extent, k).astype(float)
    d = np.hypot(off[:, None, 0] - off[None, :, 0], off[:, None, 1] - off[None, :, 1])
    nb = np.full((len(off), 7), -1)
    for i in range(len(off)):
        idx = np.flatnonzero(d[i] <= 1.2 * k)
        nb[i, :len(idx)] = idx
    return np.where(nb < 0, np.arange(len(off))[:, None], nb)


def pct(scores: np.ndarray, tile: int, hexal: int) -> float:
    return float((scores < scores[tile, hexal]).mean() * 100)


def main(video: str = "dataset3/cam1", k: int = 13) -> None:
    import flyvis
    from flyvis.utils.activity_utils import LayerActivity

    v = {x.key: x for x in list_videos(ROOT)}[video]
    lab = load_labels(v.labels_path)
    start = first_visible_run(lab, CLIP_FRAMES, v.n_frames)
    frames = np.stack([f for _, f in iter_gray_frames(v, start, start + CLIP_FRAMES)])
    tiling = EyeTiling(*frames.shape[1:], kernel_size=k)
    stim = HexRenderer(tiling).render_clip(frames, v.fps)

    network = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000").init_network()
    cell_types = [t.decode() if isinstance(t, bytes) else str(t)
                  for t in network.connectome.unique_cell_types[:]]
    per_type: dict[str, list[np.ndarray]] = {}
    for b in range(0, tiling.n_tiles, TILE_BATCH):
        chunk = stim[b:b + TILE_BATCH]
        with torch.no_grad():
            state = network.fade_in_state(1.0, DT, chunk[:, 0])
            resp = network.simulate(chunk, DT, initial_state=state)
        la = LayerActivity(resp.cpu(), network.connectome, keepref=True)
        for t in cell_types:
            per_type.setdefault(t, []).append(la[t].numpy().astype(np.float32))
        del resp, la
    receptors = stim[:, :, 0].cpu().numpy()
    per_type["_framediff"] = [np.abs(np.diff(receptors, axis=1, prepend=receptors[:, :1]))]

    last = last_step_of_frame(len(frames), v.fps)
    nb = neighbour_index(tiling.extent, k)
    targets = []
    for i in range(SKIP, len(frames)):
        row = lab[lab.frame == start + i].iloc[0]
        targets.append((last[i], *tiling.locate(row.x * v.scale, row.y * v.scale)))

    rows = []
    n_hex = receptors.shape[-1]
    partial = {t: p[0].shape[-1] for t, p in per_type.items() if p[0].shape[-1] != n_hex}
    print("types without one cell per column (skipped):", partial)
    for t, parts in per_type.items():
        if t in partial:
            continue
        act = np.concatenate(parts)  # (tiles, steps, 721)
        dev = np.abs(act - np.median(act, axis=1, keepdims=True))
        res = {"cell_type": t}
        for name in ("dev", "lag", "ring"):
            res[name] = []
        for s, tile, hexal in targets:
            d = dev[:, s]
            lagged = dev[:, max(0, s - LAG + 1):s + 1].max(axis=1)
            ring = lagged[:, nb].max(axis=2)
            res["dev"].append(pct(d, tile, hexal))
            res["lag"].append(pct(lagged, tile, hexal))
            res["ring"].append(pct(ring, tile, hexal))
        rows.append({k_: (np.median(v_) if isinstance(v_, list) else v_) for k_, v_ in res.items()})
        del act, dev

    df = pd.DataFrame(rows).sort_values("ring", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"celltype_probe_{video.replace('/', '_')}_k{k}"
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(df.round(2).to_string(index=False))
    motion = df[df.cell_type.str.match(r"T[45][a-d]$")]
    print(json.dumps({"T4T5_median_ring": round(float(motion.ring.median()), 2),
                      "framediff_ring": round(float(df[df.cell_type == "_framediff"].ring.iloc[0]), 2)}))


if __name__ == "__main__":
    main(*sys.argv[1:2], *map(int, sys.argv[2:3]))
