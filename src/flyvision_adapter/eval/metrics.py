"""Per-frame scoring and per-group summaries.

- hit@R: the detector's single strongest location is within R px (1080p scale) of the label,
  over frames where the drone is visible. Drones span ~5-60 px, so R=25 is "on the drone".
- AUC: does the strongest response separate drone-visible frames from drone-absent ones
  (Mann-Whitney on `peak`), i.e. can the detector tell when there is nothing to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..detectors.base import Clip, Detections

WARMUP = 30  # frames skipped at clip start: filters/backgrounds are still settling
RADII = (25, 50)


def frame_table(clip: Clip, det: Detections, name: str) -> pd.DataFrame:
    err = np.hypot(*(det.xy - clip.gt_xy).T)
    df = pd.DataFrame({"detector": name, "clip": clip.key, "frame": np.arange(len(err)),
                       "visible": clip.visible, "err": np.where(clip.visible, err, np.nan),
                       "peak": det.peak, "ms_per_frame": det.ms_per_frame})
    return df[df.frame >= WARMUP]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(frames: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        vis = g[g.visible]
        out = {f"hit@{r}": (vis.err <= r).mean() for r in RADII}
        out["median_err_px"] = vis.err.median()
        # AUC within each clip (peak units drift with scene brightness), then averaged
        aucs = [auc(c.peak[c.visible].to_numpy(), c.peak[~c.visible].to_numpy())
                for _, c in g.groupby("clip")]
        out["auc"] = np.nanmean(aucs) if np.isfinite(aucs).any() else np.nan
        out["ms_per_frame"] = g.groupby("clip").ms_per_frame.first().mean()
        out["n_visible"] = len(vis)
        return pd.Series(out)

    return frames.groupby(by).apply(one, include_groups=False).reset_index()
