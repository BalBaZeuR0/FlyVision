"""Manual 2D drone labels of CenekAlbl/drone-tracking-datasets (`<dataset>/detections/camN.txt`).

Each row is `frame_no x y` with 1-based frame numbers; `0 0` means the drone was not labelled
(out of view / occluded). Some files start with a ` frame no.  x  y` header line, some don't.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(path: str | Path) -> pd.DataFrame:
    """Return one row per labelled frame: `frame` (0-based video index), `x`, `y`, `visible`."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue  # header or blank
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    df = pd.DataFrame({
        "frame": arr[:, 0].astype(np.int64) - 1,
        "x": arr[:, 1],
        "y": arr[:, 2],
    })
    df["visible"] = ~((df.x == 0) & (df.y == 0))
    return df
