import os
from pathlib import Path

import pytest

# local copy on the Windows PC; on noron the same tree lives under data/
_CANDIDATES = [
    Path(os.environ.get("DRONE_DATA", "data/drone-tracking-datasets")),
    Path.home() / "OneDrive/Desktop/DatasetDroneTracker/drone-tracking-datasets",
]


@pytest.fixture(scope="session")
def data_root() -> Path:
    for p in _CANDIDATES:
        if (p / "dataset1" / "detections").is_dir():
            return p
    pytest.skip("drone-tracking-datasets not found")
