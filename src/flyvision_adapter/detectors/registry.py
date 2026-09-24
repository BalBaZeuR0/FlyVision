"""Build detectors from (family, params) — the same specs the dev sweep writes out, so the test
benchmark runs exactly the configuration chosen on dev clips."""

from __future__ import annotations

from .hexdet import HexEMD, ReceptorDiff
from .pixel import MOG2, FrameDiff
from .small_target import FrameDiffST, SmallTarget

_network = None


def _flyvis_network():
    global _network
    if _network is None:
        from .flyvis_motion import load_network

        _network = load_network()
    return _network


def build(name: str, family: str, params: dict):
    p = dict(params)
    st = p.pop("st", False)
    if family == "framediff":
        det = FrameDiffST(**p) if st else FrameDiff(**p)
    elif family == "mog2":
        det = MOG2(**p)
    elif family == "receptor_diff":
        det = ReceptorDiff()
    elif family == "emd":
        det = HexEMD(**p)
    elif family == "flyvis":
        from .flyvis_motion import FlyvisMotion

        det = FlyvisMotion(network=_flyvis_network(), **p)
    else:
        raise ValueError(f"unknown detector family {family!r}")
    if st and family in ("receptor_diff", "emd", "flyvis"):
        det = SmallTarget(det)
    det.name = name
    if isinstance(det, SmallTarget):
        det.inner.name = f"{name}/inner"
    return det
