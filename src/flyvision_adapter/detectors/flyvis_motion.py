"""flyvis (Lappalainen et al. 2024) connectome-constrained network as a detector.

Readout per chosen cell type: |activity - running mean| (the running mean removes each cell's
tonic baseline, which otherwise swamps the target-evoked response — phase 1 finding), max over
the last `lag` steps of each frame, optionally scaled by that type's mean over the frame so that
types with large tonic swings don't drown the others, then summed over types per column.
Default = the 8 T4/T5 motion detectors, untuned; `scripts/tune_dev.py` chooses on dev clips.
"""

from __future__ import annotations

import numpy as np
import torch

from ..eye.temporal import DT
from .base import HexDetector, HexInput, lag_max
from .filters import lowpass

MOTION_TYPES = tuple(f"T{a}{b}" for a in (4, 5) for b in "abcd")


def load_network(model: str = "flow/0000/000"):
    import flyvis

    return flyvis.NetworkView(flyvis.results_dir / model).init_network()


def type_index(network, cell_types) -> torch.Tensor:
    """(n_types, 721) node indices of each type, in the same hexal order as the input."""
    from flyvis.utils.activity_utils import LayerActivity

    la = LayerActivity(torch.zeros(1, network.connectome.nodes.type[:].shape[0]),
                       network.connectome, keepref=True)
    return torch.as_tensor(np.stack([dict.__getitem__(la, t) for t in cell_types]))


def columnar_types(network) -> list[str]:
    """Cell types with one cell per column (all but Lawf1/Lawf2)."""
    from flyvis.utils.activity_utils import LayerActivity

    la = LayerActivity(torch.zeros(1, network.connectome.nodes.type[:].shape[0]),
                       network.connectome, keepref=True)
    names = [t.decode() if isinstance(t, bytes) else str(t)
             for t in network.connectome.unique_cell_types[:]]
    return [t for t in names if len(dict.__getitem__(la, t)) == 721]


@torch.no_grad()
def simulate_types(network, index: torch.Tensor, stim: torch.Tensor) -> torch.Tensor:
    """(tiles, steps, 721) receptor input -> (tiles, steps, n_types, 721) activity."""
    chunk = stim[:, :, None]
    state = network.fade_in_state(1.0, DT, chunk[:, 0])
    resp = network.simulate(chunk, DT, initial_state=state)
    return resp[..., index.to(resp.device)]


def type_frame_maps(act: torch.Tensor, last_step: np.ndarray, tau: float, lag: int
                    ) -> torch.Tensor:
    """(tiles, steps, n_types, 721) activity -> (T, tiles, n_types, 721) per-frame deviation."""
    tiles, steps, n, h = act.shape
    flat = act.reshape(tiles, steps, n * h)
    dev = (flat - lowpass(flat, tau)).abs()
    return lag_max(dev, last_step, lag).view(-1, tiles, n, h)


def combine(maps: torch.Tensor, normalize: bool) -> torch.Tensor:
    """(T, tiles, n_types, 721) -> (T, tiles, 721)."""
    if normalize:
        maps = maps / maps.mean(dim=(1, 3), keepdim=True).clamp_min(1e-8)
    return maps.sum(dim=2)


class FlyvisMotion(HexDetector):
    def __init__(self, cell_types=MOTION_TYPES, tau_baseline: float = 0.5, lag: int = 6,
                 normalize: bool = False, tile_batch: int = 8, network=None,
                 name: str | None = None):
        self.network = network or load_network()
        self.cell_types = tuple(cell_types)
        self.index = type_index(self.network, self.cell_types)
        self.tau_baseline, self.lag, self.normalize = tau_baseline, lag, normalize
        self.tile_batch = tile_batch
        self.name = name or ("flyvis_t4t5" if self.cell_types == MOTION_TYPES
                             else "flyvis_" + "+".join(self.cell_types))

    def score(self, inp: HexInput) -> torch.Tensor:
        maps = []
        for b in range(0, inp.stim.shape[0], self.tile_batch):
            act = simulate_types(self.network, self.index, inp.stim[b:b + self.tile_batch])
            maps.append(type_frame_maps(act, inp.last_step, self.tau_baseline, self.lag))
            del act
        return combine(torch.cat(maps, dim=1), self.normalize)
