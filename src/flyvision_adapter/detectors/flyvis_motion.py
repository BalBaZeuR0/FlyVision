"""flyvis (Lappalainen et al. 2024) connectome-constrained network as a detector.

Readout: summed |activity - running mean| of the T4/T5 motion detectors (all 8 subtypes), per
column. The running mean removes each cell's tonic baseline, which otherwise swamps the
target-evoked response (phase 1 finding); it is causal, unlike the clip median used there.
"""

from __future__ import annotations

import numpy as np
import torch

from ..eye.temporal import DT
from .base import HexDetector, HexInput, lag_max
from .filters import lowpass

MOTION_TYPES = tuple(f"T{a}{b}" for a in (4, 5) for b in "abcd")


class FlyvisMotion(HexDetector):
    name = "flyvis_t4t5"

    def __init__(self, model: str = "flow/0000/000", cell_types=MOTION_TYPES,
                 tau_baseline: float = 0.5, tile_batch: int = 8):
        import flyvis
        from flyvis.utils.activity_utils import LayerActivity

        self.network = flyvis.NetworkView(flyvis.results_dir / model).init_network()
        la = LayerActivity(torch.zeros(1, self.network.connectome.nodes.type[:].shape[0]),
                           self.network.connectome, keepref=True)
        # (n_types, 721) node indices, in the same hexal order as the input
        self.index = torch.as_tensor(np.stack([dict.__getitem__(la, t) for t in cell_types]))
        self.tau_baseline, self.tile_batch = tau_baseline, tile_batch
        self.name = f"flyvis_{'t4t5' if tuple(cell_types) == MOTION_TYPES else '+'.join(cell_types)}"

    @torch.no_grad()
    def activity(self, stim: torch.Tensor) -> torch.Tensor:
        """(tiles, steps, 721) receptor input -> (tiles, steps, n_types, 721) readout cells."""
        out = []
        idx = self.index.to(stim.device)
        for b in range(0, stim.shape[0], self.tile_batch):
            chunk = stim[b:b + self.tile_batch, :, None]
            state = self.network.fade_in_state(1.0, DT, chunk[:, 0])
            resp = self.network.simulate(chunk, DT, initial_state=state)
            out.append(resp[..., idx])
            del resp
        return torch.cat(out)

    def score(self, inp: HexInput) -> torch.Tensor:
        act = self.activity(inp.stim)
        tiles, steps = act.shape[:2]
        flat = act.flatten(2)
        dev = (flat - lowpass(flat, self.tau_baseline)).abs()
        energy = dev.view(tiles, steps, *act.shape[2:]).sum(dim=2)
        return lag_max(energy, inp.last_step, self.lag)

