# FlyVision_Adapter

Small-target (drone) detection on real multi-camera footage
([CenekAlbl/drone-tracking-datasets](https://github.com/CenekAlbl/drone-tracking-datasets)),
comparing detectors of increasing biological fidelity:

1. classical CV baseline (background subtraction)
2. lightweight fly-inspired adapter — Reichardt/EMD motion detector + center-surround
3. connectome-constrained adapter — [flyvis](https://github.com/TuragaLab/flyvis)
   (Lappalainen et al. 2024, Nature), plus a shuffled-connectome control

Metrics: pixel error, precision/recall/F1, AUC, ms/frame.

## Where things run
- Code is written on the Windows PC; nothing heavy runs there.
- All runs happen on `noron` (Ubuntu, RTX 5060) under
  `/media/noron/DISK02/FlyVision_Adapter` — never on `/` (99% full).

## Phase 0 (setup check)
```bash
# local: push code to noron
bash scripts/remote/sync_code.sh
# on noron (or via ssh):
bash scripts/remote/setup_noron.sh      # venv + torch cu128 + flyvis + pretrained models
.venv/bin/python scripts/phase0_check.py  # GPU, cell-type inventory (LC?), sim speed
```
