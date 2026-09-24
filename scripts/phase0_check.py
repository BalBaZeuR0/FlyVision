"""Phase 0: does flyvis run on noron's RTX 5060, which cell types does it model, and how fast is it?

Run on noron from the project dir:  .venv/bin/python scripts/phase0_check.py
Writes outputs/phase0.json (pull it back with scripts/remote/fetch_outputs.sh).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

OUT = Path("outputs/phase0.json")
MODEL = "flow/0000/000"  # first model of the paper's pretrained ensemble
DT = 1 / 100  # flyvis integration step (s)
VIDEO_FPS = 30  # drone videos are ~30 fps -> ~3.3 sim steps per video frame
N_STEPS = 100
TILE_BATCHES = (1, 4, 8, 16, 32)  # 64 tiles OOMs the 8GB card
# a 1920x1080 frame covered by BoxEye windows (31 columns x 13 px ~ 403 px wide), rough count
TILES_PER_FRAME = 15


def gpu_report() -> dict:
    rep = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if rep["cuda_available"]:
        rep["device"] = torch.cuda.get_device_name(0)
        rep["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        rep["torch_cuda"] = torch.version.cuda
        # a real kernel launch: a torch build without sm_120 kernels fails here, not at import
        x = torch.randn(1024, 1024, device="cuda")
        rep["matmul_ok"] = bool(torch.isfinite(x @ x).all())
    return rep


def cell_type_report(network) -> dict:
    types = [t.decode() if isinstance(t, bytes) else str(t)
             for t in network.connectome.unique_cell_types[:]]
    prefixes = ("LC", "LPLC", "LLPC", "Li", "LT", "LPi")
    return {
        "n_cell_types": len(types),
        "n_cells": int(network.connectome.nodes.type[:].shape[0]),
        "cell_types": types,
        # LC/LPLC are the fly's small-object detectors; if absent we need our own readout layer
        "lobula_object_types": [t for t in types if t.startswith(prefixes)],
        "has_T4_T5": all(any(t.startswith(p) for t in types) for p in ("T4", "T5")),
    }


def boxeye_report() -> dict:
    from flyvis.datasets.rendering import BoxEye

    eye = BoxEye(extent=15, kernel_size=13)
    frame = torch.rand(1, 1, 436, 436)  # [samples, frames, H, W], a bit larger than the eye
    out = eye(frame)
    return {"boxeye_in": list(frame.shape), "boxeye_out": list(out.shape)}


def timing_report(network) -> list[dict]:
    rows = []
    for batch in TILE_BATCHES:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            dt_s, shape = _time_batch(network, batch)
        except torch.OutOfMemoryError:
            rows.append({"batch_tiles": batch, "error": "CUDA OOM"})
            break  # larger batches will OOM too
        ms_per_step = 1000 * dt_s / N_STEPS
        steps_per_video_frame = (1 / VIDEO_FPS) / DT
        rows.append({
            "batch_tiles": batch,
            "response_shape": shape,
            "ms_per_step": round(ms_per_step, 3),
            # one video frame = TILES_PER_FRAME tiles, each advanced ~3.3 steps; batching amortises
            "est_ms_per_video_frame": round(
                ms_per_step * steps_per_video_frame * TILES_PER_FRAME / batch, 2),
            "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 2**20)
                            if torch.cuda.is_available() else None),
        })
    return rows


def _time_batch(network, batch: int) -> tuple[float, list[int]]:
    movie = torch.rand(batch, N_STEPS, 1, 721)
    with torch.no_grad():
        state = network.fade_in_state(1.0, DT, movie[:, 0])  # (batch, 1, 721)
        network.simulate(movie[:, :5], DT, initial_state=state)  # warm-up
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        resp = network.simulate(movie, DT, initial_state=state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0, list(resp.shape)


def t4_readout_check(network) -> dict:
    movie = torch.rand(1, 20, 1, 721)
    with torch.no_grad():
        state = network.fade_in_state(1.0, DT, movie[:, 0])
        layer = network.simulate(movie, DT, initial_state=state, as_layer_activity=True)
    return {"T4c_shape": list(layer["T4c"].shape)}


def main() -> None:
    report: dict = {"gpu": gpu_report()}

    import flyvis

    report["flyvis_version"] = flyvis.__version__
    report["connectome_file"] = flyvis.connectome_file.name
    network = flyvis.NetworkView(flyvis.results_dir / MODEL).init_network()

    # each section is independent: one failing (API drift) shouldn't hide the others
    for name, fn in [("cells", lambda: cell_type_report(network)),
                     ("boxeye", boxeye_report),
                     ("t4_readout", lambda: t4_readout_check(network)),
                     ("timing", lambda: timing_report(network))]:
        try:
            report[name] = fn()
        except Exception as e:  # noqa: BLE001 — record and move on
            report[name] = {"error": f"{type(e).__name__}: {e}"}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k != "cells"}
    summary["cells"] = dict(report.get("cells", {}))
    if "cell_types" in summary["cells"]:
        summary["cells"]["cell_types"] = " ".join(summary["cells"]["cell_types"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
