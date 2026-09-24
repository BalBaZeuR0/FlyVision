#!/usr/bin/env bash
# Run ON noron, from $PROJECT_DIR: venv + torch (cu128, required for the RTX 5060 / Blackwell)
# + flyvis + its pretrained models. Idempotent — safe to re-run.
set -euo pipefail
source "$(dirname "$0")/noron_env.sh"
cd "$PROJECT_DIR"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$FLYVIS_ROOT_DIR"

# flyvis reads FLYVIS_ROOT_DIR from a .env in the working directory
echo "FLYVIS_ROOT_DIR=$FLYVIS_ROOT_DIR" > .env

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# torch first, from the cu128 index: flyvis leaves torch unpinned, and older CUDA builds
# have no kernels for sm_120
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e ".[bio,dev]"

.venv/bin/flyvis download-pretrained
echo "setup done"
