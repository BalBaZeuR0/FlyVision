#!/usr/bin/env bash
# Run locally (Git Bash): push the code (not data/outputs/.venv) to noron.
set -euo pipefail
source "$(dirname "$0")/noron_env.sh"
cd "$(dirname "$0")/../.."
tar -cf - --anchored --exclude=./.venv --exclude=./data --exclude=./outputs --exclude=./.git --no-anchored \
    --exclude=__pycache__ --exclude='*.egg-info' . \
  | ssh -o BatchMode=yes "$NORON_HOST" "mkdir -p '$PROJECT_DIR' && tar -xf - -C '$PROJECT_DIR'"
echo "synced -> $NORON_HOST:$PROJECT_DIR"
