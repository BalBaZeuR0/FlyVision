#!/usr/bin/env bash
# Run locally (Git Bash): pull noron's outputs/ back into the local repo.
set -euo pipefail
source "$(dirname "$0")/noron_env.sh"
cd "$(dirname "$0")/../.."
mkdir -p outputs
ssh -o BatchMode=yes "$NORON_HOST" "cd '$PROJECT_DIR' && tar -cf - outputs" | tar -xf - -C .
echo "fetched outputs/ from $NORON_HOST"
