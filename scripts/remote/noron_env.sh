# Shared settings for every script that runs on noron. `source` it, don't execute it.
# noron's `/` is 99% full: every cache, temp file and download must land on DISK02.
export NORON_HOST="noron@noron-hp-elite-tower-800-g9-desktop-pc"
export PROJECT_DIR="/media/noron/DISK02/FlyVision_Adapter"
export DATA_DIR="$PROJECT_DIR/data/drone-tracking-datasets"
export FLYVIS_ROOT_DIR="$PROJECT_DIR/flyvis_data"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export TMPDIR="$PROJECT_DIR/.cache/tmp"
