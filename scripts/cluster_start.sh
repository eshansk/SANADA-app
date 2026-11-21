#!/usr/bin/env bash
# ==============================================================================
# scripts/cluster_start.sh
# SANADA Campus HDFS/Spark Cluster Startup Script
# ==============================================================================

set -eo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

export PATH="$HOME/.local/bin:$BASE_DIR/.venv/bin:$BASE_DIR/venv/bin:$PATH"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
        source "$BASE_DIR/.venv/bin/activate"
    elif [ -f "$BASE_DIR/venv/bin/activate" ]; then
        source "$BASE_DIR/venv/bin/activate"
    fi
fi

echo "=================================================================="
echo " 🚀 Starting SANADA Campus Cluster Service (Offline Mode)"
echo "=================================================================="

export SANADA_DEPLOY_MODE="cluster"
export OFFLINE_MODE="${OFFLINE_MODE:-1}"
export HF_HUB_OFFLINE=1
export HF_HOME="$PWD/models/checkpoints/hf_cache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo " • Deploy Mode    : $SANADA_DEPLOY_MODE"
echo " • Offline Mode   : $OFFLINE_MODE"
echo " • Checkpoint Dir : $PWD/models/checkpoints"
echo " • HF Cache       : $HF_HOME"
echo " • OMP Threads    : $OMP_NUM_THREADS"
echo " • Port           : ${PORT:-8000}"
echo "=================================================================="

# Synchronize / verify checkpoints locally or via HDFS
python3 scripts/download_checkpoints.py

echo ""
echo "Launching FastAPI server & static UI bundle on 0.0.0.0:${PORT:-8000}..."
exec python3 -m uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"
