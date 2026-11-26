#!/usr/bin/env bash
# ==============================================================================
# run_local.sh
# SANADA Local Development & Single-Command Launcher
# ==============================================================================

set -eo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

export PATH="$HOME/.local/bin:$BASE_DIR/.venv/bin:$BASE_DIR/venv/bin:$PATH"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
        source "$BASE_DIR/.venv/bin/activate"
    elif [ -f "$BASE_DIR/venv/bin/activate" ]; then
        source "$BASE_DIR/venv/bin/activate"
    fi
fi

# Install dependencies if requested or handle clearvoice with --no-deps (NumPy 2.x safe)
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    echo "Installing clearvoice with --no-deps..."
    pip install --no-deps clearvoice
elif ! python3 -c "import clearvoice" &>/dev/null; then
    echo "Notice: clearvoice is not installed. Attempting installation with --no-deps..."
    pip install --no-deps clearvoice 2>/dev/null || echo "clearvoice installation skipped (optional or offline)."
fi

# Run checkpoint verification & Phase 0 SOTA scan
python3 scripts/download_checkpoints.py

PORT="${PORT:-8000}"
echo "=================================================================="
echo " 🎧 Starting SANADA Local Server on port $PORT"
echo "=================================================================="
exec python3 -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"
