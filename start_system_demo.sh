#!/usr/bin/env bash
# ==============================================================================
# start_system_demo.sh
# SANADA Live Demo High-Utilization GPU Launcher (Crash-Safe)
# ==============================================================================
# Keeps active SOTA Wave 1-4 models hot in GPU memory using a bounded LRU cache
# sized to Unified Memory (Apple Silicon MPS) or VRAM (NVIDIA CUDA), while
# leaving safe OS/browser headroom so 16 GB Macs and laptop GPUs never crash.
# ==============================================================================

set -eo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Always prefer project-local virtualenv
if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
    source "$BASE_DIR/.venv/bin/activate"
elif [ -f "$BASE_DIR/venv/bin/activate" ]; then
    source "$BASE_DIR/venv/bin/activate"
fi
export PATH="$BASE_DIR/.venv/bin:$BASE_DIR/venv/bin:$HOME/.local/bin:$PATH"

APP_PORT="${PORT:-8000}"
PREWARM="0"
CACHE_GB_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            APP_PORT="$2"
            shift 2
            ;;
        --prewarm)
            PREWARM="1"
            shift
            ;;
        --cache-gb)
            CACHE_GB_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./start_system_demo.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port <PORT>      Port to listen on (default: 8000)"
            echo "  --prewarm          Pre-load core models (E1, A+, B, C1-2spk, E2) into GPU at startup"
            echo "  --cache-gb <GB>    Override GPU LRU model cache budget in GB (auto-detected by default)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage instructions."
            exit 1
            ;;
    esac
done

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found in PATH."
    exit 1
fi

if ! python3 -c "import uvicorn, fastapi" &>/dev/null; then
    echo "Error: Required packages missing. Run: python3 setup_sanada.py"
    exit 1
fi

OS_NAME="$(uname -s)"
GPU_PROFILE="CPU (Fast-CPU fallback)"
AUTO_CACHE_GB="0"

if [ "$OS_NAME" = "Darwin" ]; then
    MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 17179869184)"
    MEM_GB="$((MEM_BYTES / 1073741824))"

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_PREFER_METAL=1

    # On 16 GB Apple Silicon, recommendedMaxWorkingSetSize is ~10.67 GB:
    # 1.05x (~11.2 GB hard cap) + 0.80x (~8.5 GB soft GC) maximizes GPU usage
    # while preserving ~4.8 GB headroom for macOS, WindowServer, and browser.
    if [ "$MEM_GB" -ge 24 ]; then
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-1.15}"
        export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.90}"
        AUTO_CACHE_GB="7.5"
        GPU_PROFILE="Apple Silicon MPS (${MEM_GB} GB Unified Memory | All-Models Hot Cache)"
    elif [ "$MEM_GB" -ge 15 ]; then
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-1.05}"
        export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.80}"
        AUTO_CACHE_GB="6.5"
        GPU_PROFILE="Apple Silicon MPS (${MEM_GB} GB Unified Memory | Demo High-GPU Safe Mode)"
    else
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.85}"
        export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.60}"
        AUTO_CACHE_GB="2.5"
        GPU_PROFILE="Apple Silicon MPS (${MEM_GB} GB Unified Memory | Compact Safe Cache)"
    fi
else
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
    VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9' || true)"
    if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" -gt 0 ]; then
        VRAM_GB="$((VRAM_MB / 1024))"
        if [ "$VRAM_GB" -ge 12 ]; then
            AUTO_CACHE_GB="7.5"
        elif [ "$VRAM_GB" -ge 8 ]; then
            AUTO_CACHE_GB="5.0"
        elif [ "$VRAM_GB" -ge 6 ]; then
            AUTO_CACHE_GB="3.2"
        else
            AUTO_CACHE_GB="1.8"
        fi
        GPU_PROFILE="NVIDIA CUDA (${VRAM_GB} GB VRAM | Expandable Segments + LRU Cache)"
    fi
fi

FINAL_CACHE_GB="${CACHE_GB_OVERRIDE:-$AUTO_CACHE_GB}"
if [ "$FINAL_CACHE_GB" != "0" ]; then
    export SANADA_DEMO_GPU_CACHE=1
    export SANADA_GPU_CACHE_MAX_GB="$FINAL_CACHE_GB"
else
    export SANADA_DEMO_GPU_CACHE=0
fi

export SANADA_DEMO_PREWARM="$PREWARM"
export SANADA_DEPLOY_MODE="local"
export HF_HOME="$BASE_DIR/models/checkpoints/hf_cache"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

if python3 -c "import socket; s=socket.socket(); s.settimeout(0.5); r=s.connect_ex(('127.0.0.1', int('$APP_PORT'))); s.close(); exit(0 if r==0 else 1)" 2>/dev/null; then
    echo "Warning: Port $APP_PORT is already in use. Stop the existing server or pass --port <PORT>."
fi

echo "=================================================================="
echo " SANADA LIVE DEMO SERVER (HIGH-GPU CRASH-SAFE MODE)"
echo "=================================================================="
echo " • Profile            : $GPU_PROFILE"
echo " • Demo GPU LRU Cache : SANADA_DEMO_GPU_CACHE=$SANADA_DEMO_GPU_CACHE (Max ${FINAL_CACHE_GB} GB)"
echo " • Startup Pre-warm   : $PREWARM"
echo " • Web Studio & Docs  : http://localhost:$APP_PORT  (/docs, /api/health)"
echo "=================================================================="

# Strictly single worker so only ONE process owns the GPU memory pool
exec python3 -m uvicorn backend.app:app --host 0.0.0.0 --port "$APP_PORT" --workers 1
