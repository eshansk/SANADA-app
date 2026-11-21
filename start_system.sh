#!/usr/bin/env bash
# ==============================================================================
# start_system.sh
# SANADA Full-Stack Production & Local Development Launcher
# ==============================================================================

set -eo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Ensure user-installed binaries and virtualenv paths are accessible
export PATH="$HOME/.local/bin:$BASE_DIR/.venv/bin:$BASE_DIR/venv/bin:$PATH"

# Auto-activate virtualenv if present and not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
        source "$BASE_DIR/.venv/bin/activate"
    elif [ -f "$BASE_DIR/venv/bin/activate" ]; then
        source "$BASE_DIR/venv/bin/activate"
    fi
fi

MODE="direct"
BACKEND_ARG="http://127.0.0.1:8000"
PROXY_PORT="${PORT:-3000}"
APP_PORT="${PORT:-8000}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --proxy)
            MODE="proxy"
            shift
            ;;
        --backend)
            BACKEND_ARG="$2"
            shift 2
            ;;
        --port)
            if [ "$MODE" = "proxy" ]; then
                PROXY_PORT="$2"
            else
                APP_PORT="$2"
            fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./start_system.sh [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  (Default)                 Run unified FastAPI backend & frontend at http://localhost:8000"
            echo "  --proxy                   Run frontend dev_server reverse proxy on port 3000"
            echo "  --backend <URL>           Target backend URL for proxy mode (default: http://127.0.0.1:8000)"
            echo "  --port <PORT>             Port to listen on (default: 8000 for direct, 3000 for proxy)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage instructions."
            exit 1
            ;;
    esac
done

# Ensure python3 is available
if ! command -v python3 &>/dev/null; then
    echo "❌ Error: python3 is not found in PATH."
    echo "   Please install Python 3.10+ before running SANADA."
    exit 1
fi

check_port() {
    local target_port="$1"
    python3 -c "import socket; s = socket.socket(); s.settimeout(0.5); res = s.connect_ex(('127.0.0.1', int('$target_port'))); s.close(); exit(0 if res == 0 else 1)" 2>/dev/null
}

if [ "$MODE" = "proxy" ]; then
    if check_port "$PROXY_PORT"; then
        echo "⚠️  Warning: Port $PROXY_PORT is already in use."
        echo "   Use --port <PORT> to choose another port, or stop the conflicting service."
    fi

    echo "=================================================================="
    echo " 🎧 Starting SANADA Dev Proxy Server"
    echo "=================================================================="
    echo " • Target Backend URL : $BACKEND_ARG"
    echo " • Proxy Port         : $PROXY_PORT"
    echo " • Serving UI from    : $BASE_DIR/frontend"
    echo "=================================================================="

    export BACKEND_URL="$BACKEND_ARG"
    export PORT="$PROXY_PORT"

    exec python3 dev_server.py
else
    # Validate required backend packages
    if ! python3 -c "import uvicorn, fastapi" &>/dev/null; then
        echo "❌ Error: Required packages (fastapi, uvicorn) are not available to python3."
        echo "   Please run: pip install -r requirements.txt"
        exit 1
    fi

    if check_port "$APP_PORT"; then
        echo "⚠️  Warning: Port $APP_PORT is already in use by another process."
        echo "   Use --port <PORT> to choose another port, or terminate the existing process:"
        echo "   fuser -k $APP_PORT/tcp"
    fi

    echo "=================================================================="
    echo " 🎧 Starting SANADA Local Unified Server (API + Web UI)"
    echo "=================================================================="
    echo " • Listening on       : http://0.0.0.0:$APP_PORT"
    echo " • Web UI & API Docs  : http://localhost:$APP_PORT (UI at /, Docs at /docs)"
    echo "=================================================================="

    export SANADA_DEPLOY_MODE="local"
    export HF_HOME="$BASE_DIR/models/checkpoints/hf_cache"

    exec python3 -m uvicorn backend.app:app --host 0.0.0.0 --port "$APP_PORT"
fi
