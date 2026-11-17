#!/usr/bin/env bash
# ==============================================================================
# start_system.sh
# SANADA Full-Stack App Launcher (Hugging Face Cloud Backend Mode)
# Starts the lightweight client & API proxy server connected to Hugging Face
# ==============================================================================

BACKEND_URL="${BACKEND_URL:-https://eshansk-sanada-backend.hf.space}"
PORT="${PORT:-3000}"

echo "=================================================================="
echo " 🎧 Starting SANADA App (Hugging Face Backend Mode)"
echo "=================================================================="
echo " • Cloud Backend URL : $BACKEND_URL"
echo " • Checking backend health..."

# Quick health ping to Hugging Face Space
HEALTH_CHECK=$(curl -s -m 6 "$BACKEND_URL/api/health" 2>/dev/null)
if [[ $HEALTH_CHECK == *"online"* ]]; then
    echo " ✅ Hugging Face Backend is ONLINE and healthy!"
else
    echo " ⚠️ Hugging Face backend is connecting or waking up..."
    echo "    Space URL: https://huggingface.co/spaces/eshansk/sanada-backend"
fi

cleanup() {
    echo ""
    echo "Shutting down SANADA App server..."
    kill $SERVER_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Export configuration for dev_server.py
export BACKEND_URL="$BACKEND_URL"
export PORT="$PORT"

echo ""
echo " Starting SANADA App Server on http://0.0.0.0:$PORT ..."
python3 dev_server.py &
SERVER_PID=$!

echo "------------------------------------------------------------------"
echo " 🚀 System is LIVE and ready:"
echo "    • Local Web Client     : http://localhost:$PORT"
echo "    • Cloud Backend API    : $BACKEND_URL"
echo "    • Cloud API Docs       : $BACKEND_URL/docs"
echo "    • Cloud API Health     : $BACKEND_URL/api/health"
echo "------------------------------------------------------------------"
echo " Press [CTRL + C] anytime to stop the server."
echo "=================================================================="

wait
