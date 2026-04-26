#!/bin/zsh
set -euo pipefail

ROOT="/Users/lakshaykumar/Documents/Playground"
SESSION="diffusion_entropy_dashboard"
LOG="$ROOT/diffusion_entropy_dashboard.log"
APP="$ROOT/diffusion_entropy_dashboard.py"
PORT="8503"

mkdir -p "$ROOT"
touch "$LOG"

screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
screen -dmS "$SESSION" zsh -lc "cd '$ROOT' && exec python3 -m streamlit run '$APP' --server.port $PORT --server.address 127.0.0.1 --server.headless true >> '$LOG' 2>&1"

sleep 3

echo "Dashboard session: $SESSION"
echo "URL: http://127.0.0.1:$PORT"
echo
screen -ls || true
echo
lsof -nP -iTCP:$PORT -sTCP:LISTEN || true
