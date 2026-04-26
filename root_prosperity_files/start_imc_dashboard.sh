#!/bin/zsh
set -euo pipefail

ROOT="/Users/lakshaykumar/Documents/Playground"
SESSION="imc_round_dashboard"
LOG="$ROOT/streamlit_dashboard.log"
APP="$ROOT/imc_round1_dashboard.py"

mkdir -p "$ROOT"
touch "$LOG"

screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
screen -dmS "$SESSION" zsh -lc "cd '$ROOT' && exec python3 -m streamlit run '$APP' --server.port 8501 --server.address 127.0.0.1 --server.headless true >> '$LOG' 2>&1"

sleep 3

echo "Dashboard session: $SESSION"
echo "URL: http://127.0.0.1:8501"
echo
screen -ls || true
echo
lsof -nP -iTCP:8501 -sTCP:LISTEN || true
