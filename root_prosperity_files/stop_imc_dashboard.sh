#!/bin/zsh
set -euo pipefail

SESSION="imc_round_dashboard"
screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
echo "Stopped dashboard session: $SESSION"
