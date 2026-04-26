#!/bin/zsh
set -euo pipefail

SESSION="eco_assignment_dashboard"

screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
echo "Stopped session: $SESSION"
