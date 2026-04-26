#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec python3 -m streamlit run imc_round1_dashboard.py
