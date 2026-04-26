# IMC Prosperity Dashboard

This repo contains the Streamlit dashboard I used to analyze and backtest IMC Prosperity strategies locally.

## What's included

- `imc_round1_dashboard.py` — main Streamlit app
- `round_backtester_core.py` — shared local backtester logic
- `backtester/` — prosperity backtester package used by the dashboard
- `main.py` — current combined strategy file
- `start_imc_dashboard.sh` / `stop_imc_dashboard.sh` — quick local run helpers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./start_imc_dashboard.sh
```

The dashboard will be available at [http://127.0.0.1:8501](http://127.0.0.1:8501).

To stop it:

```bash
./stop_imc_dashboard.sh
```

## Data paths

By default the dashboard looks for local data in:

- `~/Downloads/ROUND1`
- `~/Downloads/ROUND2`
- `~/Downloads/ROUND3`

You can override these with environment variables:

- `IMC_DOWNLOADS_ROOT`
- `IMC_ROUND1_DATA_DIR`
- `IMC_ROUND2_DATA_DIR`
- `IMC_ROUND3_DATA_DIR`
- `IMC_STRATEGY_ROOT`
- `IMC_P3_RESOURCES_DIR`
- `IMC_P3_R3_HUMAN_DISTRIBUTION_IMAGE`

## Notes

- Submission history is stored under `dashboard_submission_history/`.
- Some round-specific panels expect the bundled `backtester/` directory to stay in this repo.
