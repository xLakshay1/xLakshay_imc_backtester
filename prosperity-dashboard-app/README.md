# Prosperity Dashboard App

Clean deployable bundle of the IMC Prosperity Streamlit dashboard.

## Included

- `imc_round1_dashboard.py` — main Streamlit app
- `round_backtester_core.py` — deterministic local submission backtester
- `backtester/` — trimmed prosperity backtester packages used by the app
- `strategies/main.py` — sample strategy file
- `data/ROUND1`, `data/ROUND2`, `data/ROUND3` — drop your price/trade CSVs here

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run imc_round1_dashboard.py
```

## Streamlit Community Cloud

Deploy with:

- **Main file path:** `imc_round1_dashboard.py`
- **Python:** 3.11 or 3.12

### Important

This repo does **not** include your personal `ROUND1/ROUND2/ROUND3` datasets by default.

For full dashboard functionality, either:

1. commit the CSV datasets into:
   - `data/ROUND1`
   - `data/ROUND2`
   - `data/ROUND3`

or

2. provide alternate paths with environment variables:
   - `IMC_DATA_ROOT`
   - `IMC_ROUND1_DATA_DIR`
   - `IMC_ROUND2_DATA_DIR`
   - `IMC_ROUND3_DATA_DIR`
   - `IMC_STRATEGY_ROOT`
   - `IMC_P3_RESOURCES_DIR`
   - `IMC_P3_R3_HUMAN_DISTRIBUTION_IMAGE`

## Notes

- Prosperity 3 Round 3 option analysis uses bundled `backtester/prosperity3bt/resources/round3`.
- The Prosperity 3 human-distribution screenshot is optional; the app will show a fallback note if it is missing.
