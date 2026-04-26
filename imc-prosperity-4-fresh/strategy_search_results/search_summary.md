# Round 2 Strategy Search Summary

Date: 2026-04-18

Locked baseline:

- `mainbh_osmium_ret5_pressure_alpha.py`
- Baseline website recheck: `8954.9375`

Recommended final file:

- `main_final_round2_top05.py`
- This file matches `cand_no_micro_depth_top05.py`
- Final syntax check passed with `python3 -m py_compile`

Important warning:

- The website backtester is randomized across uploads.
- The same strategy can show materially different PnL when uploaded again.
- Do not judge the strategy by one lucky or unlucky upload. Use the robustness retests below.

## Best Single-Run Website Scores

| File | Website PnL | Idea |
| --- | ---: | --- |
| `cand_no_micro_depth_top05.py` | 9075.7334 | Remove unstable micro/depth trend terms and reduce top-level imbalance weight to 0.50 |
| `cand_no_micro_depth_top075.py` | 9033.9336 | Same, but top-level imbalance weight 0.75 |
| `cand_no_micro_depth_alpha_scale075.py` | 8987.4688 | Remove micro/depth and reduce total alpha scale |
| `cand_no_micro_depth.py` | 8961.8750 | Remove unstable micro/depth trend terms |
| `cand_no_micro_depth_top05_alpha125.py` | 8947.8750 | Top05 with higher alpha scale |
| `cand_no_micro_depth_top05_ema005.py` | 8938.2812 | Top05 with slower fair-value EMA |
| `cand_no_micro_depth_top05_alpha075.py` | 8913.8125 | Top05 with lower alpha scale |
| `cand_no_micro_depth_pressure3.py` | 8898.8750 | Remove micro/depth with weaker pressure |
| `cand_offset_tighter.py` | 8888.4688 | Slightly tighter quotes |

## Clean Robustness Retests

These used unique filenames to avoid duplicate-upload confusion on the website.

| Strategy bucket | Runs | Mean PnL | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Top05 winner | 2 | 8951.4727 | 8932.8125 | 8970.1328 |
| Baseline | 2 | 8886.3281 | 8832.4688 | 8940.1875 |
| Top075 neighbor | 2 | 8875.5703 | 8765.0000 | 8986.1406 |
| No micro/depth only | 2 | 8802.6719 | 8691.2812 | 8914.0625 |

## Conclusion

The best candidate is `cand_no_micro_depth_top05.py`, copied to `main_final_round2_top05.py`.

The alpha is simple:

- Keep the baseline buy-and-hold Pepper behavior.
- Keep Osmium market making and the existing return/pressure alpha.
- Remove microprice/depth-vwap trend terms because they were unstable across rounds.
- Keep top-level pressure, but damp it to 50% strength instead of trusting it fully.

This is not a huge miracle alpha. It is a cleaner, less noisy version of the current best Osmium logic, and its robustness retests beat the locked baseline by about `65.14` PnL on average.

