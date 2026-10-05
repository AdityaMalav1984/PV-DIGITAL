# PV Digital Twin
Weather-aware, data-driven digital twin for PV forecasting, inverter performance anomaly detection and what-if simulation.

## Run
```
pip install -r requirements.txt
# the repository already contains the 4 CSVs in ./data/
python run_pipeline.py          # typically ~1–3+ min depending on CPU; forecasting is the bottleneck
streamlit run app.py            # dashboard
python forecast_lstm.py         # optional LSTM benchmark (needs torch)
```
Outputs: `outputs/tables/*.csv` (every number for the report), `outputs/figures/*.png` (Fig 3-12 per plant),
`outputs/data_quality_report.json`, `models/twin_plant*.joblib`.

## Layout
| file | module |
|---|---|
| `src/data.py` | 1-2 ingestion, cleaning, timestamp sync, day flag |
| `src/forecast.py` | 3-4 lag/rolling/cyclic features, persistence, seasonal-naive, LR, RF, GBM, XGBoost, cross-plant transfer |
| `src/twin.py` | 5-6 global benchmark + per-inverter twin, peer feature, uncertainty bands, calibrated anomaly scoring, degradation experiment |
| `src/simulate.py` | 7 scenario engine (irradiance, temperature, degradation, outage, energy loss) |
| `app.py` | 8 Streamlit dashboard |

## Design decisions worth defending in the viva
1. **Twin = weather + time + contemporaneous peer-inverter signal; no lagged power.** The global fleet model is retained as a benchmark, while the operational twin is per-inverter.
   Lagged power can follow a fault and hide the residual; peer output helps absorb spatial/cloud variability that a single weather sensor misses.
2. **Twin is trained on normal behaviour** (daytime zero-output rows removed from training) and uses absolute-error loss (robust).
3. **Forecasting uses three feature variants**: history-only, history + measured weather at origin, history + weather at target time
   (*assumes a perfect weather forecast*, so it is an upper bound, not a deployable result).
4. **Chronological 70/15/15 split**; all reported metrics are on the test part.
5. **Anomaly score** = 0.4·normalised residual + 0.3·persistence (2 h) + 0.3·peer deviation. Display scores retain these components, but alert status uses per-inverter validation-calibrated residual thresholds plus EWMA smoothing.
   Thresholds are empirical advisory limits, not fault probabilities.
6. **Degradation-injection experiment** removes real zero-output events first so it isolates gradual degradation.

## Data findings (also go in the report)
* Plant 1 `DC_POWER` is ~10x too large (DC/AC = 10.2); the loader detects and rescales it.
* Plant 1 generation uses `dd-mm-yyyy` timestamps, Plant 2 ISO; handled.
* Both plants: 34 days (15 May - 17 Jun 2020), 22 inverters. Plant 1 has 3157/3264 timestamps, Plant 2 3259/3264, but many inverter rows are missing,
  so complete plant-total timestamps are 3057 (P1) and 2355 (P2).
* Plant 2 has strong inverter heterogeneity and reduced inverter counts at some timestamps. The global twin performs poorly on several inverter groups; the per-inverter + peer twin improves Plant 2 test nMAE, but does not eliminate the gap.
* Validation calibration drives false-alert rates down, but there are no real fault labels, so alert thresholds remain advisory.

## Limitations (state these honestly)
* No fault labels => **anomaly detection, not fault diagnosis**. Do not claim detection accuracy against real faults.
* ML temperature response is unreliable on this data (temperature is collinear with irradiance). The simulator therefore defaults to an
  *assumed* physical coefficient (-0.4 %/°C, editable); "raw ML twin" mode is kept for transparency and gives implausible results on Plant 2.
* Irradiance scenarios are capped at the training maximum (tree models cannot extrapolate).
* The Streamlit dashboard requires the pinned Streamlit dependency and should be exercised after rebuilding outputs.
  The full pipeline can be CPU-bound because of repeated model fits; tests cover the core twin/data path. PatchTST/Transformer is not implemented.
