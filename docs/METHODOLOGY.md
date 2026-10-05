# Methodology

## Data
Two PV plants are evaluated at 15-minute resolution using inverter generation and plant weather measurements. Generation is merged with weather on timestamp; duplicate and invalid rows are removed.

## Cleaning
Plant-wide DC/AC median ratio is used to detect the known order-of-magnitude DC unit mismatch. Remaining extreme DC/AC outliers are capped for feature quality. Daytime is defined from irradiation.

## Digital twin
The original fleet-wide HistGradientBoostingRegressor is retained as a benchmark. The operational twin now fits one model per inverter using irradiation, ambient temperature, module temperature, and cyclic time features. Lagged power and inverter ID are not used as predictors so a fault cannot be hidden by recent output.

## Uncertainty
Per-inverter 5th and 95th percentile gradient-boosting models provide prediction bands. These are predictive intervals, not probabilities of hardware failure.

## Anomaly calibration
Under-performance thresholds are calibrated from the chronological validation split only, per inverter, using empirical 99th/99.9th percentiles with robust MAD guards. EWMA smoothing (alpha=0.3) reduces one-sample noise. The held-out test set is used only for final reporting.

## Forecasting
The forecasting experiments retain history-only, measured-weather-at-origin, and future-weather oracle regimes. Persistence and same-time-yesterday baselines are included. Future measured weather must be described as an upper bound, not deployable weather forecasting.

## Limitations
Plant 2 has substantial inverter heterogeneity and periods with reduced inverter counts. Synthetic degradation experiments demonstrate sensitivity, but do not establish real-world fault-classification accuracy. Thresholds remain advisory unless validated against labeled operational events.
