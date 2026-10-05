from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODELS = ROOT / "models"

STEP_MIN = 15
DT_H = STEP_MIN / 60          # interval length in hours (15 min -> 0.25 h)
DAY_IRR = 0.05                # irradiation above this => "daytime" (DayFlag = 1)

# forecast horizons in 15-min steps. Add "24h": 96 if you want a day-ahead horizon.
HORIZONS = {"15min": 1, "1h": 4, "4h": 16}

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15        # chronological split, remaining 15% = test

TWIN_FEATURES = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
                 "hour_sin", "hour_cos"]

# --- anomaly-score settings (prototype values, NOT validated engineering thresholds) ---
NR_THR = 0.15      # normalised residual above which an interval is a "candidate" abnormal
NR_CAP = 0.50      # 50 % under-performance maps to a component score of 1.0
W1, W2, W3 = 0.4, 0.3, 0.3   # residual / persistence / peer-deviation weights
WARN, HIGH = 0.3, 0.6
MIN_EXPECTED_FRAC = 0.05     # ignore intervals where expected power < 5 % of rated
