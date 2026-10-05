"""Modules 3-4: feature engineering + forecasting engine (plant-level)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import HORIZONS, STEP_MIN, TRAIN_FRAC, VAL_FRAC, DAY_IRR

VARIANTS = ["history", "weather_now", "weather_future"]
# history        : lags + rolling stats + time                 (no weather)
# weather_now    : history + weather measured at forecast origin
# weather_future : history + weather at target time (assumes a perfect weather forecast -> upper bound)


def make_xy(total: pd.Series, wx: pd.DataFrame, h: int, variant: str):
    P = total
    X = pd.DataFrame(index=P.index)
    X["P_t"] = P
    for k in (1, 4, 8):
        X[f"P_t-{k}"] = P.shift(k)
    X["roll_mean_4"] = P.rolling(4).mean()
    X["roll_std_4"] = P.rolling(4).std()
    if h <= 96:                                   # same clock time yesterday (relative to target)
        X["P_same_time_yesterday"] = P.shift(96 - h)
    tt = P.index + pd.Timedelta(minutes=STEP_MIN * h)
    hr = tt.hour + tt.minute / 60
    X["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    if variant == "weather_now":
        X = X.join(wx)
    elif variant == "weather_future":
        X = X.join(wx.shift(-h).add_suffix("_fut"))
    y = P.shift(-h)
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def chrono_split(n):
    i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return slice(0, i1), slice(i1, i2), slice(i2, n)


def metrics(y, yhat, rated, irr=None):
    y, yhat = np.asarray(y), np.asarray(yhat)
    m = y > 0.10 * rated                               # MAPE masked: PV output -> 0 at night
    out = {"MAE": mean_absolute_error(y, yhat),
           "RMSE": mean_squared_error(y, yhat) ** 0.5,
           "R2": r2_score(y, yhat),
           "nMAE_%": 100 * mean_absolute_error(y, yhat) / rated,
           "MAPE_%(>10%rated)": 100 * np.mean(np.abs((y[m] - yhat[m]) / y[m])) if m.any() else np.nan}
    return out


def model_zoo():
    z = {"LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
         "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0),
         "GradientBoosting(sklearn)": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)}
    try:
        from xgboost import XGBRegressor
        z["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                                    colsample_bytree=0.8, random_state=0, n_jobs=-1)
    except ImportError:
        pass
    return z


def run_forecast_experiments(total, wx, plant):
    rated = float(total.quantile(0.995))
    rows, preds = [], {}
    for hname, h in HORIZONS.items():
        for variant in VARIANTS:
            X, y = make_xy(total, wx, h, variant)
            tr, va, te = chrono_split(len(X))
            if variant == "history":                   # baselines need no training
                rows.append(dict(plant=plant, model="Persistence", variant="-", horizon=hname,
                                 **metrics(y.iloc[te], X["P_t"].iloc[te], rated)))
                if "P_same_time_yesterday" in X:
                    rows.append(dict(plant=plant, model="SeasonalNaive(24h)", variant="-", horizon=hname,
                                     **metrics(y.iloc[te], X["P_same_time_yesterday"].iloc[te], rated)))
                preds[(hname, "Persistence")] = pd.Series(X["P_t"].iloc[te].values, index=y.index[te])
            for name, mdl in model_zoo().items():
                mdl.fit(X.iloc[tr], y.iloc[tr])
                yhat = np.clip(mdl.predict(X.iloc[te]), 0, None)
                rows.append(dict(plant=plant, model=name, variant=variant, horizon=hname,
                                 **metrics(y.iloc[te], yhat, rated)))
                preds[(hname, name, variant)] = pd.Series(yhat, index=y.index[te])
            preds[(hname, "actual")] = y.iloc[te]
    return pd.DataFrame(rows), preds, rated


def cross_plant_transfer(series_a, series_b, name_a, name_b, variant="weather_now"):
    """Train on plant A (train part), test on plant B (its test part)."""
    rows = []
    rated_b = float(series_b[0].quantile(0.995))
    for hname, h in HORIZONS.items():
        Xa, ya = make_xy(*series_a, h, variant)
        Xb, yb = make_xy(*series_b, h, variant)
        tra, _, _ = chrono_split(len(Xa))
        _, _, teb = chrono_split(len(Xb))
        _, _, tea = chrono_split(len(Xa))
        trb, _, _ = chrono_split(len(Xb))
        m_tr = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xa.iloc[tra], ya.iloc[tra])
        m_in = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xb.iloc[trb], yb.iloc[trb])
        rows.append(dict(train=name_b, test=name_b, horizon=hname, **metrics(yb.iloc[teb], np.clip(m_in.predict(Xb.iloc[teb]), 0, None), rated_b)))
        rows.append(dict(train=name_a, test=name_b, horizon=hname, **metrics(yb.iloc[teb], np.clip(m_tr.predict(Xb.iloc[teb]), 0, None), rated_b)))
    return pd.DataFrame(rows)
