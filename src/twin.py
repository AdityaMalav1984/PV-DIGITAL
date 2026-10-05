"""Digital twin, uncertainty intervals, validation-calibrated anomaly scoring."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import *


def split_times(df):
    t = np.sort(df.DATE_TIME.unique())
    return t[int(len(t) * TRAIN_FRAC)], t[int(len(t) * (TRAIN_FRAC + VAL_FRAC))]


def _with_peer_feature(df):
    out = df.copy()
    w = out.pivot(index="DATE_TIME", columns="INVERTER", values="AC_POWER")
    peer = w.apply(lambda r: r.median() if len(r.dropna()) == 1 else np.nan, axis=1)
    # efficient leave-one-out peer median
    peer_med = pd.DataFrame(index=w.index, columns=w.columns, dtype=float)
    for c in w.columns:
        peer_med[c] = w.drop(columns=c).median(axis=1)
    pm = peer_med.stack(future_stack=True).rename("PEER_MEDIAN_AC").reset_index()
    return out.merge(pm, on=["DATE_TIME", "INVERTER"], how="left")


BASE_FEATURES = TWIN_FEATURES
MODEL_FEATURES = TWIN_FEATURES + ["PEER_MEDIAN_AC"]

def _model(loss="absolute_error"):
    return HistGradientBoostingRegressor(loss=loss, max_iter=160, learning_rate=0.06,
                                         random_state=42)


def fit_twin(df):
    """Fit a fleet-wide benchmark and per-inverter operational twins.

    The global model is retained for ablation/generalisation. Per-inverter models
    define each inverter's normal behaviour without using lagged power, so faults
    cannot be hidden by autoregression.
    """
    t_val, t_test = split_times(df)
    tr = _with_peer_feature(df[df.DATE_TIME < t_val].copy())
    tr = tr[~((tr.AC_POWER < 1) & (tr.IRRADIATION > 0.1))]
    features = MODEL_FEATURES
    global_model = _model().fit(tr[features], tr.AC_POWER)
    models = {}
    for inv, g in tr.groupby("INVERTER"):
        if len(g) < 200:
            continue
        models[inv] = _model().fit(g[features], g.AC_POWER)
    rated = float(tr.AC_POWER.quantile(0.995))
    tw = {"model": global_model, "models": models, "features": features, "rated": rated,
          "irr_max": float(tr.IRRADIATION.max()), "t_val": t_val, "t_test": t_test,
          "twin_mode": "per_inverter"}
    # Calibrate anomaly thresholds from validation predictions only.
    val = df[(df.DATE_TIME >= t_val) & (df.DATE_TIME < t_test)].copy()
    val["expected"] = predict(tw, val)
    val["under_ratio"] = ((val.expected - val.AC_POWER) / (val.expected + 1e-6)).where(val.expected > MIN_EXPECTED_FRAC * rated)
    thresholds = {}
    for inv, g in val.groupby("INVERTER"):
        r = g.under_ratio.dropna().to_numpy()
        if len(r) < 200:
            continue
        med = float(np.median(r)); mad = float(np.median(np.abs(r - med)) + 1e-9)
        thresholds[inv] = {
            "warn": float(max(np.quantile(r, .99), med + 3 * 1.4826 * mad)),
            "high": float(max(np.quantile(r, .999), med + 5 * 1.4826 * mad)),
            "median": med, "mad": mad,
        }
    tw["thresholds"] = thresholds
    return tw


def predict(twin, df):
    df = _with_peer_feature(df)
    df[twin["features"]] = df[twin["features"]].fillna(0)
    out = np.empty(len(df), dtype=float)
    used = np.zeros(len(df), dtype=bool)
    if twin.get("models"):
        for inv, idx in df.groupby("INVERTER").groups.items():
            pos = df.index.get_indexer(idx)
            if inv in twin["models"]:
                out[pos] = twin["models"][inv].predict(df.loc[idx, twin["features"]])
                used[pos] = True
    if not used.all():
        out[~used] = twin["model"].predict(df.loc[~used, twin["features"]])
    out = np.clip(out, 0, None)
    return np.where(df["IRRADIATION"].values <= 0, 0.0, out)


def predict_global(twin, df):
    df = _with_peer_feature(df)
    df[twin["features"]] = df[twin["features"]].fillna(0)
    p = np.clip(twin["model"].predict(df[twin["features"]]), 0, None)
    return np.where(df["IRRADIATION"].values <= 0, 0.0, p)


def fit_prediction_intervals(twin, df, quantiles=(0.05, 0.95)):
    """Fit per-inverter quantile twins on the training split for uncertainty bands."""
    t_val = twin["t_val"]
    tr = _with_peer_feature(df[df.DATE_TIME < t_val].copy())
    tr = tr[~((tr.AC_POWER < 1) & (tr.IRRADIATION > 0.1))]
    qmodels = {q: {} for q in quantiles}
    for inv, g in tr.groupby("INVERTER"):
        if len(g) < 200:
            continue
        for q in quantiles:
            qmodels[q][inv] = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                max_iter=120, learning_rate=0.06, random_state=42).fit(g[twin["features"]], g.AC_POWER)
    twin["quantile_models"] = qmodels
    return twin


def predict_interval(twin, df):
    df2 = _with_peer_feature(df.copy())
    df2[twin["features"]] = df2[twin["features"]].fillna(0)
    lo = np.full(len(df2), np.nan); hi = np.full(len(df2), np.nan)
    qmodels = twin.get("quantile_models", {})
    for inv, idx in df2.groupby("INVERTER").groups.items():
        pos = df2.index.get_indexer(idx)
        if inv in qmodels.get(.05, {}) and inv in qmodels.get(.95, {}):
            X = df2.loc[idx, twin["features"]]
            lo[pos] = np.clip(qmodels[.05][inv].predict(X), 0, None)
            hi[pos] = np.clip(qmodels[.95][inv].predict(X), 0, None)
    return lo, hi


def wide(df, col):
    return df.pivot(index="DATE_TIME", columns="INVERTER", values=col)


def score_matrices(A, E, rated, irr):
    valid_np = (E.values > MIN_EXPECTED_FRAC * rated) & (irr.reindex(E.index).values[:, None] > DAY_IRR) & A.notna().values
    valid = pd.DataFrame(valid_np, index=E.index, columns=E.columns)
    Ev, Av = E.where(valid), A.where(valid)
    ratio = (Ev - Av) / (Ev + 1e-6)
    c1 = (ratio.rolling(4, min_periods=2).mean().clip(lower=0) / NR_CAP).clip(upper=1)
    flag = (ratio > NR_THR).astype(float).where(valid)
    c2 = flag.rolling(8, min_periods=4).mean()
    PR = Av / (Ev + 1e-6)
    med = PR.median(axis=1).clip(lower=0.1)
    dev = (-PR.sub(PR.median(axis=1), axis=0)).div(med, axis=0)
    c3 = (dev.rolling(4, min_periods=2).mean().clip(lower=0) / NR_CAP).clip(upper=1)
    score = (W1 * c1 + W2 * c2 + W3 * c3).where(valid)
    return score, ratio, PR


def status_from_score(s):
    s = np.asarray(s, dtype=float)
    return np.select([np.isnan(s), s < WARN, s < HIGH], ["N/A (night)", "Normal", "Warning"], "High")


def score_long(df, twin):
    df = df.copy()
    df["expected_global"] = predict_global(twin, df)
    df["expected"] = predict(twin, df)
    df["residual"] = df.AC_POWER - df.expected
    lo, hi = predict_interval(twin, df) if twin.get("quantile_models") else (np.full(len(df), np.nan), np.full(len(df), np.nan))
    df["expected_lo"], df["expected_hi"] = lo, hi
    A, E = wide(df, "AC_POWER"), wide(df, "expected")
    irr = df.groupby("DATE_TIME").IRRADIATION.first()
    score, ratio, PR = score_matrices(A, E, twin["rated"], irr)
    for name, M in (("score", score), ("under_ratio", ratio), ("perf_ratio", PR)):
        df = df.merge(M.stack(future_stack=True).rename(name).reset_index(), on=["DATE_TIME", "INVERTER"], how="left")
    # EWMA smooths one-sample noise. Alert status uses validation-calibrated ratio thresholds.
    df["underperf_ewma"] = df.groupby("INVERTER")["under_ratio"].transform(lambda s: s.ewm(alpha=.3, adjust=False).mean())
    def status_row(r):
        if pd.isna(r.underperf_ewma): return "N/A (night)"
        th = twin.get("thresholds", {}).get(r.INVERTER)
        if not th: return status_from_score(r.score)
        if r.underperf_ewma >= th["high"]: return "High"
        if r.underperf_ewma >= th["warn"]: return "Warning"
        return "Normal"
    df["status"] = [status_row(r) for r in df[["underperf_ewma", "INVERTER", "score"]].itertuples(index=False)]
    return df


def twin_accuracy(sdf, twin):
    te = sdf[sdf.DATE_TIME >= twin["t_test"]]
    res = {"inverter_MAE_kW": float((te.AC_POWER - te.expected).abs().mean()),
           "inverter_nMAE_%": float(100 * (te.AC_POWER - te.expected).abs().mean() / twin["rated"]),
           "inverter_R2": float(1 - ((te.AC_POWER - te.expected) ** 2).sum() / ((te.AC_POWER - te.AC_POWER.mean()) ** 2).sum())}
    ok = ~((te.expected > MIN_EXPECTED_FRAC * twin["rated"]) & (te.AC_POWER < 1))
    res["inverter_nMAE_excl_zero_output_%"] = float(100 * (te.AC_POWER - te.expected).abs()[ok].mean() / twin["rated"])
    tot = te.groupby("DATE_TIME")[["AC_POWER", "expected"]].sum()
    n_inv = te.INVERTER.nunique()
    res["plant_nMAE_%"] = float(100 * (tot.AC_POWER - tot.expected).abs().mean() / (n_inv * twin["rated"]))
    # Benchmark against fleet-wide model for the ablation.
    eg = te.expected_global
    res["global_twin_nMAE_%"] = float(100 * (te.AC_POWER - eg).abs().mean() / twin["rated"])
    return res


def inverter_health_table(sdf, t0, t1):
    w = sdf[(sdf.DATE_TIME >= t0) & (sdf.DATE_TIME < t1) & sdf.score.notna()]
    g = w.groupby("INVERTER")
    out = pd.DataFrame({"mean_perf_ratio": g.perf_ratio.mean(), "mean_score": g.score.mean(),
                        "pct_warning_or_high_%": 100 * g.status.apply(lambda s: s.isin(["Warning", "High"]).mean()),
                        "pct_high_%": 100 * g.status.apply(lambda s: (s == "High").mean()),
                        "median_perf_ratio": g.perf_ratio.median(), "zero_output_daytime_%": 100 * g.AC_POWER.apply(lambda x: (x < 1).mean()),
                        "mean_underperf_%": 100 * g.under_ratio.mean()})
    return out.sort_values("mean_score", ascending=False)


def degradation_experiment(sdf, twin, levels=(0, .05, .10, .15, .20, .30)):
    te = sdf[sdf.DATE_TIME >= twin["t_test"]]
    A0, E = wide(te, "AC_POWER"), wide(te, "expected")
    A0 = A0.mask((A0 < 1) & (E > MIN_EXPECTED_FRAC * twin["rated"]))
    irr = te.groupby("DATE_TIME").IRRADIATION.first(); days = pd.Series(A0.index.date, index=A0.index); rows = []
    for d in levels:
        for inv in A0.columns:
            A = A0.copy(); A[inv] = A[inv] * (1 - d)
            score, _, _ = score_matrices(A, E, twin["rated"], irr); s = score[inv].dropna()
            dd = (s >= WARN).groupby(days.reindex(s.index)).mean()
            rows.append(dict(degradation=d, inverter=inv, interval_detect_rate=(s >= WARN).mean(), high_rate=(s >= HIGH).mean(), day_detect_rate=(dd >= .5).mean(), mean_score=s.mean()))
    return pd.DataFrame(rows)
