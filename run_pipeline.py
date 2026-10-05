"""Runs the whole PV Digital Twin pipeline (steps 1-9) and writes tables, figures, models, scored data."""
import json, datetime as dt, sklearn, joblib, warnings
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import *
from src.data import load_plant, plant_series
from src.forecast import run_forecast_experiments, cross_plant_transfer
from src.twin import (fit_twin, fit_prediction_intervals, score_long, twin_accuracy, inverter_health_table, degradation_experiment, wide)
from src.simulate import simulate

warnings.filterwarnings("ignore")
for d in (OUT, FIG, MODELS, OUT / "tables"):
    d.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
TAB = OUT / "tables"
data, reports, series, twins, scored = {}, [], {}, {}, {}
fc_all, health_all, deg_all, scen_all, twin_acc = [], [], [], [], []

for p in (1, 2):
    name = f"Plant {p}"
    print(f"=== {name} ===")
    df, rep = load_plant(p); data[p] = df; reports.append(rep)
    series[p] = plant_series(df); total, wx = series[p]
    rep["complete_plant_timestamps"] = int(total.notna().sum())

    # ---- forecasting experiments
    fc, preds, rated_plant = run_forecast_experiments(total, wx, name)
    fc_all.append(fc)
    # ---- digital twin + anomaly scoring
    tw = fit_twin(df); tw = fit_prediction_intervals(tw, df); twins[p] = tw
    sdf = score_long(df, tw); scored[p] = sdf
    acc = {"plant": name, **twin_accuracy(sdf, tw)}; twin_acc.append(acc)
    t_end = sdf.DATE_TIME.max() + pd.Timedelta(minutes=15)
    health = inverter_health_table(sdf, tw["t_test"], t_end).assign(plant=name); health_all.append(health.reset_index())
    val = sdf[(sdf.DATE_TIME >= tw["t_val"]) & (sdf.DATE_TIME < tw["t_test"]) & sdf.score.notna()]
    rep["val_share_warning_or_high_%"] = float(100 * val.status.isin(["Warning", "High"]).mean())
    rep["val_share_high_%"] = float(100 * (val.status == "High").mean())
    rep["val_p99_underperformance_ratio"] = float(val.under_ratio.quantile(0.99))
    rep["val_p999_underperformance_ratio"] = float(val.under_ratio.quantile(0.999))
    # ---- degradation-injection experiment
    deg = degradation_experiment(sdf, tw).assign(plant=name); deg_all.append(deg)
    # ---- scenarios over the test period
    win = sdf[sdf.DATE_TIME >= tw["t_test"]]
    inv = health.sort_values("median_perf_ratio").index[len(health) // 2]   # a typical inverter
    scen = {"Baseline (measured)": {}, "Irradiance x0.8": dict(irr_mult=.8), "Irradiance x1.2": dict(irr_mult=1.2),
            "Temp +5C (physical -0.4%/C)": dict(d_temp=5), "Temp +5C (raw ML twin)": dict(d_temp=5, temp_method="model"),
            f"{inv} degradation 10%": dict(degradation={inv: .10}), f"{inv} degradation 20%": dict(degradation={inv: .20}),
            f"{inv} outage": dict(outage=[inv])}
    for sname, kw in scen.items():
        summ, per, _ = simulate(tw, win, **kw)
        scen_all.append({"plant": name, "scenario": sname, **summ, "monitored_inverter": inv,
                         "inverter_score": per.loc[inv, "anomaly_score"], "inverter_status": per.loc[inv, "status"]})
    # ---- save artefacts
    joblib.dump(tw, MODELS / f"twin_plant{p}.joblib")
    with open(MODELS / f"twin_plant{p}.json", "w") as mf:
        json.dump({"saved_at": dt.datetime.now(dt.timezone.utc).isoformat(), "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__, "features": list(tw["features"]), "plant": p, "date_range": rep["period"], "n_rows": len(df), "twin_mode": tw.get("twin_mode"), "n_inverter_models": len(tw.get("models", {})), "thresholds_calibrated": len(tw.get("thresholds", {}))}, mf, indent=2)
    keep = ["DATE_TIME", "INVERTER", "AC_POWER", "DC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
            "hour_sin", "hour_cos", "expected", "residual", "perf_ratio", "under_ratio", "score", "status"]
    sdf[keep].round(4).to_csv(OUT / f"plant{p}_scored.csv.gz", index=False)
    rows = []
    for hname, h in HORIZONS.items():       # best weather-aware (measured weather) model per horizon -> dashboard
        sub = fc[(fc.horizon == hname) & (fc.variant == "weather_now")].sort_values("MAE")
        best = sub.iloc[0].model
        a, pr, ps = preds[(hname, "actual")], preds[(hname, best, "weather_now")], preds[(hname, "Persistence")]
        rows.append(pd.DataFrame({"horizon": hname, "target_time": a.index + pd.Timedelta(minutes=STEP_MIN * h),
                                  "actual": a.values, "prediction": pr.values, "persistence": ps.values, "model": best}))
    pd.concat(rows).to_csv(OUT / f"plant{p}_forecast.csv", index=False)

    # ================= FIGURES =================
    tag = f"plant{p}"
    # Fig 3 / 4 / 5
    d = df[df.DAY == 1].sample(min(12000, (df.DAY == 1).sum()), random_state=0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].scatter(d.IRRADIATION, d.AC_POWER, s=2, alpha=.3); ax[0].set(xlabel="Irradiation (kW/m²)", ylabel="Inverter AC power (kW)", title=f"{name}: AC power vs irradiance")
    sc = ax[1].scatter(d.IRRADIATION, d.AC_POWER, c=d.MODULE_TEMPERATURE, s=2, cmap="viridis"); plt.colorbar(sc, label="Module temp (°C)"); ax[1].set(xlabel="Irradiation (kW/m²)", title="coloured by module temperature")
    fig.tight_layout(); fig.savefig(FIG / f"fig3_{tag}_power_vs_irradiance.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 4)); ax.scatter(d.DC_POWER, d.AC_POWER, s=2, alpha=.3); mx = d.DC_POWER.max(); ax.plot([0, mx], [0, mx], "r--", lw=1, label="AC = DC")
    ax.set(xlabel="DC power (kW)", ylabel="AC power (kW)", title=f"{name}: AC vs DC power"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / f"fig4_{tag}_ac_vs_dc.png"); plt.close(fig)
    prof = df.assign(hour=df.DATE_TIME.dt.hour + df.DATE_TIME.dt.minute / 60).groupby("hour").agg(AC=("AC_POWER", "mean"), irr=("IRRADIATION", "mean"))
    fig, ax = plt.subplots(figsize=(6, 3.4)); ax.plot(prof.index, prof.AC, label="mean inverter AC (kW)"); ax.set(xlabel="Hour of day", ylabel="kW", title=f"{name}: daily profile")
    a2 = ax.twinx(); a2.plot(prof.index, prof.irr, "orange", label="irradiation"); a2.set_ylabel("kW/m²"); a2.grid(False); fig.tight_layout(); fig.savefig(FIG / f"fig5_{tag}_daily_profile.png"); plt.close(fig)
    # Fig 6 actual vs predicted (1 h, 3 test days)
    a, pr, ps = preds[("1h", "actual")], preds[("1h", fc[(fc.horizon == "1h") & (fc.variant == "weather_now")].sort_values("MAE").iloc[0].model, "weather_now")], preds[("1h", "Persistence")]
    a3 = a.iloc[:288]
    fig, ax = plt.subplots(figsize=(9, 3.4)); ax.plot(a3.index, a3.values, "k", label="actual (t+1h)"); ax.plot(a3.index, pr.loc[a3.index].values, label="weather-aware ML"); ax.plot(a3.index, ps.loc[a3.index].values, ":", label="persistence")
    ax.set(ylabel="Plant AC (kW)", title=f"{name}: 1-hour-ahead forecast (test period; x = forecast origin)"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / f"fig6_{tag}_actual_vs_predicted.png"); plt.close(fig)
    # Fig 7 model comparison (1h) and Fig 8 error vs horizon
    f1 = fc[fc.horizon == "1h"].copy(); f1["label"] = f1.model + np.where(f1.variant == "-", "", " | " + f1.variant)
    f1 = f1.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(7, 4.2)); ax.barh(f1.label, f1.RMSE); ax.invert_yaxis(); ax.set(xlabel="RMSE (kW)", title=f"{name}: 1-hour-ahead RMSE by model"); fig.tight_layout(); fig.savefig(FIG / f"fig7_{tag}_model_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for lab, sel in {"Persistence": fc.model == "Persistence", "RF history-only": (fc.model == "RandomForest") & (fc.variant == "history"),
                     "RF + measured weather": (fc.model == "RandomForest") & (fc.variant == "weather_now"),
                     "RF + future weather (oracle)": (fc.model == "RandomForest") & (fc.variant == "weather_future")}.items():
        s = fc[sel].set_index("horizon").loc[list(HORIZONS)]; ax.plot(list(HORIZONS), s["nMAE_%"], "o-", label=lab)
    ax.set(ylabel="nMAE (% of rated)", xlabel="Forecast horizon", title=f"{name}: error vs horizon"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / f"fig8_{tag}_error_vs_horizon.png"); plt.close(fig)
    # Fig 9 inverter performance
    fig, ax = plt.subplots(figsize=(8, 3.6)); h2 = health.sort_index(); ax.bar(h2.index, h2.median_perf_ratio, color=np.where(h2.median_perf_ratio < .9, "tab:red", "tab:blue"))
    ax.axhline(1, color="k", lw=.8); ax.set(ylim=(.6, 1.15), ylabel="median actual/expected", title=f"{name}: inverter performance ratio (test period)"); plt.xticks(rotation=90); fig.tight_layout(); fig.savefig(FIG / f"fig9_{tag}_inverter_performance.png"); plt.close(fig)
    # Fig 10 residual / anomaly score for worst inverter over 3 test days
    worst = health.index[0]; w = sdf[(sdf.INVERTER == worst) & (sdf.DATE_TIME >= tw["t_test"])].iloc[:288]
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True); ax[0].plot(w.DATE_TIME, w.AC_POWER, "k", label="actual"); ax[0].plot(w.DATE_TIME, w.expected, label="expected (twin)"); ax[0].set_ylabel("kW"); ax[0].legend(); ax[0].set_title(f"{name}: {worst} (highest mean anomaly score in test period)")
    ax[1].plot(w.DATE_TIME, w.score, "r"); ax[1].axhline(WARN, ls="--", c="orange"); ax[1].axhline(HIGH, ls="--", c="red"); ax[1].set(ylabel="anomaly score", ylim=(0, 1.05)); fig.tight_layout(); fig.savefig(FIG / f"fig10_{tag}_residual_anomaly_score.png"); plt.close(fig)
    # Fig 11 degradation detection
    g = deg.groupby("degradation")[["interval_detect_rate", "high_rate", "day_detect_rate"]].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.6)); ax.plot(g.index * 100, g.interval_detect_rate, "o-", label="intervals ≥ Warning"); ax.plot(g.index * 100, g.high_rate, "s-", label="intervals = High"); ax.plot(g.index * 100, g.day_detect_rate, "^-", label="days flagged (≥50% intervals)")
    ax.set(xlabel="Injected degradation (%)", ylabel="detection rate", title=f"{name}: detection vs injected degradation"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / f"fig11_{tag}_degradation_detection.png"); plt.close(fig)
    # Fig 12 energy loss by scenario
    sc_df = pd.DataFrame([r for r in scen_all if r["plant"] == name]); sc_df = sc_df[sc_df.scenario != "Baseline (measured)"]
    fig, ax = plt.subplots(figsize=(7, 3.8)); ax.barh(sc_df.scenario, sc_df.loss_MWh, color=np.where(sc_df.loss_MWh >= 0, "tab:red", "tab:green")); ax.invert_yaxis(); ax.axvline(0, c="k", lw=.8)
    ax.set(xlabel="Energy loss over test period (MWh, negative = gain)", title=f"{name}: scenario energy impact"); fig.tight_layout(); fig.savefig(FIG / f"fig12_{tag}_scenario_energy_loss.png"); plt.close(fig)

# ---- cross-plant generalisation (RQ5)
xp = pd.concat([cross_plant_transfer(series[1], series[2], "Plant 1", "Plant 2"), cross_plant_transfer(series[2], series[1], "Plant 2", "Plant 1")])
# twin transfer: fit on A (train part), evaluate on B test
import src.twin as T
from sklearn.ensemble import HistGradientBoostingRegressor
tr_rows = []
for a, b in ((1, 2), (2, 1)):
    te = scored[b][scored[b].DATE_TIME >= twins[b]["t_test"]]
    e = T.predict(twins[a], te); ok = ~((e > MIN_EXPECTED_FRAC * twins[b]["rated"]) & (te.AC_POWER < 1))
    own = te.expected
    tr_rows.append({"twin_trained_on": f"Plant {a}", "tested_on": f"Plant {b}", "nMAE_%_excl_zero_output": 100 * np.abs(e - te.AC_POWER)[ok].mean() / twins[b]["rated"],
                    "own_plant_twin_nMAE_%": 100 * np.abs(own - te.AC_POWER)[ok].mean() / twins[b]["rated"]})

pd.concat(fc_all).round(3).to_csv(TAB / "forecast_performance.csv", index=False)
pd.concat(health_all).round(3).to_csv(TAB / "inverter_health_test_period.csv", index=False)
pd.concat(deg_all).round(3).to_csv(TAB / "degradation_experiment.csv", index=False)
pd.DataFrame(scen_all).round(3).to_csv(TAB / "scenario_results.csv", index=False)
pd.DataFrame(twin_acc).round(3).to_csv(TAB / "twin_accuracy.csv", index=False)
xp.round(3).to_csv(TAB / "cross_plant_forecast_transfer.csv", index=False)
pd.DataFrame(tr_rows).round(3).to_csv(TAB / "cross_plant_twin_transfer.csv", index=False)
json.dump(reports, open(OUT / "data_quality_report.json", "w"), indent=2, default=str)
print("done; xgboost available:", any("XGBoost" in set(f.model) for f in fc_all))
