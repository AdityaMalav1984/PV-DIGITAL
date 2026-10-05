"""Module 7: scenario simulator (model-based what-if, not a validated physical model)."""
import numpy as np
import pandas as pd
from .config import DT_H
from .twin import predict, wide, score_matrices, status_from_score


def simulate(twin, win, irr_mult=1.0, d_temp=0.0, degradation=None, outage=None,
             temp_method="physical", temp_coeff=-0.004):
    """win: long dataframe for the chosen window (output of score_long).
    degradation: {inverter: fraction}, outage: iterable of inverters set to 0 kW.
    temp_method: "physical" -> power scaled by (1 + temp_coeff * d_temp)  [assumed coefficient, editable]
                 "model"    -> d_temp is fed to the ML twin (only ~34 days of data: temperature is collinear
                               with irradiance, so this response can be unreliable - see README)."""
    degradation, outage = degradation or {}, set(outage or [])
    s = win.copy()
    s["IRRADIATION"] = np.minimum(s.IRRADIATION * irr_mult, twin["irr_max"])   # trees cannot extrapolate
    if temp_method == "model":
        s["AMBIENT_TEMPERATURE"] = s.AMBIENT_TEMPERATURE + d_temp
        s["MODULE_TEMPERATURE"] = s.MODULE_TEMPERATURE + d_temp
    s["expected_sim"] = predict(twin, s)
    if temp_method == "physical":
        s["expected_sim"] = s.expected_sim * max(0.0, 1 + temp_coeff * d_temp)
    ratio = np.where(win.expected.values > 1.0, s.expected_sim.values / np.maximum(win.expected.values, 1e-6), 1.0)
    ratio = np.clip(ratio, 0, 3)
    deg = s.INVERTER.map(degradation).fillna(0).values
    s["actual_sim"] = win.AC_POWER.values * ratio * (1 - deg)   # keeps real noise / any real anomaly
    s.loc[s.INVERTER.isin(outage), "actual_sim"] = 0.0

    per = s.groupby("INVERTER").agg(baseline_kWh=("AC_POWER", lambda x: x.sum() * DT_H),
                                    simulated_kWh=("actual_sim", lambda x: x.sum() * DT_H))
    per["loss_kWh"] = per.baseline_kWh - per.simulated_kWh
    A, E = wide(s, "actual_sim"), wide(s, "expected_sim")
    irr = s.groupby("DATE_TIME").IRRADIATION.first()
    score, _, _ = score_matrices(A, E, twin["rated"], irr)
    per["anomaly_score"] = score.median()
    per["status"] = status_from_score(per.anomaly_score)
    ts = s.groupby("DATE_TIME").agg(baseline_kW=("AC_POWER", "sum"), simulated_kW=("actual_sim", "sum"),
                                    expected_sim_kW=("expected_sim", "sum"))
    summary = {"baseline_MWh": per.baseline_kWh.sum() / 1000, "simulated_MWh": per.simulated_kWh.sum() / 1000,
               "loss_MWh": per.loss_kWh.sum() / 1000,
               "loss_%": 100 * per.loss_kWh.sum() / max(per.baseline_kWh.sum(), 1e-9)}
    return summary, per, ts
