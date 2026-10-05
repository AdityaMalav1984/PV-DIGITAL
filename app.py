"""Streamlit dashboard:  streamlit run app.py   (run `python run_pipeline.py` first)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from src.config import OUT, MODELS, WARN, HIGH
from src.simulate import simulate

st.set_page_config(page_title="PV Digital Twin", layout="wide")


@st.cache_data
def load(p):
    s = pd.read_csv(OUT / f"plant{p}_scored.csv.gz", parse_dates=["DATE_TIME"])
    f = pd.read_csv(OUT / f"plant{p}_forecast.csv", parse_dates=["target_time"])
    return s, f


@st.cache_resource
def get_twin(p):
    return joblib.load(MODELS / f"twin_plant{p}.joblib")


STATUS_COLOR = {"Normal": "🟢", "Warning": "🟠", "High": "🔴", "N/A (night)": "⚫"}

st.title("☀️ PV Digital Twin")
st.caption("Data-driven digital twin: expected vs actual PV output, inverter anomaly scoring and what-if simulation. "
           "Anomaly = deviation from learned normal behaviour, NOT a confirmed hardware fault.")

plant = st.sidebar.radio("Plant", [1, 2], format_func=lambda p: f"Plant {p}")
sdf, fdf = load(plant)
twin = get_twin(plant)
days = sorted(sdf.DATE_TIME.dt.date.unique())
day = st.sidebar.select_slider("Day", options=days, value=days[len(days) - 3])
dd = sdf[sdf.DATE_TIME.dt.date == day]
times = sorted(dd[dd.score.notna()].DATE_TIME.unique())
if not times:
    st.warning("No daytime scored data on this day."); st.stop()
t_sel = st.sidebar.select_slider("Time", options=times, value=times[len(times) // 2], format_func=lambda t: pd.Timestamp(t).strftime("%H:%M"))
now = dd[dd.DATE_TIME == t_sel].set_index("INVERTER")
if day >= pd.Timestamp(twin["t_test"]).date():
    st.sidebar.success("Day is in the held-out TEST period")
else:
    st.sidebar.info("Day is in the training/validation period (twin has seen this data)")

tab1, tab2, tab3, tab4 = st.tabs(["Plant overview", "Forecast", "Inverter health", "Simulation"])

with tab1:
    act, exp_ = now.AC_POWER.sum(), now.expected.sum()
    worst = "High" if (now.status == "High").any() else "Warning" if (now.status == "Warning").any() else "Normal"
    c = st.columns(4)
    c[0].metric("Current power", f"{act:,.0f} kW"); c[1].metric("Expected power", f"{exp_:,.0f} kW")
    c[2].metric("Deviation", f"{100 * (act - exp_) / max(exp_, 1):+.1f} %")
    c[3].metric("Plant status", {"Normal": "NORMAL", "Warning": "WATCH", "High": "ATTENTION"}[worst])
    tot = dd.groupby("DATE_TIME")[["AC_POWER", "expected", "IRRADIATION"]].agg({"AC_POWER": "sum", "expected": "sum", "IRRADIATION": "first"})
    fig = go.Figure()
    fig.add_scatter(x=tot.index, y=tot.AC_POWER, name="Actual"); fig.add_scatter(x=tot.index, y=tot.expected, name="Expected (twin)")
    fig.add_vline(x=t_sel, line_dash="dot"); fig.update_layout(yaxis_title="Plant AC power (kW)", height=380, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Plant total only reliable when all inverters report; missing rows lower the 'actual' line.")

with tab2:
    hz = st.selectbox("Forecast horizon", sorted(fdf.horizon.unique(), key=lambda s: {"15min": 1, "1h": 4, "4h": 16}.get(s, 99)), index=1)
    f = fdf[fdf.horizon == hz]
    lo, hi = f.target_time.min(), f.target_time.max()
    d0 = st.slider("Start", lo.to_pydatetime(), hi.to_pydatetime(), lo.to_pydatetime(), format="DD/MM HH:mm")
    w = f[(f.target_time >= d0) & (f.target_time < pd.Timestamp(d0) + pd.Timedelta(days=3))]
    fig = go.Figure()
    fig.add_scatter(x=w.target_time, y=w.actual, name="Actual", line=dict(color="black"))
    fig.add_scatter(x=w.target_time, y=w.prediction, name=f"Forecast ({w.model.iloc[0] if len(w) else ''})")
    fig.add_scatter(x=w.target_time, y=w.persistence, name="Persistence", line=dict(dash="dot"))
    fig.update_layout(yaxis_title="Plant AC power (kW)", height=400, margin=dict(t=20)); st.plotly_chart(fig, use_container_width=True)
    st.caption("Held-out test-period forecasts (weather measured at forecast origin).")

with tab3:
    tbl = pd.DataFrame({"Expected (kW)": now.expected.round(0), "Actual (kW)": now.AC_POWER.round(0),
                        "Deviation %": (100 * (now.AC_POWER - now.expected) / now.expected.clip(lower=1)).round(1),
                        "Score": now.score.round(2), "Status": [STATUS_COLOR[s] + " " + s for s in now.status]})
    st.subheader(f"Inverter status at {pd.Timestamp(t_sel):%d %b %H:%M}")
    st.dataframe(tbl, use_container_width=True, height=420)
    piv = dd.pivot(index="INVERTER", columns="DATE_TIME", values="score")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, zmin=0, zmax=1, colorscale="YlOrRd", colorbar=dict(title="score")))
    fig.update_layout(height=520, title=f"Anomaly score, {day} (blank = night / not evaluated)", margin=dict(t=40)); st.plotly_chart(fig, use_container_width=True)
    inv = st.selectbox("Inspect inverter", sorted(dd.INVERTER.unique()))
    x = dd[dd.INVERTER == inv]
    fig = go.Figure(); fig.add_scatter(x=x.DATE_TIME, y=x.AC_POWER, name="Actual"); fig.add_scatter(x=x.DATE_TIME, y=x.expected, name="Expected")
    fig.update_layout(height=320, yaxis_title="kW", margin=dict(t=20)); st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Score bands (prototype, not validated): Normal < {WARN}, Warning {WARN}-{HIGH}, High ≥ {HIGH}.")

with tab4:
    st.subheader("What-if simulation")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Irradiance multiplier", 0.3, 1.5, 1.0, 0.05)
    dT = c1.slider("Temperature change (°C)", -10, 15, 0)
    tm = c1.radio("Temperature effect", ["physical", "model"], help="physical: power × (1 + coeff·ΔT); model: ΔT fed to the ML twin (can be unreliable, see README)")
    coeff = c1.number_input("Temp. coefficient (per °C)", value=-0.004, format="%.4f", disabled=(tm != "physical"))
    invs = c2.multiselect("Affected inverter(s)", sorted(sdf.INVERTER.unique()), default=["INV-04"])
    deg = c2.slider("Degradation of affected inverter(s) (%)", 0, 60, 20)
    outage = c2.checkbox("Outage (0 kW) of affected inverter(s)")
    start = c3.selectbox("Start time", [f"{h:02d}:00" for h in range(6, 19)], index=3)
    dur = c3.slider("Duration (h)", 1, 12, 4)
    if st.button("RUN SIMULATION", type="primary"):
        t0 = pd.Timestamp(f"{day} {start}"); t1 = t0 + pd.Timedelta(hours=dur)
        win = dd[(dd.DATE_TIME >= t0) & (dd.DATE_TIME < t1)]
        if win.empty:
            st.error("No data in that window.")
        else:
            summ, per, ts = simulate(twin, win, irr_mult=k, d_temp=dT, degradation={i: deg / 100 for i in invs},
                                     outage=invs if outage else [], temp_method=tm, temp_coeff=coeff)
            m = st.columns(4)
            m[0].metric("Baseline energy", f"{summ['baseline_MWh']:.2f} MWh"); m[1].metric("Simulated energy", f"{summ['simulated_MWh']:.2f} MWh")
            m[2].metric("Energy loss", f"{summ['loss_MWh']:.2f} MWh", f"{-summ['loss_%']:.1f} %")
            aff = per.loc[[i for i in invs if i in per.index]] if invs else per
            worst = "High" if (aff.status == "High").any() else "Warning" if (aff.status == "Warning").any() else "Normal"
            m[3].metric("Anomaly status (affected inverters)", worst)
            fig = go.Figure(); fig.add_scatter(x=ts.index, y=ts.baseline_kW, name="Baseline (measured)"); fig.add_scatter(x=ts.index, y=ts.simulated_kW, name="Simulated")
            fig.update_layout(height=340, yaxis_title="Plant AC (kW)", margin=dict(t=20)); st.plotly_chart(fig, use_container_width=True)
            st.dataframe(per.round(2), use_container_width=True)
    st.caption("Model-based what-if, not a validated physical simulation. Irradiance is capped at the maximum seen in training.")
