"""Modules 1-2: ingestion, cleaning, timestamp synchronisation."""
import numpy as np
import pandas as pd
from .config import DATA, DAY_IRR, STEP_MIN


def _parse_dt(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    if s.str.match(r"^\d{2}-\d{2}-\d{4}").all():          # Plant-1 generation uses dd-mm-yyyy
        return pd.to_datetime(s, format="%d-%m-%Y %H:%M")
    return pd.to_datetime(s)


def add_time_features(df: pd.DataFrame, col="DATE_TIME") -> pd.DataFrame:
    h = df[col].dt.hour + df[col].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return df


def load_plant(p: int):
    """Return (inverter-level merged dataframe, cleaning report dict)."""
    gen = pd.read_csv(DATA / f"Plant_{p}_Generation_Data.csv")
    wx = pd.read_csv(DATA / f"Plant_{p}_Weather_Sensor_Data.csv")
    rep = {"plant": p, "gen_rows_raw": len(gen), "weather_rows_raw": len(wx)}

    gen["DATE_TIME"] = _parse_dt(gen["DATE_TIME"])
    wx["DATE_TIME"] = _parse_dt(wx["DATE_TIME"])

    rep["missing_values"] = int(gen.isna().sum().sum() + wx.isna().sum().sum())
    rep["duplicate_gen_rows"] = int(gen.duplicated(["DATE_TIME", "SOURCE_KEY"]).sum())
    gen = gen.drop_duplicates(["DATE_TIME", "SOURCE_KEY"]).dropna()
    wx = wx.drop_duplicates("DATE_TIME").dropna()

    n0 = len(gen)
    gen = gen[(gen.AC_POWER >= 0) & (gen.DC_POWER >= 0)]
    wx = wx[(wx.IRRADIATION >= 0) & wx.AMBIENT_TEMPERATURE.between(-20, 70)
            & wx.MODULE_TEMPERATURE.between(-20, 100)]
    rep["invalid_rows_removed"] = n0 - len(gen)

    # Detect a plant-wide unit mismatch using the median DC/AC ratio, robust to outliers.
    day = gen[gen.AC_POWER > 50]
    ratio = float((day.DC_POWER / day.AC_POWER.replace(0, np.nan)).dropna().median()) if len(day) else 0.0
    rep["dc_ac_ratio_raw"] = round(ratio, 2)
    rep["dc_rescaled_by_10"] = ratio > 5
    if ratio > 5:
        gen["DC_POWER"] = gen["DC_POWER"] / 10
        print(f"[rescale] plant {p}: DC/AC median {ratio:.2f} -> divide DC by 10")
    # Remaining extreme ratios are capped for plotting/features; AC remains the target.
    ratio2 = gen.DC_POWER / gen.AC_POWER.replace(0, np.nan)
    mask = ratio2 > 2
    gen.loc[mask, "DC_POWER"] = gen.loc[mask, "AC_POWER"] * 2

    keys = sorted(gen.SOURCE_KEY.unique())
    gen["INVERTER"] = gen.SOURCE_KEY.map({k: f"INV-{i + 1:02d}" for i, k in enumerate(keys)})
    rep["n_inverters"] = len(keys)

    wx = wx[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df = gen.merge(wx, on="DATE_TIME", how="inner")
    rep["rows_without_weather"] = len(gen) - len(df)
    df["DAY"] = (df.IRRADIATION > DAY_IRR).astype(int)
    df = add_time_features(df)
    df = df.sort_values(["DATE_TIME", "INVERTER"]).reset_index(drop=True)
    df = df[["DATE_TIME", "INVERTER", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD",
             "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
             "DAY", "hour_sin", "hour_cos"]]

    full = pd.date_range(df.DATE_TIME.min(), df.DATE_TIME.max(), freq=f"{STEP_MIN}min")
    rep["period"] = f"{df.DATE_TIME.min():%Y-%m-%d} to {df.DATE_TIME.max():%Y-%m-%d}"
    rep["expected_timestamps"] = len(full)
    rep["timestamps_present"] = int(df.DATE_TIME.nunique())
    return df, rep


def plant_series(df: pd.DataFrame):
    """Plant-level total AC power (only where every inverter reported) + weather, on a regular grid."""
    wide = df.pivot(index="DATE_TIME", columns="INVERTER", values="AC_POWER")
    n = wide.shape[1]
    total = wide.sum(axis=1, min_count=n)
    total[wide.notna().sum(axis=1) < n] = np.nan
    wx = df.groupby("DATE_TIME")[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]].first()
    grid = pd.date_range(total.index.min(), total.index.max(), freq=f"{STEP_MIN}min")
    return total.reindex(grid).rename("AC_TOTAL"), wx.reindex(grid)
