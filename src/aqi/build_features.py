"""Build the modelling table: CPCB AQI target + leak-free features.

The task is forecasting AQI `horizon` hours ahead. Features may only use
information available at prediction time, so every pollutant-derived feature is
a lag or a trailing window -- never a same-timestamp concentration, which would
otherwise let the model invert the CPCB formula and score near-perfectly while
being useless in production.

One deliberate exception: weather AT the target hour. Weather is forecastable,
so a production caller can supply a forecast for the horizon hour; pollutant
concentrations cannot be forecast that way. These "lead" weather features were
the single biggest accuracy lever in the rolling-origin CV sweep (Winter MAE
25.0 -> 22.7), so they are included -- the serving contract requires the caller
to pass forecasted weather for the horizon hour.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

from .cpcb import compute_aqi
from .locations import INDIAN_HOLIDAYS

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
WEATHER = ["Temperature", "Humidity", "Pressure", "Wind_Speed",
           "Wind_Direction", "Precipitation", "Cloud_Cover"]
# Base + long lags. The long tail (96h..168h) lets the model see the same hour
# on previous days and the weekly cycle.
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 168]
ROLL_WINDOWS = [6, 24, 72]
# Weekly / fortnightly AQI context: how unusual is now vs the recent normal.
LONG_ROLLS = [168, 336]
# Lagged CPCB sub-indices say WHICH pollutant drives AQI and where it is
# heading. Functions of past concentrations only, so no leak.
SUB_LAGS = [1, 2, 24, 48]
# Weather is forecastable, so its value at the target hour is legitimately known
# at prediction time (from a forecast). Pollutant concentrations are not.
LEAD_WEATHER = ["Wind_Speed", "Precipitation", "Temperature", "Humidity",
                "Cloud_Cover", "Pressure"]


def add_temporal_features(df):
    dt = df["datetime"]
    df["Hour"] = dt.dt.hour
    df["Day"] = dt.dt.day
    df["Month"] = dt.dt.month
    df["Year"] = dt.dt.year
    df["DayOfWeek"] = dt.dt.dayofweek
    df["DayOfYear"] = dt.dt.dayofyear
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Is_Rush_Hour"] = df["Hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    df["Is_Night"] = df["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df["Is_Holiday"] = dt.dt.strftime("%Y-%m-%d").isin(INDIAN_HOLIDAYS).astype(int)
    df["Season"] = df["Month"].map(lambda m: 1 if m in (12, 1, 2) else
                                   2 if m in (3, 4, 5) else
                                   3 if m in (6, 7, 8, 9) else 4)
    # Cyclical encodings so the model sees 23:00 and 00:00 as adjacent.
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Wind_Sin"] = np.sin(np.radians(df["Wind_Direction"]))
    df["Wind_Cos"] = np.cos(np.radians(df["Wind_Direction"]))
    return df


def add_history_features(df, columns, lags, rolling=True):
    """Lags, trailing stats and deltas for one city's chronologically sorted frame."""
    new = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        for lag in lags:
            new[f"{col}_Lag_{lag}h"] = s.shift(lag)
        if rolling:
            # shift(1) first so a window never includes the current hour.
            prev = s.shift(1)
            for w in ROLL_WINDOWS:
                roll = prev.rolling(w, min_periods=max(2, w // 3))
                new[f"{col}_RollMean_{w}h"] = roll.mean()
                new[f"{col}_RollStd_{w}h"] = roll.std()
        for a, b in ((1, 2), (1, 24)):
            if a in lags and b in lags:
                new[f"{col}_Delta_{a}_{b}h"] = (new[f"{col}_Lag_{a}h"] -
                                                new[f"{col}_Lag_{b}h"])
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_lead_weather(df, horizon):
    """Weather at / over the forecast window -- available from a forecast at
    prediction time. Ventilation and rain over the window drive dispersion."""
    for c in LEAD_WEATHER:
        if c in df.columns:
            lead = df[c].shift(-horizon)
            df[f"{c}_Lead_{horizon}h"] = lead
            df[f"{c}_LeadDelta_{horizon}h"] = lead - df[c]
    if "Wind_Speed" in df.columns:
        df[f"Wind_Speed_LeadMean_{horizon}h"] = (
            df["Wind_Speed"].shift(-horizon).rolling(horizon, min_periods=6).mean())
    if "Precipitation" in df.columns:
        df[f"Precipitation_LeadSum_{horizon}h"] = (
            df["Precipitation"].shift(-horizon).rolling(horizon, min_periods=6).sum())
    return df


def build(raw_path, out_path, horizon, min_history):
    raw = pd.read_csv(raw_path, parse_dates=["datetime"])
    raw = raw.sort_values(["City", "datetime"]).reset_index(drop=True)
    print(f"Loaded {len(raw):,} rows, {raw['City'].nunique()} cities")

    history_cols = POLLUTANTS + WEATHER + ["AQI"]
    sub_cols = [f"SubIndex_{p}" for p in POLLUTANTS]
    # Same-timestamp pollutant concentrations and sub-indices leak the target.
    leaky = set(POLLUTANTS + sub_cols + ["AQI", "AQI_Category", "Dominant_Pollutant"])
    dropped = leaky | {"datetime", "target_AQI", "State", "City"}

    # Per city: the lag/rolling features are all within-city anyway, and holding
    # only one city's expanded frame at a time keeps peak memory ~36x lower.
    parts = []
    for city, g in raw.groupby("City", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        g = compute_aqi(g, group_col="City")
        g = add_temporal_features(g)
        g = add_history_features(g, history_cols, LAGS)

        sub_present = [c for c in sub_cols if c in g.columns]
        g = add_history_features(g, sub_present, SUB_LAGS, rolling=False)

        prev = g["AQI"].shift(1)
        for w in LONG_ROLLS:
            roll = prev.rolling(w, min_periods=w // 3)
            g[f"AQI_RollMean_{w}h"] = roll.mean()
            g[f"AQI_RollStd_{w}h"] = roll.std()
        # How unusual is right now versus this city's recent normal?
        g["AQI_Ratio_24h_168h"] = (g["AQI_RollMean_24h"] /
                                   g["AQI_RollMean_168h"].replace(0, np.nan))

        g = add_lead_weather(g, horizon)

        g["target_AQI"] = g["AQI"].shift(-horizon)
        g = g.dropna(subset=["target_AQI"])
        g = g[g[f"AQI_Lag_{min_history}h"].notna()]

        keep_cols = [c for c in g.columns if c not in dropped]
        part = g[["City", "State", "datetime"] + keep_cols + ["target_AQI"]].copy()
        for c in part.columns:
            if part[c].dtype == "float64":
                part[c] = part[c].astype("float32")
        parts.append(part)
        print(f"  {city:<22} {len(part):>7,} rows", flush=True)

    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df.sort_values(["City", "datetime"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in ("City", "State", "datetime", "target_AQI")]

    # Open-Meteo serves NH3 only over Europe, so its columns are entirely empty
    # for India. All-NaN/constant columns carry no signal and break the
    # histogram binner in HistGradientBoostingRegressor.
    degenerate = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if degenerate:
        print(f"Dropping {len(degenerate)} empty/constant columns "
              f"(e.g. {', '.join(degenerate[:3])})")
        feature_cols = [c for c in feature_cols if c not in degenerate]
        df = df.drop(columns=degenerate)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\nFeatures: {len(feature_cols)}  Rows: {len(df):,}")
    print(f"Range: {df['datetime'].min()} .. {df['datetime'].max()}")
    print(f"Target mean {df['target_AQI'].mean():.1f}, std {df['target_AQI'].std():.1f}")
    print(f"Saved -> {out_path}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/aqi/raw.csv")
    p.add_argument("--out", default="data/aqi/features.parquet")
    p.add_argument("--params", default="params.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.params))["aqi"]
    build(args.raw, args.out, cfg["horizon_hours"], cfg["min_history_hours"])


if __name__ == "__main__":
    main()
