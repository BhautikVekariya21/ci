"""Build the modelling table: CPCB AQI target + leak-free features.

The task is forecasting AQI `horizon` hours ahead. Features may only use
information available at prediction time, so every pollutant-derived feature is
a lag or a trailing window -- never a same-timestamp concentration, which would
otherwise let the model invert the CPCB formula and score near-perfectly while
being useless in production.
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
LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
ROLL_WINDOWS = [6, 24, 72]


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


def add_history_features(df, columns):
    """Lags, trailing stats and deltas, computed per city."""
    g = df.groupby("City", sort=False)
    new = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = g[col]
        for lag in LAGS:
            new[f"{col}_Lag_{lag}h"] = s.shift(lag)
        # shift(1) first so a window never includes the current hour.
        prev = s.shift(1)
        for w in ROLL_WINDOWS:
            grouped_prev = prev.groupby(df["City"], sort=False)
            new[f"{col}_RollMean_{w}h"] = grouped_prev.transform(
                lambda x, w=w: x.rolling(w, min_periods=max(2, w // 3)).mean())
            new[f"{col}_RollStd_{w}h"] = grouped_prev.transform(
                lambda x, w=w: x.rolling(w, min_periods=max(2, w // 3)).std())
        new[f"{col}_Delta_1h"] = new[f"{col}_Lag_1h"] - new[f"{col}_Lag_2h"]
        new[f"{col}_Delta_24h"] = new[f"{col}_Lag_1h"] - new[f"{col}_Lag_24h"]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def build(raw_path, out_path, horizon, min_history):
    df = pd.read_csv(raw_path, parse_dates=["datetime"])
    df = df.sort_values(["City", "datetime"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows, {df['City'].nunique()} cities")

    df = compute_aqi(df, group_col="City")
    print(f"AQI computed for {df['AQI'].notna().sum():,} rows "
          f"(mean {df['AQI'].mean():.1f}, max {df['AQI'].max():.1f})")

    df = add_temporal_features(df)
    df = add_history_features(df, POLLUTANTS + WEATHER + ["AQI"])

    # Target: AQI `horizon` hours in the future for this city.
    df["target_AQI"] = df.groupby("City", sort=False)["AQI"].shift(-horizon)

    # Weather forecasts are legitimately available ahead of time, so the
    # same-timestamp weather columns stay. Same-timestamp pollutant
    # concentrations and AQI/sub-index columns are dropped: they leak.
    leaky = POLLUTANTS + ["AQI", "AQI_Category", "Dominant_Pollutant"]
    leaky += [c for c in df.columns if c.startswith("SubIndex_")]
    feature_cols = [c for c in df.columns
                    if c not in leaky + ["datetime", "target_AQI", "State", "City"]]

    df = df.dropna(subset=["target_AQI"])
    # Require enough lag history for the row to be meaningful.
    df = df[df[f"AQI_Lag_{min_history}h"].notna()]

    # Open-Meteo serves NH3 only over Europe, so its columns are entirely empty
    # for India. All-NaN/constant columns carry no signal and break the
    # histogram binner in HistGradientBoostingRegressor.
    degenerate = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if degenerate:
        print(f"Dropping {len(degenerate)} empty/constant columns "
              f"(e.g. {', '.join(degenerate[:3])})")
        feature_cols = [c for c in feature_cols if c not in degenerate]

    keep = ["City", "State", "datetime"] + feature_cols + ["target_AQI"]
    out = df[keep]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nFeatures: {len(feature_cols)}  Rows: {len(out):,}")
    print(f"Target mean {out['target_AQI'].mean():.1f}, std {out['target_AQI'].std():.1f}")
    print(f"Saved -> {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/aqi/raw.csv")
    p.add_argument("--out", default="data/aqi/features.csv")
    p.add_argument("--params", default="params.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.params))["aqi"]
    build(args.raw, args.out, cfg["horizon_hours"], cfg["min_history_hours"])


if __name__ == "__main__":
    main()
