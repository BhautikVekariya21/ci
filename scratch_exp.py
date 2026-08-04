"""Experiment harness: rolling-origin CV over feature/target variants.

Runs on a representative subset of cities for speed. The winning config is
retrained on all 36 cities afterwards.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, ".")
from src.aqi.cpcb import compute_aqi
from src.aqi.build_features import add_temporal_features, POLLUTANTS, WEATHER, LAGS, ROLL_WINDOWS

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "PostMonsoon", 11: "PostMonsoon"}

# Span the pollution range rather than only the worst offenders.
SUBSET = ["Delhi", "Faridabad", "Ludhiana", "Jaipur", "Lucknow", "Patna",
          "Kolkata", "Raipur", "Mumbai", "Hyderabad", "Chennai", "Bengaluru"]

HORIZON = 24
MIN_HISTORY = 24

# Weather is forecastable, so its value AT THE TARGET HOUR is legitimately
# known at prediction time. Pollutant concentrations are not.
LEAD_WEATHER = ["Wind_Speed", "Precipitation", "Temperature", "Humidity",
                "Cloud_Cover", "Pressure"]


def history_features(g, columns, extra_lags=()):
    lags = list(LAGS) + list(extra_lags)
    new = {}
    for col in columns:
        if col not in g.columns:
            continue
        s = g[col]
        for lag in lags:
            new[f"{col}_Lag_{lag}h"] = s.shift(lag)
        prev = s.shift(1)
        for w in ROLL_WINDOWS:
            roll = prev.rolling(w, min_periods=max(2, w // 3))
            new[f"{col}_RollMean_{w}h"] = roll.mean()
            new[f"{col}_RollStd_{w}h"] = roll.std()
        new[f"{col}_Delta_1h"] = new[f"{col}_Lag_1h"] - new[f"{col}_Lag_2h"]
        new[f"{col}_Delta_24h"] = new[f"{col}_Lag_1h"] - new[f"{col}_Lag_24h"]
    return pd.concat([g, pd.DataFrame(new, index=g.index)], axis=1)


def build(raw, weather_lead=False, extra_lags=(), long_roll=False):
    parts = []
    hist_cols = POLLUTANTS + WEATHER + ["AQI"]
    for city, g in raw.groupby("City", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        g = compute_aqi(g, group_col="City")
        g = add_temporal_features(g)
        g = history_features(g, hist_cols, extra_lags)

        if long_roll:
            prev = g["AQI"].shift(1)
            for w in (168, 336):
                r = prev.rolling(w, min_periods=w // 3)
                g[f"AQI_RollMean_{w}h"] = r.mean()
                g[f"AQI_RollStd_{w}h"] = r.std()

        if weather_lead:
            # Weather at the target hour -- available from a forecast in production.
            for c in LEAD_WEATHER:
                if c in g.columns:
                    g[f"{c}_Lead_{HORIZON}h"] = g[c].shift(-HORIZON)
                    g[f"{c}_LeadDelta_{HORIZON}h"] = g[f"{c}_Lead_{HORIZON}h"] - g[c]

        g["target_AQI"] = g["AQI"].shift(-HORIZON)
        g = g.dropna(subset=["target_AQI"])
        g = g[g[f"AQI_Lag_{MIN_HISTORY}h"].notna()]

        drop = set(POLLUTANTS + ["AQI", "AQI_Category", "Dominant_Pollutant",
                                 "State", "datetime", "City", "target_AQI"])
        keep = [c for c in g.columns if c not in drop and not c.startswith("SubIndex_")]
        part = g[["City", "datetime"] + keep + ["target_AQI"]].copy()
        for c in part.columns:
            if part[c].dtype == "float64":
                part[c] = part[c].astype("float32")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    # Open-Meteo returns no NH3 for India, so every NH3 feature is all-NaN.
    dead = [c for c in out.columns if c not in ("City", "datetime") and out[c].isna().all()]
    if dead:
        out = out.drop(columns=dead)
    return out.sort_values("datetime").reset_index(drop=True)


def add_climatology(train, test, feature_cols):
    """City x month x hour mean AQI, fitted on TRAIN only."""
    t = train.copy()
    t["month"] = t["datetime"].dt.month
    t["hour"] = t["datetime"].dt.hour
    clim = t.groupby(["City", "month", "hour"])["target_AQI"].mean().rename("Clim_AQI")
    city_clim = t.groupby("City")["target_AQI"].mean().rename("Clim_City")

    out = []
    for d in (train, test):
        d = d.copy()
        d["month"] = d["datetime"].dt.month
        d["hour"] = d["datetime"].dt.hour
        d = d.merge(clim, on=["City", "month", "hour"], how="left")
        d = d.merge(city_clim, on="City", how="left")
        d["Clim_AQI"] = d["Clim_AQI"].fillna(d["Clim_City"])
        d = d.drop(columns=["month", "hour"])
        out.append(d)
    return out[0], out[1], feature_cols + ["Clim_AQI", "Clim_City"]


def run_cv(df, label, log_target=False, climatology=False, n_folds=3, model_kw=None):
    df = df.sort_values("datetime").reset_index(drop=True)
    base_cols = [c for c in df.columns if c not in ("City", "datetime", "target_AQI")]

    city_codes = {c: i for i, c in enumerate(sorted(df["City"].unique()))}
    df["City_Code"] = df["City"].map(city_codes)
    base_cols.append("City_Code")

    times = df["datetime"]
    span_start, span_end = times.min(), times.max()
    total = (span_end - span_start).total_seconds()

    rows = []
    for k in range(n_folds):
        # Expanding window: train on everything before the fold, test the next slice.
        train_end = span_start + pd.Timedelta(seconds=total * (0.55 + 0.15 * k))
        test_end = span_start + pd.Timedelta(seconds=total * (0.55 + 0.15 * (k + 1)))
        train = df[df["datetime"] <= train_end]
        test = df[(df["datetime"] > train_end) & (df["datetime"] <= test_end)]
        if len(test) < 1000:
            continue

        cols = base_cols
        if climatology:
            train, test, cols = add_climatology(train, test, base_cols)

        y_train = train["target_AQI"].values
        y_test = test["target_AQI"].values
        if log_target:
            y_train = np.log1p(y_train)

        kw = dict(max_iter=400, learning_rate=0.06, max_depth=9, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  random_state=41)
        kw.update(model_kw or {})
        m = HistGradientBoostingRegressor(**kw)
        m.fit(train[cols].values, y_train)

        pred = m.predict(test[cols].values)
        if log_target:
            pred = np.expm1(pred)

        e = np.abs(pred - y_test)
        season = test["datetime"].dt.month.map(SEASONS).values
        rows.append(pd.DataFrame({"fold": k, "err": e, "season": season,
                                  "actual": y_test}))

    r = pd.concat(rows, ignore_index=True)
    per_season = r.groupby("season")["err"].mean()
    overall = r["err"].mean()
    within25 = (r["err"] <= 25).mean()
    print(f"{label:<38}{overall:8.2f}{within25 * 100:8.1f}  " +
          "  ".join(f"{s[:4]}:{per_season.get(s, float('nan')):5.1f}"
                    for s in ["Winter", "Summer", "Monsoon", "PostMonsoon"]))
    return overall, per_season


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="use all cities")
    args = p.parse_args()

    t0 = time.time()
    raw = pd.read_csv("data/aqi/raw.csv", parse_dates=["datetime"])
    if not args.full:
        raw = raw[raw["City"].isin(SUBSET)]
    raw = raw.sort_values(["City", "datetime"]).reset_index(drop=True)
    print(f"raw: {len(raw):,} rows, {raw['City'].nunique()} cities  ({time.time() - t0:.0f}s)\n")

    print(f"{'variant':<38}{'MAE':>8}{'w/in25%':>8}  per-season MAE")
    print("-" * 96)

    base = build(raw)
    run_cv(base, "A baseline")
    run_cv(base, "B + log target", log_target=True)
    run_cv(base, "C + climatology", climatology=True)
    run_cv(base, "D + log + climatology", log_target=True, climatology=True)
    del base

    wl = build(raw, weather_lead=True)
    run_cv(wl, "E + weather-lead")
    run_cv(wl, "F + weather-lead + log", log_target=True)
    run_cv(wl, "G + weather-lead + log + clim", log_target=True, climatology=True)
    del wl

    full = build(raw, weather_lead=True, extra_lags=(120, 168), long_roll=True)
    run_cv(full, "H + all feats", log_target=True, climatology=True)
    print(f"\ntotal {time.time() - t0:.0f}s")
