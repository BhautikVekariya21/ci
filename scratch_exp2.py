"""Rolling-origin CV over improvement candidates for the AQI forecaster.

The seasonal diagnostic showed error is multiplicative (~10% relative in every
season and band) while the model optimises squared error on a raw target with
skew 1.46. That mismatch is the main lever, so the variants here isolate it:
log target, absolute-error objective, and richer leak-free history.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, ".")
from src.aqi.cpcb import compute_aqi
from src.aqi.build_features import add_temporal_features, POLLUTANTS, WEATHER

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "PostMonsoon", 11: "PostMonsoon"}
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "PostMonsoon"]

# Span the pollution range rather than only the worst offenders.
SUBSET = ["Delhi", "Faridabad", "Ludhiana", "Jaipur", "Lucknow", "Patna",
          "Kolkata", "Raipur", "Mumbai", "Hyderabad", "Chennai", "Bengaluru"]

HORIZON = 24
MIN_HISTORY = 24

LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
EXTRA_LAGS = [96, 120, 168]
ROLL_WINDOWS = [6, 24, 72]
LONG_ROLLS = [168, 336]

# Weather is forecastable, so its value AT THE TARGET HOUR is legitimately known
# at prediction time. Pollutant concentrations are not.
LEAD_WEATHER = ["Wind_Speed", "Precipitation", "Temperature", "Humidity",
                "Cloud_Cover", "Pressure"]

SUB_COLS = [f"SubIndex_{p}" for p in POLLUTANTS]


def history_features(g, columns, lags, extra_lags=(), rolling=True):
    all_lags = list(lags) + list(extra_lags)
    new = {}
    for col in columns:
        if col not in g.columns:
            continue
        s = g[col]
        for lag in all_lags:
            new[f"{col}_Lag_{lag}h"] = s.shift(lag)
        if rolling:
            # shift(1) first so a window never includes the current hour.
            prev = s.shift(1)
            for w in ROLL_WINDOWS:
                roll = prev.rolling(w, min_periods=max(2, w // 3))
                new[f"{col}_RollMean_{w}h"] = roll.mean()
                new[f"{col}_RollStd_{w}h"] = roll.std()
        for a, b in ((1, 2), (1, 24)):
            if a in all_lags and b in all_lags:
                new[f"{col}_Delta_{a}_{b}h"] = new[f"{col}_Lag_{a}h"] - new[f"{col}_Lag_{b}h"]
    return pd.concat([g, pd.DataFrame(new, index=g.index)], axis=1)


def build_rich(raw):
    """One frame carrying every candidate feature; variants select subsets."""
    parts = []
    for city, g in raw.groupby("City", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        g = compute_aqi(g, group_col="City")
        g = add_temporal_features(g)

        # Lagged CPCB sub-indices say WHICH pollutant drives AQI and where it is
        # heading. They are functions of past concentrations only, so no leak.
        g = history_features(g, POLLUTANTS + WEATHER + ["AQI"], LAGS, EXTRA_LAGS)
        sub_present = [c for c in SUB_COLS if c in g.columns]
        g = history_features(g, sub_present, [1, 2, 24, 48], (), rolling=False)

        prev = g["AQI"].shift(1)
        for w in LONG_ROLLS:
            r = prev.rolling(w, min_periods=w // 3)
            g[f"AQI_RollMean_{w}h"] = r.mean()
            g[f"AQI_RollStd_{w}h"] = r.std()
        # How unusual is right now versus this city's recent normal?
        g["AQI_Ratio_24h_168h"] = (g["AQI_RollMean_24h"] /
                                   g["AQI_RollMean_168h"].replace(0, np.nan))

        for c in LEAD_WEATHER:
            if c in g.columns:
                lead = g[c].shift(-HORIZON)
                g[f"{c}_Lead_{HORIZON}h"] = lead
                g[f"{c}_LeadDelta_{HORIZON}h"] = lead - g[c]
        # Ventilation over the forecast window drives dispersion.
        if "Wind_Speed" in g.columns:
            fwd = g["Wind_Speed"].shift(-HORIZON).rolling(HORIZON, min_periods=6).mean()
            g[f"Wind_Speed_LeadMean_{HORIZON}h"] = fwd
        if "Precipitation" in g.columns:
            g[f"Precipitation_LeadSum_{HORIZON}h"] = (
                g["Precipitation"].shift(-HORIZON).rolling(HORIZON, min_periods=6).sum())

        g["target_AQI"] = g["AQI"].shift(-HORIZON)
        g = g.dropna(subset=["target_AQI"])
        g = g[g[f"AQI_Lag_{MIN_HISTORY}h"].notna()]

        drop = set(POLLUTANTS + SUB_COLS + ["AQI", "AQI_Category", "Dominant_Pollutant",
                                            "State", "datetime", "City", "target_AQI"])
        keep = [c for c in g.columns if c not in drop]
        part = g[["City", "datetime"] + keep + ["target_AQI"]].copy()
        for c in part.columns:
            if part[c].dtype == "float64":
                part[c] = part[c].astype("float32")
        parts.append(part)
        print(f"  {city:<14}{len(part):>8,} rows", flush=True)

    out = pd.concat(parts, ignore_index=True)
    dead = [c for c in out.columns
            if c not in ("City", "datetime") and out[c].nunique(dropna=True) <= 1]
    if dead:
        print(f"  dropping {len(dead)} empty/constant cols")
        out = out.drop(columns=dead)
    return out.sort_values("datetime").reset_index(drop=True)


def select_cols(df, sub_lags, lead, long_roll, extra_lags):
    """Column subset for a variant, so every variant reads the same built frame."""
    cols = []
    for c in df.columns:
        if c in ("City", "datetime", "target_AQI"):
            continue
        if c.startswith("SubIndex_") and not sub_lags:
            continue
        if ("_Lead_" in c or "_LeadDelta_" in c or "_LeadMean_" in c
                or "_LeadSum_" in c) and not lead:
            continue
        if (any(f"_{w}h" in c for w in LONG_ROLLS) or c == "AQI_Ratio_24h_168h") and not long_roll:
            continue
        if any(f"_Lag_{L}h" in c for L in EXTRA_LAGS) and not extra_lags:
            continue
        cols.append(c)
    return cols


def add_climatology(train, test, cols):
    """City x month x hour mean AQI, fitted on TRAIN only."""
    t = train[["City", "datetime", "target_AQI"]].copy()
    t["m"] = t["datetime"].dt.month
    t["h"] = t["datetime"].dt.hour
    clim = t.groupby(["City", "m", "h"])["target_AQI"].mean().rename("Clim_AQI")
    city_clim = t.groupby("City")["target_AQI"].mean().rename("Clim_City")

    out = []
    for d in (train, test):
        d = d.copy()
        d["m"] = d["datetime"].dt.month
        d["h"] = d["datetime"].dt.hour
        d = d.merge(clim, on=["City", "m", "h"], how="left")
        d = d.merge(city_clim, on="City", how="left")
        d["Clim_AQI"] = d["Clim_AQI"].fillna(d["Clim_City"])
        # Where does the model sit relative to this city's seasonal normal?
        d["Clim_Ratio"] = d["AQI_Lag_1h"] / d["Clim_AQI"].replace(0, np.nan)
        out.append(d.drop(columns=["m", "h"]))
    return out[0], out[1], cols + ["Clim_AQI", "Clim_City", "Clim_Ratio"]


def make_model(kind, log_target, model_kw):
    kw = dict(model_kw or {})
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        base = dict(n_estimators=1200, learning_rate=0.05, num_leaves=255,
                    min_child_samples=40, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.7, reg_lambda=1.0, max_bin=255,
                    n_jobs=-1, random_state=41, verbose=-1)
        base.update(kw)
        return LGBMRegressor(**base)
    from sklearn.ensemble import HistGradientBoostingRegressor
    base = dict(max_iter=400, learning_rate=0.06, max_depth=9, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                random_state=41)
    base.update(kw)
    return HistGradientBoostingRegressor(**base)


def run_cv(df, label, sub_lags=False, lead=False, long_roll=False, extra_lags=False,
           climatology=False, log_target=False, kind="hgb", model_kw=None, n_folds=3):
    base_cols = select_cols(df, sub_lags, lead, long_roll, extra_lags)

    city_codes = {c: i for i, c in enumerate(sorted(df["City"].unique()))}
    df = df.copy()
    df["City_Code"] = df["City"].map(city_codes)
    base_cols = base_cols + ["City_Code"]

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
            y_train = np.log1p(np.clip(y_train, 0, None))

        m = make_model(kind, log_target, model_kw)
        m.fit(train[cols].values, y_train)
        pred = m.predict(test[cols].values)
        if log_target:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, 500)

        rows.append(pd.DataFrame({
            "err": np.abs(pred - y_test),
            "season": test["datetime"].dt.month.map(SEASONS).values,
            "actual": y_test,
        }))

    r = pd.concat(rows, ignore_index=True)
    per_season = r.groupby("season")["err"].mean()
    overall = r["err"].mean()
    within25 = (r["err"] <= 25).mean()
    print(f"{label:<34}{overall:8.2f}{within25 * 100:8.1f}  " +
          "  ".join(f"{s[:4]}:{per_season.get(s, float('nan')):5.1f}" for s in SEASON_ORDER),
          flush=True)
    return overall, per_season


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated variant letters")
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else None

    def maybe(letter, *a, **kw):
        if only is None or letter in only:
            run_cv(*a, **kw)

    t0 = time.time()
    raw = pd.read_csv("data/aqi/raw.csv", parse_dates=["datetime"])
    if not args.full:
        raw = raw[raw["City"].isin(SUBSET)]
    raw = raw.sort_values(["City", "datetime"]).reset_index(drop=True)
    print(f"raw: {len(raw):,} rows, {raw['City'].nunique()} cities ({time.time()-t0:.0f}s)\n")

    df = build_rich(raw)
    del raw
    print(f"built {df.shape}  ({time.time()-t0:.0f}s)\n")

    print(f"{'variant':<34}{'MAE':>8}{'w/in25%':>8}  per-season MAE")
    print("-" * 92)

    maybe("A", df, "A baseline (current prod)")
    maybe("B", df, "B + log target", log_target=True)
    maybe("C", df, "C + MAE objective", model_kw={"loss": "absolute_error"})
    maybe("D", df, "D + log + MAE obj", log_target=True,
          model_kw={"loss": "absolute_error"})
    maybe("E", df, "E + log + sublags", log_target=True, sub_lags=True)
    maybe("F", df, "F + log + sublags + lead", log_target=True, sub_lags=True, lead=True)
    maybe("G", df, "G F + longroll + xlags", log_target=True, sub_lags=True,
          lead=True, long_roll=True, extra_lags=True)
    maybe("H", df, "H G + climatology", log_target=True, sub_lags=True, lead=True,
          long_roll=True, extra_lags=True, climatology=True)
    maybe("I", df, "I H as lgbm", log_target=True, sub_lags=True, lead=True,
          long_roll=True, extra_lags=True, climatology=True, kind="lgbm")
    maybe("J", df, "J lgbm + MAE obj", log_target=True, sub_lags=True, lead=True,
          long_roll=True, extra_lags=True, climatology=True, kind="lgbm",
          model_kw={"objective": "l1"})

    print(f"\ntotal {time.time()-t0:.0f}s")
