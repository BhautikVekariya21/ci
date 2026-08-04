"""Compare the model's AQI forecast against currently observed AQI.

Fetches recent hours for a city, computes the CPCB AQI now, builds the same
features the model was trained on, and prints observed-vs-forecast side by side.
"""
import argparse
import pickle

import pandas as pd
import requests

from .build_features import add_history_features, add_temporal_features, POLLUTANTS, WEATHER
from .cpcb import compute_aqi
from .fetch_data import POLLUTANT_VARS, WEATHER_VARS, AIR_QUALITY_URL, WEATHER_FORECAST_URL
from .locations import INDIA_CITIES

# Enough history to fill the 72h lags and the 24h rolling windows.
PAST_DAYS = 10


def fetch_recent(city):
    info = INDIA_CITIES[city]
    common = {"latitude": info["lat"], "longitude": info["lon"],
              "timezone": "Asia/Kolkata", "past_days": PAST_DAYS, "forecast_days": 2}
    s = requests.Session()

    air = s.get(AIR_QUALITY_URL, params={**common, "hourly": ",".join(POLLUTANT_VARS)},
                timeout=120).json()["hourly"]
    wx = s.get(WEATHER_FORECAST_URL,
               params={**common, "hourly": ",".join(v for v in WEATHER_VARS
                                                    if v != "boundary_layer_height")},
               timeout=120).json()["hourly"]

    a = pd.DataFrame({"datetime": pd.to_datetime(air["time"]),
                      **{o: air[k] for k, o in POLLUTANT_VARS.items() if k in air}})
    w = pd.DataFrame({"datetime": pd.to_datetime(wx["time"]),
                      **{o: wx[k] for k, o in WEATHER_VARS.items() if k in wx}})

    df = a.merge(w, on="datetime", how="left")
    df.insert(0, "City", city)
    df.insert(1, "State", info["state"])
    return df.sort_values("datetime").reset_index(drop=True)


def compare(city, model_path="models/aqi_model.pkl"):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    horizon = bundle["horizon_hours"]

    raw = fetch_recent(city)
    df = compute_aqi(raw, group_col="City")
    df = add_temporal_features(df)
    df = add_history_features(df, POLLUTANTS + WEATHER + ["AQI"])
    df["City_Code"] = df["City"].map(bundle["city_codes"])

    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing {len(missing)} features, e.g. {missing[:5]}")

    ready = df[df[feature_cols].notna().all(axis=1)].copy()
    if ready.empty:
        raise SystemExit("not enough history to build a complete feature row")

    ready["forecast"] = bundle["model"].predict(ready[feature_cols].values)
    ready["forecast_for"] = ready["datetime"] + pd.Timedelta(hours=horizon)

    # Where the forecast target time is itself observed, we can score it.
    observed = df.set_index("datetime")["AQI"]
    ready["actual_at_target"] = ready["forecast_for"].map(observed)

    latest = ready.iloc[-1]
    now_aqi = df.loc[df["AQI"].notna(), "AQI"].iloc[-1]
    now_time = df.loc[df["AQI"].notna(), "datetime"].iloc[-1]

    print(f"\n=== {city} ===")
    print(f"Current AQI  {now_aqi:6.1f}   (observed at {now_time})")
    print(f"Forecast     {latest['forecast']:6.1f}   (for {latest['forecast_for']}, +{horizon}h)")
    print(f"Change       {latest['forecast'] - now_aqi:+6.1f}")

    scored = ready.dropna(subset=["actual_at_target"])
    if not scored.empty:
        err = (scored["forecast"] - scored["actual_at_target"]).abs()
        print(f"\nBacktest on the last {len(scored)} forecastable hours:")
        print(f"  MAE {err.mean():.1f}   worst {err.max():.1f}")

    print(f"\nLast 12 forecastable hours:")
    tail = ready.tail(12)
    for _, r in tail.iterrows():
        actual = r["actual_at_target"]
        actual_s = f"{actual:6.1f}" if pd.notna(actual) else "     -"
        print(f"  made {r['datetime']:%m-%d %H:%M} -> for {r['forecast_for']:%m-%d %H:%M}  "
              f"forecast {r['forecast']:6.1f}   actual {actual_s}")


def main():
    p = argparse.ArgumentParser(description="Compare forecast AQI vs current AQI")
    p.add_argument("--city", default="Delhi", choices=sorted(INDIA_CITIES))
    p.add_argument("--model", default="models/aqi_model.pkl")
    args = p.parse_args()
    compare(args.city, args.model)


if __name__ == "__main__":
    main()
