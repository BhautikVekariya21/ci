"""Fetch hourly pollutant concentrations and weather for Indian cities.

Uses Open-Meteo's air-quality API, which reports genuine ug/m3 concentrations.
The prior notebook used WAQI's `iaqi` values, which are already AQI sub-indices
(WAQI Delhi returns aqi=95 and iaqi.pm25.v=95 -- the same number), so training
on them made the target a near-copy of a feature.
"""
import argparse
import os
import time

import pandas as pd
import requests

from .locations import INDIA_CITIES

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

POLLUTANT_VARS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2", "carbon_monoxide": "CO", "ozone": "O3", "ammonia": "NH3",
}
WEATHER_VARS = {
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "surface_pressure": "Pressure", "wind_speed_10m": "Wind_Speed",
    "wind_direction_10m": "Wind_Direction", "precipitation": "Precipitation",
    "cloud_cover": "Cloud_Cover", "boundary_layer_height": "Boundary_Layer_Height",
}


def _get(session, url, params, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"    failed after {retries} attempts: {type(e).__name__}")
                return None
            time.sleep(5 * (attempt + 1))
    return None


def _hourly_frame(payload, varmap):
    if not payload or "hourly" not in payload:
        return pd.DataFrame()
    hourly = payload["hourly"]
    data = {"datetime": pd.to_datetime(hourly["time"])}
    for api_name, out_name in varmap.items():
        if api_name in hourly:
            data[out_name] = hourly[api_name]
    return pd.DataFrame(data)


def _date_chunks(start_date, end_date, months=6):
    """Split a range into windows. Open-Meteo times out on multi-year spans."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    while start <= end:
        stop = min(start + pd.DateOffset(months=months) - pd.Timedelta(days=1), end)
        chunks.append((start.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        start = stop + pd.Timedelta(days=1)
    return chunks


def fetch_city(session, city, info, start_date, end_date):
    """Fetch pollutants + weather for one city, merged on the hour."""
    parts = []
    for chunk_start, chunk_end in _date_chunks(start_date, end_date):
        common = {"latitude": info["lat"], "longitude": info["lon"],
                  "start_date": chunk_start, "end_date": chunk_end,
                  "timezone": "Asia/Kolkata"}

        air = _hourly_frame(_get(session, AIR_QUALITY_URL,
                                 {**common, "hourly": ",".join(POLLUTANT_VARS)}), POLLUTANT_VARS)
        if air.empty:
            continue

        weather_vars = ",".join(v for v in WEATHER_VARS if v != "boundary_layer_height")
        weather = _hourly_frame(_get(session, WEATHER_ARCHIVE_URL,
                                     {**common, "hourly": weather_vars}), WEATHER_VARS)

        parts.append(air.merge(weather, on="datetime", how="left") if not weather.empty else air)
        time.sleep(0.5)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True).drop_duplicates("datetime")
    df.insert(0, "City", city)
    df.insert(1, "State", info["state"])
    return df


def fetch_all(start_date, end_date, cities=None):
    cities = cities or list(INDIA_CITIES)
    session = requests.Session()
    frames = []

    for i, city in enumerate(cities, 1):
        info = INDIA_CITIES[city]
        print(f"[{i}/{len(cities)}] {city} ({info['state']})...", end=" ", flush=True)
        df = fetch_city(session, city, info, start_date, end_date)
        if df.empty:
            print("no data")
        else:
            frames.append(df)
            pm = df["PM2.5"].notna().sum()
            print(f"{len(df)} rows, {pm} with PM2.5")
        time.sleep(1.0)

    if not frames:
        raise RuntimeError("No data fetched for any city.")
    return pd.concat(frames, ignore_index=True).sort_values(["City", "datetime"])


def main():
    p = argparse.ArgumentParser(description="Fetch India AQI training data")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default="data/aqi/raw.csv")
    p.add_argument("--cities", nargs="*", default=None)
    args = p.parse_args()

    df = fetch_all(args.start, args.end, args.cities)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nSaved {len(df):,} rows x {df.shape[1]} cols -> {args.out}")
    print(f"Cities: {df['City'].nunique()}  Range: {df['datetime'].min()} .. {df['datetime'].max()}")


if __name__ == "__main__":
    main()
