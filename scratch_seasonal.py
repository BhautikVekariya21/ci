"""Where is the current model weakest? Per-season and per-band breakdown."""
import pickle

import numpy as np
import pandas as pd

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "PostMonsoon", 11: "PostMonsoon"}

bundle = pickle.load(open("models/aqi_model.pkl", "rb"))
df = pd.read_parquet("data/aqi/features.parquet").sort_values("datetime")

cutoff = df["datetime"].quantile(0.80)
test = df[df["datetime"] > cutoff].copy()
test["City_Code"] = test["City"].map(bundle["city_codes"])
pred = bundle["model"].predict(test[bundle["feature_cols"]].values)
if bundle.get("log_target"):
    pred = np.expm1(pred)
test["pred"] = np.clip(pred, 0, 500)
test["abs_err"] = (test["pred"] - test["target_AQI"]).abs()
test["season"] = test["datetime"].dt.month.map(SEASONS)

print(f"Test: {test['datetime'].min()} .. {test['datetime'].max()}  {len(test):,} rows")
print(f"Overall MAE {test['abs_err'].mean():.2f}\n")

print("=== By season ===")
s = test.groupby("season").agg(
    n=("abs_err", "size"), mean_aqi=("target_AQI", "mean"),
    mae=("abs_err", "mean"),
)
s["mae_pct"] = s["mae"] / s["mean_aqi"] * 100
# Persistence baseline per season for context.
test["persist_err"] = (test["AQI_Lag_1h"] - test["target_AQI"]).abs()
s["persist_mae"] = test.groupby("season")["persist_err"].mean()
s["lift_vs_persist"] = (1 - s["mae"] / s["persist_mae"]) * 100
print(s.round(2).to_string())

print("\n=== By month ===")
test["month"] = test["datetime"].dt.month
m = test.groupby("month").agg(n=("abs_err", "size"), mean_aqi=("target_AQI", "mean"),
                              mae=("abs_err", "mean"))
m["mae_pct"] = m["mae"] / m["mean_aqi"] * 100
print(m.round(2).to_string())

print("\n=== By AQI band ===")
test["band"] = pd.cut(test["target_AQI"], [0, 50, 100, 200, 300, 400, 1000],
                      labels=["Good", "Satisf", "Moder", "Poor", "VPoor", "Severe"])
b = test.groupby("band").agg(n=("abs_err", "size"), mean_aqi=("target_AQI", "mean"),
                             mae=("abs_err", "mean"))
b["mae_pct"] = b["mae"] / b["mean_aqi"] * 100
print(b.round(2).to_string())

print("\n=== Error skew: is the error multiplicative? ===")
test["rel_err"] = test["abs_err"] / test["target_AQI"].clip(lower=1)
print(f"abs err  mean {test['abs_err'].mean():.2f}  median {test['abs_err'].median():.2f}")
print(f"rel err  mean {test['rel_err'].mean():.3f}  median {test['rel_err'].median():.3f}")
print(f"target skew: {test['target_AQI'].skew():.2f}   log-target skew: {np.log1p(test['target_AQI']).skew():.2f}")

print("\n=== Worst season x city combos ===")
sc = test.groupby(["season", "City"])["abs_err"].mean().sort_values(ascending=False)
print(sc.head(12).round(1).to_string())
