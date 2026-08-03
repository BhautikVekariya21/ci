"""Train the AQI forecasting model.

Split is chronological, not random: a random split would put a city's 14:00 and
16:00 readings on opposite sides of the boundary, and neighbouring hours are so
correlated that the test score would be meaningless.
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .cpcb import AQI_CATEGORIES

MODELS = {
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
}


def chronological_split(df, test_fraction):
    cutoff = df["datetime"].quantile(1 - test_fraction)
    train = df[df["datetime"] <= cutoff].copy()
    test = df[df["datetime"] > cutoff].copy()
    print(f"Split at {cutoff}: {len(train):,} train / {len(test):,} test")
    return train, test


def aqi_category(values):
    bounds = [b for b, _ in AQI_CATEGORIES]
    labels = [lab for _, lab in AQI_CATEGORIES]
    return pd.cut(values, bins=[-np.inf] + bounds, labels=labels)


def evaluate(y_true, y_pred, label):
    metrics = {
        f"{label}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{label}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{label}_r2": float(r2_score(y_true, y_pred)),
        f"{label}_category_accuracy": float(
            (aqi_category(pd.Series(y_true)).values ==
             aqi_category(pd.Series(y_pred)).values).mean()),
    }
    within = {f"{label}_within_{t}": float((np.abs(y_true - y_pred) <= t).mean())
              for t in (10, 25, 50)}
    return {**metrics, **within}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/aqi/features.csv")
    p.add_argument("--model-out", default="models/aqi_model.pkl")
    p.add_argument("--metrics-out", default="evaluation/aqi_metrics.json")
    p.add_argument("--params", default="params.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.params))["aqi"]

    df = pd.read_csv(args.features, parse_dates=["datetime"]).sort_values("datetime")
    train, test = chronological_split(df, cfg["test_fraction"])

    drop = ["City", "State", "datetime", "target_AQI"]
    feature_cols = [c for c in df.columns if c not in drop]

    # City carries real signal (local emission baseline), so keep it as an
    # explicit categorical code rather than dropping it.
    city_codes = {c: i for i, c in enumerate(sorted(df["City"].unique()))}
    for split in (train, test):
        split["City_Code"] = split["City"].map(city_codes)
    feature_cols.append("City_Code")

    X_train, y_train = train[feature_cols].values, train["target_AQI"].values
    X_test, y_test = test[feature_cols].values, test["target_AQI"].values

    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("type")
    print(f"\nTraining {name} on {len(feature_cols)} features...")
    model = MODELS[name](**model_cfg)
    model.fit(X_train, y_train)

    metrics = {**evaluate(y_train, model.predict(X_train), "train"),
               **evaluate(y_test, model.predict(X_test), "test")}

    # Persistence baseline: predict AQI stays at its last observed value.
    if "AQI_Lag_1h" in feature_cols:
        baseline = test["AQI_Lag_1h"].values
        metrics.update(evaluate(y_test, baseline, "baseline_persistence"))

    metrics.update({
        "horizon_hours": cfg["horizon_hours"],
        "n_features": len(feature_cols),
        "n_train": len(train),
        "n_test": len(test),
        "train_end": str(train["datetime"].max()),
        "test_end": str(test["datetime"].max()),
    })

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols,
                     "city_codes": city_codes,
                     "horizon_hours": cfg["horizon_hours"]}, f)

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'':<28}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'Cat.Acc':>9}")
    for label in ("train", "test", "baseline_persistence"):
        if f"{label}_mae" in metrics:
            print(f"{label:<28}{metrics[f'{label}_mae']:>8.2f}"
                  f"{metrics[f'{label}_rmse']:>8.2f}{metrics[f'{label}_r2']:>8.3f}"
                  f"{metrics[f'{label}_category_accuracy']:>9.3f}")
    print(f"\nTest within +/-25 AQI: {metrics['test_within_25']:.1%}")
    print(f"Saved model -> {args.model_out}")
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
