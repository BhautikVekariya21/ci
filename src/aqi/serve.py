"""Flask API serving AQI forecasts."""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("AQI_MODEL_PATH", "models/aqi_model.pkl")

app = Flask(__name__)
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


@app.route("/health")
def health():
    try:
        get_bundle()
        return jsonify(status="ok"), 200
    except Exception as e:
        return jsonify(status="error", detail=str(e)), 503


@app.route("/predict", methods=["POST"])
def predict():
    bundle = get_bundle()
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify(error="expected a JSON body"), 400

    rows = payload if isinstance(payload, list) else [payload]
    df = pd.DataFrame(rows)

    missing = [c for c in bundle["feature_cols"] if c not in df.columns]
    if missing:
        return jsonify(error="missing features", missing=missing[:20],
                       missing_count=len(missing)), 400

    preds = bundle["model"].predict(df[bundle["feature_cols"]].values)
    return jsonify(horizon_hours=bundle["horizon_hours"],
                   predicted_aqi=[round(float(p), 1) for p in preds])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
