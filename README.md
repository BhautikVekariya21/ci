# mlops / ci

Two pipelines live in this repo:

- **AQI forecasting** (`src/aqi/`) — predicts India's CPCB Air Quality Index 24 hours ahead.
- **Tweet sentiment** (`src/`, `dvc.yaml`) — the original text classification pipeline.

## AQI forecasting

### Why this was rewritten

The earlier `aqi.ipynb` collected data from WAQI's `iaqi` field and treated those
values as PM2.5 concentrations in µg/m³. They are not — they are already AQI
sub-indices. WAQI returns `aqi: 95` and `iaqi.pm25.v: 95` for Delhi: the same
number. Any model trained that way has its own target sitting in the feature
matrix, so it scores well offline and diverges from real AQI in production.

The rewrite fixes three things:

1. **Real concentrations.** Open-Meteo's air-quality API returns genuine µg/m³
   values, so the AQI is computed rather than copied.
2. **Correct AQI formula.** `src/aqi/cpcb.py` implements the CPCB National AQI:
   per-pollutant sub-indices over their mandated averaging windows (24h for
   particulates, 8h for CO/O₃), AQI as the worst sub-index, requiring ≥3
   pollutants with at least one particulate.
3. **No target leakage.** Every pollutant feature is a lag or trailing window.
   Same-timestamp concentrations are dropped, since they would let the model
   invert the CPCB formula instead of forecasting.

### Usage

```bash
python -m src.aqi.fetch_data --start 2024-06-01 --end 2026-08-03
python -m src.aqi.build_features
python -m src.aqi.train
```

Or via DVC: `dvc repro aqi_train`

### Serving

```bash
docker build -t aqi-model .
docker run -p 5000:5000 aqi-model
```

`GET /health` and `POST /predict` with a JSON object of feature values.

### Configuration

Model and horizon settings live under the `aqi:` key in `params.yaml`.
Copy `.env.example` to `.env` for API tokens; `.env` is gitignored.
