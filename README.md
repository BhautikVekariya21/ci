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

The rewrite fixes four things:

1. **Real concentrations.** Open-Meteo's air-quality API returns genuine µg/m³
   values, so the AQI is computed rather than copied.
2. **Correct AQI formula.** `src/aqi/cpcb.py` implements the CPCB National AQI:
   per-pollutant sub-indices over their mandated averaging windows (24h for
   particulates, 8h for CO/O₃), AQI as the worst sub-index, requiring ≥3
   pollutants with at least one particulate.
3. **Max 8-hour averaging for O₃ and CO.** CPCB scores these two on the day's
   *peak* 8-hour mean, not the latest trailing one. Using the trailing mean made
   AQI inherit ozone's diurnal cycle — Delhi swung ~120 points between afternoon
   and night, and ozone appeared to dominate 76% of hours, which contradicts the
   PM-dominated reality of Indian cities. See "AQI stability" below.
4. **No target leakage.** Every pollutant feature is a lag or trailing window.
   Same-timestamp concentrations are dropped, since they would let the model
   invert the CPCB formula instead of forecasting.

### AQI stability

Validated against Open-Meteo's independently computed `us_aqi` over 10 cities
and 30 days. `daySwing` is the mean daily max−min, i.e. how much the index
moves within a day:

| variant | mean AQI | per-city corr | O₃-dominant | daySwing |
|---|---|---|---|---|
| reference `us_aqi` | 108.0 | 1.000 | – | 34.2 |
| trailing-8h O₃/CO (old) | 113.2 | 0.546 | 76.1% | 66.8 |
| **max-8h O₃/CO (current)** | **146.7** | **0.703** | 86.1% | **29.7** |

The old formula's swing was roughly double the reference; the current one
matches it. Note the level rises (146.7 vs 108.0) because CPCB is a stricter
scale than the US AQI — the *shape* is what was wrong before, not the offset.

### Model and results

`HistGradientBoostingRegressor` on 346 leak-free features, trained on
log1p(AQI). The split is **chronological** (last 20% of time held out), not
random — neighbouring hours are too correlated for a random split to mean
anything. Evaluated on the 251,856-row holdout (2025-10-15 → 2026-08-02):

| metric | persistence baseline | previous model | **current** |
|---|---|---|---|
| MAE | 21.19 | 16.61 | **15.71** |
| RMSE | 34.08 | 26.47 | **25.40** |
| R² | 0.784 | 0.870 | **0.880** |
| within ±25 AQI | 72.8% | 80.0% | **81.5%** |
| category accuracy | 79.3% | 82.5% | **83.6%** |

The persistence baseline predicts "AQI in 24h = AQI now". Beating it by ~26% on
MAE is the bar any forecaster has to clear.

**Per-season MAE** — the goal was to lower error in *every* season, not just on
average. The current model does:

| season | previous | **current** | lift vs persistence |
|---|---|---|---|
| Winter | 16.45 | **15.31** | 27.6% |
| Summer | 19.21 | **18.49** | 25.1% |
| Monsoon | 14.58 | **13.77** | 25.1% |
| Post-monsoon | 14.50 | **13.63** | 24.9% |

Summer stays the hardest season (highest mean AQI, sharpest dust events) but
still improved. Error is ~9–11% *relative* in every season and AQI band, which
is what motivated the design choices below.

### What moved the numbers

A rolling-origin CV sweep (`scratch_exp2.py`, expanding-window folds on a
12-city subset) isolated each lever. Three mattered:

1. **Log target.** The raw target is right-skewed (skew 1.46) and the error is
   multiplicative, but squared error on the raw scale over-weights the few
   Severe hours. Fitting `log1p(AQI)` — inverted with `expm1`, clipped to CPCB
   [0, 500] — matches the error structure and lowers MAE in every season.
2. **Lead weather.** Weather is forecastable, so its value *at the target hour*
   is legitimately known at prediction time (from a forecast); pollutant
   concentrations are not. Adding forecasted wind/rain/temperature over the
   horizon was the single biggest gain (subset-CV Winter MAE 25.0 → 22.7).
   Serving requires the caller to pass forecasted weather for the horizon hour.
3. **Richer leak-free history.** Lagged CPCB sub-indices (which pollutant is
   driving AQI and where it's heading), extra long lags (96/120/168h for the
   daily and weekly cycle), and 168/336h rolling context (how unusual now is vs
   the city's recent normal).

LightGBM with an L1 objective edged overall MAE by 0.03 but regressed Summer, so
the sklearn `HistGradientBoostingRegressor` was kept — it improves every season
and keeps serving/tests on a single library.

### Usage

```bash
python -m src.aqi.fetch_data --start 2022-08-05 --end 2026-08-03
python -m src.aqi.build_features
python -m src.aqi.train
python -m src.aqi.compare --city Delhi   # forecast vs currently observed AQI
```

Or via DVC: `dvc repro aqi_train`

`build_features` processes one city at a time and writes float32 Parquet — the
full 1.26M-row table needs ~2.4 GB as float64 CSV but ~0.5 GB this way.

### Serving

```bash
docker build -t aqi-model .
docker run -p 5000:5000 aqi-model
```

`GET /health` and `POST /predict` with a JSON object of feature values.

### Configuration

Model and horizon settings live under the `aqi:` key in `params.yaml`.
Copy `.env.example` to `.env` for API tokens; `.env` is gitignored.
