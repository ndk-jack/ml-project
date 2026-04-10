# Earthquake Sequence Forecasting

Binary classification pipeline to estimate whether a significant follow-up earthquake (M≥5.0) will occur within a given time horizon and spatial radius of a reference event.

## Problem Statement

Given an earthquake at time `t` and location `P`, predict the probability of at least one M≥5.0 event occurring **within radius R = 200 km** over three forecasting horizons:

| Horizon | Label | Best ROC-AUC |
|---------|-------|-------------|
| 7 days  | `label_7d`   | **0.8586** |
| 30 days | `label_30d`  | 0.8362 *(in progress)* |
| 365 days| `label_365d` | 0.9194 |

Each horizon has its own trained LightGBM model.

## Dataset

| Source | Description | Size |
|--------|-------------|------|
| USGS Kaggle (1965–2016) + USGS FDSN API (1900–2026) | Primary catalog, M≥4.0 | 539,557 events |
| USGS FDSN API (2000–2026) | Supplementary M≥2 context catalog | 649,981 events |

Raw data is not versioned — regenerate with the scripts below.

**Temporal split**: train on events before 2010, test on 2010 and later. No data leakage.

## Feature Engineering

Features are computed per reference event using a **BallTree spatial index** (Haversine) within 200 km.

### Window features (M≥4.0, coherent 1900–2026)

Computed for 4 time windows: 1d, 7d, 30d, 90d

| Feature group | Columns |
|---------------|---------|
| Event count | `count_1d`, `count_7d`, `count_30d`, `count_90d` |
| Seismicity rate | `rate_1d`, `rate_7d`, `rate_30d`, `rate_90d` |
| Seismic energy | `energy_1d`, `energy_7d`, `energy_30d`, `energy_90d` |
| Seismic moment | `moment_1d`, `moment_7d`, `moment_30d`, `moment_90d` |
| Gutenberg-Richter b-value | `b_value_1d`, `b_value_7d`, `b_value_30d`, `b_value_90d` |
| Mean magnitude | `mag_mean_1d`, …, `mag_mean_90d` |
| Max magnitude | `mag_max_1d`, …, `mag_max_90d` |
| Mean depth | `depth_mean_1d`, …, `depth_mean_90d` |
| Depth std | `depth_std_1d`, …, `depth_std_90d` |

### M≥2 enriched features (post-1999 only — completeness bias corrected)

`count_7d_m2`, `energy_7d_m2`, `b_value_7d_m2`, `count_30d_m2`, `energy_30d_m2`, `b_value_30d_m2`, `count_90d_m2`, `energy_90d_m2`, `b_value_90d_m2`

Pre-2000 events have NaN for all M2 features (intentional — no catalog completeness bias).

### Acceleration / anomaly features

| Feature | Formula |
|---------|---------|
| `accel_count` | count_7d / (count_90d / 90 × 7) |
| `accel_energy` | energy_7d / (energy_90d / 90 × 7) |
| `mag_excess` | magnitude − mag_mean_90d |

### External features

| Source | Features |
|--------|---------|
| [GEM Global Active Faults](https://github.com/GEMScienceTools/gem-global-active-faults) | `dist_to_nearest_fault_km`, `fault_slip_type_enc` |
| [World Stress Map 2016](http://www.world-stress-map.org) | `stress_regime_enc`, `shmax_sin`, `shmax_cos`, `wsm_dist_km` |
| USGS pre-2010 catalog (M≥3) | `background_rate_yr` ← **#1 feature**, `normalized_rate_30d` |

`background_rate_yr`: historical seismicity rate (M≥3 events/year within 200 km, computed on pre-2010 data only to avoid leakage). Consistently the most discriminative feature across all horizons.

## Results

### ROC-AUC by horizon (LightGBM, temporal split train<2010 / test≥2010)

| Model version | Features | 7d AUC | Notes |
|---------------|----------|--------|-------|
| XGBoost baseline | 60 | 0.8558 | |
| LightGBM + ratio features | 54 | 0.8563 | |
| LightGBM + external data | 72 | 0.8583 | |
| **Multi-horizon (separate models)** | **71** | **0.8586** | current best |

### Top features (7d model)

1. `background_rate_yr` — historical seismicity density
2. `magnitude` — mainshock magnitude
3. `mag_mean_90d` — local magnitude regime
4. `count_90d` — long-term activity level
5. `dist_to_plate_boundary_km` — tectonic context
6. `energy_90d` — cumulative energy release
7. `wsm_dist_km` — distance to nearest stress measurement
8. `depth_mean_90d` — characteristic depth of the zone
9. `elapsed_since_last_s` — time since previous event
10. `dist_to_nearest_fault_km` — proximity to mapped faults

## ML Experiment Baseline

| Property | Value |
|----------|-------|
| Official benchmark | **benchmark_v2** |
| Dataset | `dataset_v5_dedup` (`data/features_clean/dataset_v5_dedup.csv`) |
| Active targets | `label_7d`, `label_30d` |
| Feature set | `candidate_feature_set_v1` (22 features) |
| Best params | `experiments/config/best_params_v2.yaml` (Optuna-tuned) |
| MLflow backend | SQLite — `mlflow.db` at project root |
| Tracking URI | `sqlite:////path/to/ml-project/mlflow.db` |

**All new ML experiments must start from benchmark_v2.** Use `--config experiments/config/benchmark_v2.yaml` when running any experiment script.

```bash
# View all runs
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:////Users/nazlidecker/ml-project/mlflow.db --host 127.0.0.1 --port 5001
```

### benchmark_v2 results (compact model — 22 features + Optuna params)

| Target | Val PR-AUC | Test PR-AUC | Test ROC-AUC |
|--------|-----------|------------|-------------|
| `label_7d`  | 0.7043 | **0.7896** | 0.8650 |
| `label_30d` | 0.8023 | **0.8546** | 0.8412 |

These match the full-feature model within 0.001–0.002, with 22 features instead of 70+.

## Reproduce

```bash
# 0. Install dependencies (Python 3.11 project venv)
pip3.9 install -r requirements.txt

# 1. Download primary catalog (M≥4, 1900–2026, chunked by year)
python src/download_data.py
python src/update_data.py

# 2. Download supplementary M≥2 catalog (2000–2026, chunked by semester)
python src/download_m2_catalog.py

# 3. Generate labels (label_7d, label_30d, label_365d)
python src/labels.py

# 4. Feature engineering (dual-catalog, two-track)
python src/features.py

# 5. Build training datasets
python src/prepare_and_train.py       # → dataset.csv
python src/add_ratio_features.py      # → dataset_v2.csv
python src/add_external_features.py   # → dataset_v3.csv

# 6. Train multi-horizon models
python src/train_multi_horizon.py

# 7. (Optional) 30d-specific feature engineering
python src/train_30d_enhanced.py
```

## Real-time Scoring API

A FastAPI service that polls the USGS real-time feed and scores incoming earthquakes against the trained models.

### Start

```bash
# Install API dependencies
pip3.9 install fastapi uvicorn apscheduler pydantic --break-system-packages

# Run (from project root)
source .venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status, models loaded, catalog freshness |
| `GET` | `/score/latest?n=10` | Score the N most recent M≥4 USGS events |
| `POST` | `/score` | Score a single event (manual input) |
| `GET` | `/events` | List recently scored events from cache |

### Runtime artifacts required by the API

The production API expects these runtime files to be present:

- `data/external/historical_catalog_pre2010_m3.csv.gz` — historical pre-2010 M≥3 catalog used for `background_rate_yr`
- `data/external/feature_medians_v3.json` — precomputed feature medians for inference-time imputation

The rolling catalog is built at runtime from USGS and covers the last 92 days of M≥2 events.
The initial rolling refresh is triggered in background at startup, so the API can come up before the first refresh completes.

### Example response (`/score/latest?n=1`)

```json
{
  "event_id": "us7000sba8",
  "latitude": 35.7,
  "longitude": 140.1,
  "depth": 42.0,
  "magnitude": 5.2,
  "datetime": "2026-04-08T10:30:00+00:00",
  "prob_7d": 0.6821,
  "prob_30d": 0.7134,
  "prob_365d": 0.9512,
  "risk_7d": "🟠 Élevé",
  "risk_30d": "🟠 Élevé",
  "risk_365d": "🔴 Très élevé",
  "features_used": 71,
  "scored_at": "2026-04-08T10:40:09+00:00"
}
```

### Architecture

```
src/api/
├── main.py              # FastAPI app, endpoints, USGS polling (every 5 min)
├── catalog_manager.py   # Rolling 92-day in-memory catalog + external data loader
├── feature_engine.py    # Single-event feature computation (mirrors features.py)
└── scorer.py            # LightGBM model loader, uses model.feature_name() for alignment
```

On startup: loads historical catalog + GEM faults + WSM + plate boundaries, fetches last 92 days from USGS, loads 3 LightGBM models, starts background polling.

## Repository structure

```
ml-project/
├── src/
│   ├── api/                      # Real-time scoring API
│   │   ├── main.py
│   │   ├── catalog_manager.py
│   │   ├── feature_engine.py
│   │   └── scorer.py
│   ├── labels.py                 # Label generation (Haversine spatial search)
│   ├── features.py               # Feature engineering (dual-catalog)
│   ├── prepare_and_train.py      # Dataset assembly + XGBoost baseline
│   ├── add_ratio_features.py     # Acceleration features
│   ├── add_external_features.py  # GEM + WSM + background rate
│   ├── train_multi_horizon.py    # One model per temporal horizon
│   ├── train_30d_enhanced.py     # 30d-specific feature experiments
│   ├── lightgbm_shap.py          # SHAP interpretability plots
│   └── download_m2_catalog.py    # M≥2 supplementary catalog download
├── data/
│   ├── raw/                      # Not versioned
│   ├── features/                 # Not versioned (large CSVs)
│   └── external/                 # GEM faults, WSM2016, plate boundaries
├── models/                       # Trained LightGBM models (.txt)
└── reports/                      # Classification reports + SHAP plots
```

## Design notes

**Why event-level, not grid-level?** The reference earthquake itself carries strong signal (magnitude, depth, sequence context). A grid approach loses this.

**Why temporal split at 2010?** Ensures the model never trains on the future. Spatial k-fold would leak seismicity patterns from the same fault zone.

**Why separate M≥2 features?** The M≥2 catalog only covers 2000–2026. Mixing it with M≥4 features (1900–2026) in the same time windows creates a completeness bias: pre-2000 events appear artificially quiescent. The two-track approach (coherent + M2-enriched) corrects this by assigning NaN to pre-2000 events for all M2 features.

**Why background_rate_yr is #1?** It encodes the tectonic regime of the zone in a single number. A zone with 50 events/year has fundamentally different aftershock statistics than one with 2 events/year. Normalizing by this baseline removes the regional bias and lets the model focus on anomalies.
