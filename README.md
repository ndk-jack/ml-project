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

## Reproduce

```bash
# 0. Install dependencies (Python 3.9 required on macOS with Apple CLT)
pip3.9 install -r requirements.txt

# 1. Download primary catalog (M≥4, 1900–2026, chunked by year)
python3.9 src/download_data.py
python3.9 src/update_data.py

# 2. Download supplementary M≥2 catalog (2000–2026, chunked by semester)
python3.9 src/download_m2_catalog.py

# 3. Generate labels (label_7d, label_30d, label_365d)
python3.9 src/labels.py

# 4. Feature engineering (dual-catalog, two-track)
python3.9 src/features.py

# 5. Build training datasets
python3.9 src/prepare_and_train.py       # → dataset.csv
python3.9 src/add_ratio_features.py      # → dataset_v2.csv
python3.9 src/add_external_features.py   # → dataset_v3.csv

# 6. Train multi-horizon models
python3.9 src/train_multi_horizon.py

# 7. (Optional) 30d-specific feature engineering
python3.9 src/train_30d_enhanced.py
```

## Repository structure

```
ml-project/
├── src/
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
