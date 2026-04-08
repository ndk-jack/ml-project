# CLAUDE.md

Guidance for working with this repository.

## Project Overview

Earthquake sequence forecasting: given a reference earthquake, predict whether a M≥5.0 follow-up will occur within 200 km over three temporal horizons (7d / 30d / 365d). Binary classification, one LightGBM model per horizon.

**Not** a magnitude regression problem. The old regression/classification pipeline (V1–V5) has been replaced entirely.

## Stack

Python 3.9 (Apple CLT), pandas, numpy, scikit-learn, lightgbm, shap, geopandas, shapely.

**Important**: on macOS, `python3` = 3.14 (Homebrew, no packages). Always use the full path:
```bash
/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 src/<script>.py
```

## Dataset

| File | Description | Size |
|------|-------------|------|
| `data/raw/database_updated.csv` | Primary catalog M≥4.0, 1900–2026 | ~540k events |
| `data/raw/catalog_m2_m4.csv` | Supplementary M≥2 context, 2000–2026 | ~650k events |
| `data/external/gem_active_faults.geojson` | GEM Global Active Faults | 16,195 segments |
| `data/external/wsm2016.csv` | World Stress Map 2016 | encoding=latin-1 |
| `data/external/PB2002_boundaries.json` | Plate boundaries | — |

Raw and feature files are **not versioned** (too large).

## Temporal Split (canonical — never change)

| Split | Criterion | Rows |
|-------|-----------|------|
| Train | year < 2010 | 296,831 |
| Test | year ≥ 2010 | 242,726 |

No spatial leakage: labels use only future events relative to each reference earthquake.

## Pipeline (run in order)

```bash
python3.9 src/labels.py                  # → data/features/labels.csv
python3.9 src/features.py                # → data/features/features.csv  (takes ~1h)
python3.9 src/prepare_and_train.py       # → data/features/dataset.csv
python3.9 src/add_ratio_features.py      # → data/features/dataset_v2.csv
python3.9 src/add_external_features.py   # → data/features/dataset_v3.csv
python3.9 src/train_multi_horizon.py     # → models/lgbm_{7d,30d,365d}.txt
```

## Current Results (best models)

| Horizon | ROC-AUC | Avg Precision | Notes |
|---------|---------|---------------|-------|
| 7d  | **0.8586** | 0.7777 | best overall |
| 30d | 0.8362 | 0.8498 | plateau (~50% pos rate) |
| 365d | 0.9194 | 0.9888 | near-trivial (high pos rate) |

## Feature Engineering

Features computed per reference event using BallTree (Haversine, 200 km radius).

**Two-track catalog** (completeness bias fix):
- `compute_coherent_window_features`: M≥4.0 events only, consistent 1900–2026
- `compute_m2_window_features`: M≥2 context catalog, NaN for pre-2000 events

**Key features** (by importance rank in 7d model):
1. `background_rate_yr` — M≥3 events/year within 200km, computed on pre-2010 data only (leakage-safe)
2. `magnitude` — reference event magnitude
3. `mag_mean_90d` — local magnitude regime
4. `count_90d`, `energy_90d` — long-term activity
5. `dist_to_plate_boundary_km` — tectonic context

**Never use** `b_value_trend = b_value_7d - b_value_90d` with NaN fill → produces mean ~1.8 billion. Use `b_trend_30d_90d = clip(b30 - b90, -1, 1)` instead.

## Modeling Conventions

- One model per label: `label_7d`, `label_30d`, `label_365d`
- `scale_pos_weight = neg / pos` for class imbalance
- LightGBM params: `n_estimators=1000, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, early_stopping=50`
- Drop always: `datetime, ref_lat, ref_lon, latitude, longitude, wsm_quality_enc` (importance≈0)
- Metric: ROC-AUC on test set (year ≥ 2010)

## Models & Reports

```
models/
  lgbm_7d.txt          # best 7d model  (ROC-AUC 0.8586)
  lgbm_30d.txt         # best 30d model (ROC-AUC 0.8362)
  lgbm_365d.txt        # best 365d model
  lgbm_30d_enhanced.txt
reports/
  multi_horizon_summary.txt
  train_30d_enhanced_report.txt
  lgbm_v3_classification_report.txt
```

## WSM encoding fix

```python
wsm = pd.read_csv("data/external/wsm2016.csv", encoding="latin-1", low_memory=False)
```
