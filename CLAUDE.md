# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project on USGS earthquake data (1900–2026, ~540k events). Two prediction tasks:
- **Regression**: predict earthquake magnitude (target: `Magnitude`)
- **Classification**: predict if an earthquake is dangerous — `Magnitude >= 6.0` → `Dangerous=1`

## Stack

Python 3.9, pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, requests, joblib.
No virtual environment — packages installed via `pip3` to user site-packages (`~/Library/Python/3.9`).
Jupyter installed but not in PATH; run with `~/Library/Python/3.9/bin/jupyter notebook`.

## Pipeline

```bash
python3 src/download_data.py       # Kaggle USGS 1965–2016 → data/database.csv
python3 src/update_data.py         # USGS API 1900–today  → data/database_updated.csv
python3 src/preprocess_updated.py  # clean + feature engineering → data/clean_updated.csv
python3 src/train.py               # train all models → outputs/
```

## Dataset

**Raw**: `data/database_updated.csv` — 540 796 rows, 23 columns, from USGS FDSN API chunked by year.
**Clean**: `data/clean_updated.csv` — 539 030 rows, 10 columns, after filtering `Type == earthquake` and dropping columns with >50% NaN.

Key columns after preprocessing: `Date`, `Latitude`, `Longitude`, `Depth`, `Magnitude`, `Dangerous`, `dist_fault_km`, `IsCoastal`, `Season`, `Depth_category`.

Class imbalance: 97.4% non-dangerous (0) / 2.6% dangerous (1).

## Features (12 total)

| Feature | Type | Notes |
|---------|------|-------|
| `Latitude`, `Longitude` | float | Raw coordinates |
| `Depth` | float | Depth in km |
| `Year`, `Month` | int | Extracted from `Date` |
| `dist_fault_km` | float | Haversine distance to nearest active fault (GEM GeoJSON, cKDTree) |
| `IsCoastal` | int (0/1) | 1 if within 100 km of a coastline (Natural Earth 50m) |
| `Depth_category_enc` | int (0/1/2) | OrdinalEncoder: superficiel=0, intermédiaire=1, profond=2 |
| `season_*` | int (0/1) | One-hot encoded from `Season` (Automne/Hiver/Printemps/Été) |

## Best Models & Results

**Regression** (predicting `Magnitude`):

| Model | RMSE | R² |
|-------|------|-----|
| RandomForestRegressor | **0.4091** | **0.366** |
| HistGradientBoostingRegressor | 0.4108 | 0.361 |
| LinearRegression | 0.4622 | 0.191 |

Saved: `outputs/best_regressor.joblib`

**Classification** (predicting `Dangerous`):

| Model | Threshold | Precision | Recall | F1 |
|-------|-----------|-----------|--------|----|
| GradientBoostingClassifier | 0.25 | 0.575 | 0.408 | **0.480** |

Saved: `outputs/best_classifier.joblib`, `outputs/best_threshold.txt`

## Scientific Limits

R² is capped at ~0.37 with public data. Magnitude depends on fault rupture mechanics, stress accumulation, and subsurface geology — none of which are captured by location, depth, and time alone. Improvement would require seismic moment tensors, GPS crustal velocity data, or waveform features unavailable in the USGS catalog.

## Modeling Conventions

- Train/test split: `test_size=0.2, random_state=42`
- No `class_weight='balanced'` on GradientBoosting — threshold tuning used instead (seuil=0.25)
- Categorical encoding: `OrdinalEncoder` for ordered categories, `get_dummies` for nominal
- All models saved with `joblib.dump` to `outputs/`
- All plots saved to `outputs/` via `matplotlib.use('Agg')` (no display)
