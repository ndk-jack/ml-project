# Earthquake Risk Forecasting Project

This project aims to estimate future earthquake risk from historical seismic observations.  
The goal is not to predict the exact time, location, and magnitude of a single earthquake, but to estimate the probability of at least one event above a chosen magnitude threshold within a spatial radius and a future time window.

## Problem Statement

At time `t`, for an anchor point `P`, estimate the probability of at least one earthquake with magnitude ≥ `M` occurring within radius `R` over horizon `H`.

## Project Scope

The project is framed as a spatio-temporal forecasting problem.

- Study area: local or regional
- Spatial unit: regular grid of anchor points
- Spatial reference: local projection in kilometers
- Prediction horizons: `7d`, `30d`, `365d`
- Initial modeling focus: `30d`

## Dataset

**Source**: [USGS Earthquake Database](https://www.kaggle.com/datasets/usgs/earthquake-database) + USGS FDSN API  
**Size**: 539 030 earthquakes after cleaning (magnitude ≥ 4.0, 1900–2026)  
**Raw data**: not versioned — regenerate with the scripts below.

## Features

| Feature | Description |
|---------|-------------|
| `Latitude`, `Longitude` | Epicenter coordinates |
| `Depth` | Hypocenter depth (km) |
| `Year`, `Month` | Extracted from event date |
| `dist_fault_km` | Distance to nearest active fault (GEM global catalog, Haversine via cKDTree) |
| `IsCoastal` | 1 if within 100 km of a coastline (Natural Earth 50m) |
| `Depth_category` | superficiel (<70 km) / intermédiaire / profond (>300 km) |
| `Season` | Spring / Summer / Autumn / Winter |

`dist_fault_km` is the most informative engineered feature: earthquakes near active faults tend to cluster in distinct magnitude regimes.

## Results

### Regression — predicting Magnitude

| Model | RMSE | R² |
|-------|------|-----|
| RandomForestRegressor | **0.409** | **0.366** |
| HistGradientBoostingRegressor | 0.411 | 0.361 |
| LinearRegression | 0.462 | 0.191 |

### Classification — predicting Dangerous (Magnitude ≥ 6.0)

| Model | Threshold | Precision | Recall | F1 |
|-------|-----------|-----------|--------|----|
| GradientBoostingClassifier | 0.25 | 0.575 | 0.408 | **0.480** |

Class imbalance: 97.4% non-dangerous / 2.6% dangerous. Threshold lowered to 0.25 (vs default 0.50) to improve recall on the minority class.

## Scientific Conclusion

**R² plateaus at ~0.37** regardless of the model. This is expected: earthquake magnitude is governed by fault rupture mechanics, accumulated tectonic stress, and subsurface geology — variables absent from the public USGS catalog. Location, depth, and time are weak proxies. This project demonstrates that a ceiling exists with observational catalog data alone, and that more informative features (seismic moment tensors, GPS crustal velocities, fault locking rates) would be required to push further.

## Reproduce

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Download USGS Kaggle dataset (1965–2016)
python3 src/download_data.py

# 3. Download USGS API data (1900–2026, ~126 requests, several minutes)
python3 src/update_data.py

# 4. Clean data + feature engineering
python3 src/preprocess_updated.py

# 5. Train and evaluate all models
python3 src/train.py
```

Outputs saved to `outputs/`: trained models (`.joblib`), best threshold, and plots.
