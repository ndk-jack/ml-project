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
python3 src/train_time_split.py    # V1 baseline (canonical reference)
```

## Dataset

**Raw**: `data/database_updated.csv` — 540 796 rows, 23 columns, from USGS FDSN API chunked by year.
**Clean**: `data/clean_updated.csv` — 539 030 rows, 10 columns. Pre-computed features: `dist_fault_km`, `IsCoastal`, `Season`, `Depth_category`.

Merge strategy (used in all experiment scripts): dedup `clean_updated` on `(Date, Latitude, Longitude, Depth)` → left-join on exact floats. Match rate: 99.85% (792 rows median-imputed by pipeline).

Class imbalance: train 3.1% dangerous / val+test ~0.9%. The drop is a detection network expansion artifact, not a real hazard change.

## Time Split (canonical — never change)

| Split | Years | Rows |
|-------|-------|------|
| Train | ≤ 2018 | 427 290 |
| Val | 2019–2022 | 60 431 |
| Test | 2023–2025 | 48 577 |

## V1 Reference Metrics (locked)

Classifier: `RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)`
Threshold tuned on val F2 (500 points, 0.01–0.60).

| Split | PR-AUC | F2 |
|-------|--------|----|
| Val | 0.3479 | 0.5460 |
| Test | 0.3504 | 0.5200 |

Source: `outputs/metrics/metrics_time_split.json`

## Experiment Branches & Scripts

All experiments use the same RF settings and split as V1. IMPROVEMENT verdict requires **both** val PR-AUC > 0.3479 **and** test PR-AUC > 0.3504.

| Branch | Script | Verdict | Key result |
|--------|--------|---------|------------|
| `feature/modeling-v1` | `src/train_time_split.py` | — | Canonical V1 baseline |
| `experiment/modeling-v2-classification` | `src/tune_classification_v2.py` | — | Hyperparameter search (5-iter RandomizedCV, 2-fold, 20% subsample + prefit calibration) |
| `experiment/modeling-v3-features` | `src/train_features_v3.py` | — | Initial feature engineering exploration |
| `experiment/modeling-v3-1-controlled` | `src/train_features_v3_controlled.py` | **MIXED** | Full V3 set: val PR-AUC +0.0011, test F2 +0.020 but test PR-AUC −0.0017 |
| `experiment/modeling-v4-temporal-reweighting` | `src/train_temporal_reweighting_v4.py` | **NO IMPROVEMENT** | Piecewise weights (1.0→3.0 by era): test PR-AUC −0.0076, test F2 −0.008 |
| `experiment/modeling-v5-feature-ablation` | `src/train_feature_ablation_v5.py` | **weak_signal** | See ablation table below |

## V5 Ablation Results

Isolates which feature caused the V3.1 +0.020 test F2 gain.

| Setup | N | val PR-AUC | val F2 | test PR-AUC | test F2 | Verdict |
|-------|---|-----------|--------|------------|--------|---------|
| baseline_v1 | 10 | 0.3479 | 0.5460 | 0.3504 | 0.5200 | NO IMPROVEMENT |
| +dist_fault_km | 11 | 0.3505 | 0.5370 | 0.3385 | 0.5180 | MIXED |
| +IsCoastal | 11 | 0.3405 | 0.5245 | 0.3374 | 0.5217 | NO IMPROVEMENT |
| +dist_fault_km+IsCoastal | 12 | 0.3490 | 0.5385 | 0.3314 | 0.5115 | MIXED |
| +full_v3_engineered | 18 | 0.3490 | 0.5428 | 0.3487 | 0.5399 | MIXED |

**Conclusion**: No feature addition consistently improves over V1. The +0.020 test F2 in V3.1 is threshold variance, not a real gain. The feature engineering path is a dead end for this dataset.

## Scientific Limits

R² is capped at ~0.37 for regression. PR-AUC is capped near 0.35 for classification. Magnitude depends on fault rupture mechanics, stress accumulation, and subsurface geology — none captured by location, depth, and time alone. The train→val/test class rate drop (3.1% → 0.9%) reflects network expansion, not model error; temporal reweighting does not fix it.

## Outputs

```
outputs/metrics/metrics_time_split.json               # V1 reference
outputs/metrics/metrics_feature_v3_controlled.json    # V3.1 controlled
outputs/metrics/metrics_temporal_reweighting_v4.json  # V4
outputs/metrics/metrics_feature_ablation_v5.json      # V5 ablation
outputs/models/best_classifier_v3_controlled.joblib
outputs/models/best_classifier_v4.joblib
outputs/models/best_classifier_v5.joblib              # best = baseline_v1 (F2=0.546)
```

## Modeling Conventions

- Scripts are run from project root: `python3 src/<script>.py`
- Preprocessing pipeline: `ColumnTransformer(SimpleImputer(median) + OneHotEncoder(handle_unknown='ignore'))`
- Threshold tuning always on val only, never on test
- All plots saved to `outputs/` via `matplotlib.use('Agg')`
