# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project on USGS earthquake data (1965–2016). Two prediction tasks:
- **Regression**: predict earthquake magnitude
- **Classification**: predict if an earthquake is dangerous (magnitude > 6.0)

## Stack

Python 3, pandas, numpy, matplotlib, scikit-learn. No virtual environment — packages installed via `pip3` to user site-packages.

## Commands

```bash
# Download/refresh dataset
python3 src/download_data.py

# Run any script
python3 src/<script>.py

# Run a notebook (if jupyter installed)
jupyter notebook notebooks/
```

## Dataset

**File**: `data/database.csv` — sourced from Kaggle (`usgs/earthquake-database`), downloaded via `src/download_data.py`.

**Key columns**:
- `Date`, `Time` — event timestamp
- `Latitude`, `Longitude` — location
- `Depth` — depth in km
- `Magnitude` — target variable for regression
- `Type` — event type (filter to `Earthquake` only)
- `Magnitude Type`, `Depth Error`, etc. — sparse metadata, handle missing values carefully

**Label for classification**: `Magnitude > 6.0` → dangerous (1), else safe (0). Class imbalance expected.

## Project Structure

- `src/` — Python scripts (data download, preprocessing, training, evaluation)
- `notebooks/` — exploratory analysis and model experimentation
- `data/` — raw dataset (not committed to git)
- `outputs/` — saved models, plots, metrics
- `.claude/commands/` — custom Claude Code slash commands
- `.claude/rules/` — project-specific Claude rules
- `.claude/skills/data-analysis/` — reusable data analysis skills

## Key Modeling Notes

- Drop or impute sparse columns (`Depth Error`, `Magnitude Error`, `Azimuthal Gap`, etc.) before training
- Features to use: `Latitude`, `Longitude`, `Depth`, optionally engineered time features (year, month)
- For classification, address class imbalance (e.g. `class_weight='balanced'` or resampling)
- Persist trained models to `outputs/` using `joblib`
- Evaluation: regression → RMSE/MAE/R²; classification → precision, recall, F1, confusion matrix
