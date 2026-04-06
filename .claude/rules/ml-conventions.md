# ML Conventions

## Data
- Raw data lives in `data/database.csv`, cleaned data in `data/clean_database.csv`
- Never overwrite raw data — always write cleaned versions to a new file
- Features used: `Latitude`, `Longitude`, `Depth` (+ engineered time features if needed)
- Target for regression: `Magnitude`
- Target for classification: `Dangerous` (1 if Magnitude >= 6.0, else 0)

## Preprocessing
- Drop columns with > 50% missing values
- Filter `Type == "Earthquake"` to exclude nuclear explosions and other events
- Use `dropna()` after column selection — no imputation on sparse columns

## Modeling
- Always split with `train_test_split(test_size=0.2, random_state=42)`
- For classification, always set `class_weight='balanced'` to handle the ~2:1 imbalance
- Save trained models to `outputs/` with `joblib.dump(model, "outputs/<name>.joblib")`

## Evaluation
- Regression: report RMSE, MAE, R²
- Classification: report precision, recall, F1, confusion matrix
- Always print metrics to stdout and save plots to `outputs/`

## Code style
- Scripts are standalone and run from the project root: `python3 src/<script>.py`
- Use `matplotlib.use('Agg')` at the top of any script that saves plots (no display needed)
- One script per concern: preprocess, train, evaluate — no monolithic notebooks for production code
