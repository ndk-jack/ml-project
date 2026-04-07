"""
lightgbm_shap.py — LightGBM classifier + SHAP analysis for earthquake prediction.

Loads the pre-merged dataset (data/features/dataset.csv), trains LightGBM on
label_7d with the same temporal split as the XGBoost baseline, then generates
SHAP plots for global and local interpretability.

Usage : python3 src/lightgbm_shap.py
Output:
  models/lgbm_label7d.txt
  reports/lgbm_classification_report.txt
  reports/shap_summary.png
  reports/shap_bar.png
  reports/shap_waterfall_sample.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
from pathlib import Path
import warnings, subprocess, sys

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = Path("data/features/dataset.csv")
MODEL_OUT     = Path("models/lgbm_label7d.txt")
REPORT_OUT    = Path("reports/lgbm_classification_report.txt")
SHAP_DIR      = Path("reports")

TARGET        = "label_7d"
TEST_YEAR     = 2010
RANDOM_STATE  = 42
N_SHAP_SAMPLE = 5_000     # SHAP on a sample for speed (full dataset = slow)


# ── Install dependencies if needed ───────────────────────────────────────────

def ensure(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


ensure("lightgbm")
ensure("shap")
ensure("scikit-learn", "sklearn")

import lightgbm as lgb
import shap
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
)


# ── 1. Load dataset ───────────────────────────────────────────────────────────

def load_dataset():
    print("── 1. Loading dataset ──────────────────────────────────────────")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── 2. Temporal split ─────────────────────────────────────────────────────────

def temporal_split(df):
    print(f"\n── 2. Temporal split (train < {TEST_YEAR}, test >= {TEST_YEAR}) ─")

    drop_cols = ["datetime", "ref_lat", "ref_lon",
                 "latitude", "longitude",
                 "label_7d", "label_30d", "label_365d"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df["_year"] = pd.to_datetime(df["datetime"]).dt.year
    train_mask   = df["_year"] < TEST_YEAR
    test_mask    = df["_year"] >= TEST_YEAR

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, TARGET]
    X_test  = df.loc[test_mask,  feature_cols]
    y_test  = df.loc[test_mask,  TARGET]

    print(f"   Train : {len(X_train):>7,}  pos_rate={y_train.mean():.2%}")
    print(f"   Test  : {len(X_test):>7,}  pos_rate={y_test.mean():.2%}")
    print(f"   Features : {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


# ── 3. Train LightGBM ─────────────────────────────────────────────────────────

def train_lgbm(X_train, y_train, X_test, y_test):
    print("\n── 3. Training LightGBM ────────────────────────────────────────")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"   scale_pos_weight = {spw:.2f}")

    params = dict(
        objective         = "binary",
        metric            = "auc",
        n_estimators      = 1000,
        learning_rate     = 0.05,
        num_leaves        = 63,
        max_depth         = -1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 50,
        scale_pos_weight  = spw,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = -1,
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    print(f"\n   Best iteration : {model.best_iteration_}")
    return model


# ── 4. Evaluate ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, feature_cols):
    print("\n── 4. Evaluation ───────────────────────────────────────────────")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    roc_auc  = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=["No event", "Event ≥M5"])

    print(f"\n   ROC-AUC        : {roc_auc:.4f}   (XGBoost baseline: 0.8562)")
    print(f"   Avg Precision  : {avg_prec:.4f}   (XGBoost baseline: 0.7750)")
    print(f"\n   Confusion matrix:")
    print(f"             Predicted 0   Predicted 1")
    print(f"   Actual 0  {cm[0,0]:>10,}   {cm[0,1]:>10,}")
    print(f"   Actual 1  {cm[1,0]:>10,}   {cm[1,1]:>10,}")
    print(f"\n{report}")

    # Top 20 importances
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("   Top 20 feature importances (gain):")
    for feat, imp in importances.head(20).items():
        bar = "█" * int(imp / importances.max() * 40)
        print(f"   {feat:<40} {imp:>8.0f}  {bar}")

    return roc_auc, avg_prec, report, importances, y_proba


# ── 5. SHAP analysis ──────────────────────────────────────────────────────────

def shap_analysis(model, X_test, feature_cols):
    print(f"\n── 5. SHAP analysis (sample n={N_SHAP_SAMPLE:,}) ───────────────")

    # Sample for speed
    rng     = np.random.default_rng(RANDOM_STATE)
    idx     = rng.choice(len(X_test), size=min(N_SHAP_SAMPLE, len(X_test)),
                         replace=False)
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    print("   Computing SHAP values …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary: shap_values is a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # ── Plot 1: Beeswarm summary ──────────────────────────────────────────
    print("   Saving shap_summary.png …")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.title("SHAP Summary — Top 20 Features (label_7d)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Bar chart (mean |SHAP|) ──────────────────────────────────
    print("   Saving shap_bar.png …")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_cols,
                      plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP| — Feature Importance (label_7d)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Waterfall for the highest-confidence positive prediction ──
    print("   Saving shap_waterfall_sample.png …")
    base_val   = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    # Find the sample with highest predicted probability (most confident positive)
    proba_sample = model.predict_proba(X_sample)[:, 1]
    top_idx      = int(np.argmax(proba_sample))

    expl_obj = shap.Explanation(
        values    = sv[top_idx],
        base_values = base_val,
        data      = X_sample.iloc[top_idx].values,
        feature_names = feature_cols,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(expl_obj, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Most Confident Positive Prediction\n"
              f"(predicted proba = {proba_sample[top_idx]:.3f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_waterfall_sample.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("   SHAP plots saved.")
    return sv, X_sample


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    df = load_dataset()
    X_train, X_test, y_train, y_test, feature_cols = temporal_split(df)
    model = train_lgbm(X_train, y_train, X_test, y_test)

    roc_auc, avg_prec, report, importances, y_proba = evaluate(
        model, X_test, y_test, feature_cols
    )

    # Save model
    model.booster_.save_model(str(MODEL_OUT))
    print(f"\n   Model saved → {MODEL_OUT}")

    # Save report
    report_text = (
        f"Model         : LightGBM\n"
        f"Target        : {TARGET}\n"
        f"Test from year: {TEST_YEAR}\n"
        f"ROC-AUC       : {roc_auc:.4f}  (XGBoost baseline: 0.8562)\n"
        f"Avg Precision : {avg_prec:.4f}  (XGBoost baseline: 0.7750)\n\n"
        f"{report}\n\n"
        f"Top 20 features:\n{importances.head(20).to_string()}\n"
    )
    REPORT_OUT.write_text(report_text)
    print(f"   Report saved → {REPORT_OUT}")

    # SHAP
    shap_analysis(model, X_test, feature_cols)

    print("\n── Done ────────────────────────────────────────────────────────")
    print(f"   LightGBM ROC-AUC : {roc_auc:.4f}")
    print(f"   SHAP plots       : reports/shap_*.png")


if __name__ == "__main__":
    main()
