"""
train_multi_horizon.py — Train one LightGBM model per temporal horizon.

Labels:
  label_7d    → M≥5 within  7 days  & 200 km
  label_30d   → M≥5 within 30 days  & 200 km
  label_365d  → M≥5 within 365 days & 200 km

Each horizon gets its own model so features are optimised per time-scale.
Also drops wsm_quality_enc (importance≈0, wastes splits).

Usage : python3 src/train_multi_horizon.py
Output:
  models/lgbm_{horizon}.txt           (one per horizon)
  reports/multi_horizon_summary.txt
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET      = Path("data/features/dataset_v3.csv")
MODELS_DIR   = Path("models")
REPORTS_DIR  = Path("reports")

HORIZONS     = ["label_7d", "label_30d", "label_365d"]
TEST_YEAR    = 2010
RANDOM_STATE = 42

# Features to always drop (identifiers, other labels, near-zero importance)
DROP_ALWAYS  = [
    "datetime", "ref_lat", "ref_lon", "latitude", "longitude",
    "label_7d", "label_30d", "label_365d",
    "wsm_quality_enc",    # importance ≈ 0 in all previous runs
]

LGB_PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = -1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 50,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbosity         = -1,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def temporal_split(df, feature_cols, target_col):
    year       = pd.to_datetime(df["datetime"]).dt.year
    train_mask = year < TEST_YEAR
    test_mask  = year >= TEST_YEAR

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_test  = df.loc[test_mask,  feature_cols]
    y_test  = df.loc[test_mask,  target_col]

    pos_tr = y_train.mean()
    pos_te = y_test.mean()
    print(f"   Train : {len(X_train):,} rows  pos={pos_tr:.1%}")
    print(f"   Test  : {len(X_test):,}  rows  pos={pos_te:.1%}")
    return X_train, X_test, y_train, y_test


def train_horizon(df, target_col, feature_cols):
    X_train, X_test, y_train, y_test = temporal_split(df, feature_cols, target_col)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"   scale_pos_weight = {spw:.2f}  (neg={neg:,} / pos={pos:,})")

    model = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=spw)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)
    roc     = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)
    rep     = classification_report(y_test, y_pred,
                                    target_names=["No event", "Event ≥M5"])
    cm      = confusion_matrix(y_test, y_pred)

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return model, roc, ap, rep, cm, imp


def bar(val, max_val, width=30):
    n = int(val / max_val * width) if max_val > 0 else 0
    return "█" * n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("── Loading dataset_v3 ──────────────────────────────────────────")
    df = pd.read_csv(DATASET, low_memory=False)
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Feature columns common to all horizons
    feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
    print(f"   Features available: {len(feature_cols)}")
    print(f"   Dropped: {[c for c in DROP_ALWAYS if c in df.columns]}")

    results = {}

    for target in HORIZONS:
        horizon = target.replace("label_", "")
        print(f"\n{'═'*60}")
        print(f"   HORIZON : {horizon}   (target = {target})")
        print(f"{'═'*60}")

        model, roc, ap, rep, cm, imp = train_horizon(df, target, feature_cols)

        # Save model
        model_path = MODELS_DIR / f"lgbm_{horizon}.txt"
        model.booster_.save_model(str(model_path))

        results[horizon] = dict(roc=roc, ap=ap, rep=rep, cm=cm, imp=imp,
                                model_path=model_path)

        # Print confusion matrix
        print(f"\n   ROC-AUC : {roc:.4f}   Avg Precision : {ap:.4f}   "
              f"Best iter : {model.best_iteration_}")
        print(f"\n   Confusion matrix:")
        print(f"             Predicted 0   Predicted 1")
        print(f"   Actual 0  {cm[0,0]:>10,}   {cm[0,1]:>10,}")
        print(f"   Actual 1  {cm[1,0]:>10,}   {cm[1,1]:>10,}")
        print(f"\n{rep}")

        # Top 15 feature importances
        top15   = imp.head(15)
        max_imp = top15.max()
        print(f"   Top 15 features for {horizon}:")
        for feat, val in top15.items():
            print(f"   {feat:<35} {val:>6.0f}  {bar(val, max_imp)}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("   SUMMARY — ROC-AUC per horizon")
    print(f"{'═'*60}")
    for horizon, res in results.items():
        mark = " ← best" if res["roc"] == max(r["roc"] for r in results.values()) else ""
        print(f"   {horizon:<8}  ROC-AUC={res['roc']:.4f}  "
              f"AvgPrec={res['ap']:.4f}{mark}")

    # ── Feature importance comparison across horizons ─────────────────────────
    print(f"\n── Top-10 features per horizon (rank comparison) ───────────────")
    header = f"{'Feature':<35}"
    for h in results:
        header += f"  {h:>8}"
    print("   " + header)
    print("   " + "-" * len(header))

    # Collect all top-10 features from each horizon
    all_top = set()
    for res in results.values():
        all_top.update(res["imp"].head(10).index.tolist())

    # For each feature, show rank in each horizon
    rows = []
    for feat in all_top:
        row = {"feat": feat}
        for h, res in results.items():
            idx_list = list(res["imp"].index)
            row[h] = idx_list.index(feat) + 1 if feat in idx_list else 999
        row["min_rank"] = min(row[h] for h in results)
        rows.append(row)

    rows.sort(key=lambda r: r["min_rank"])
    for row in rows:
        line = f"   {row['feat']:<35}"
        for h in results:
            rank = row[h]
            line += f"  {'#'+str(rank):>8}" if rank < 999 else f"  {'—':>8}"
        print(line)

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = REPORTS_DIR / "multi_horizon_summary.txt"
    lines = ["Multi-horizon LightGBM — Summary\n", "=" * 50 + "\n"]
    for horizon, res in results.items():
        lines.append(f"\n[{horizon}]  ROC-AUC={res['roc']:.4f}  AvgPrec={res['ap']:.4f}\n")
        lines.append(res["rep"] + "\n")
        lines.append("Top 20 features:\n")
        lines.append(res["imp"].head(20).to_string() + "\n")
    report_path.write_text("".join(lines))
    print(f"\n   Report → {report_path}")

    for horizon, res in results.items():
        print(f"   Model  → {res['model_path']}")


if __name__ == "__main__":
    main()
