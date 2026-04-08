"""
train_30d_enhanced.py — 30-day horizon model with window-specific features.

Motivation: label_30d scored 0.8362 vs 0.8586 for 7d — the 30d horizon sits
between two regimes: too far for recent-activity features to dominate, too close
for geology alone to suffice. New features target that intermediate scale.

New features (all derived from existing columns — no re-running features.py):

  Omori-decay proxy
    omori_proxy        = count_1d / (count_7d / 7 + eps)
                         High → sequence is very fresh, rate still accelerating
                         Low  → Omori decay already underway

  Medium-scale seismic acceleration (7d vs 30d)
    rate_7d_vs_30d     = rate_7d / (rate_30d + eps)
    energy_7d_vs_30d   = energy_7d / (energy_30d / 30 * 7 + eps)
    moment_7d_vs_30d   = moment_7d / (moment_30d / 30 * 7 + eps)

  Stable b-value trend (clipped to [-1, 1] — avoids NaN-fill explosion)
    b_trend_30d_90d    = clip(b_value_30d - b_value_90d, -1, 1)
                         Negative = b dropping = stress accumulating

  Depth migration (seismicity getting shallower?)
    depth_migration    = depth_mean_7d - depth_mean_30d
                         Positive = recent quakes shallower than monthly avg

  Bath's law proxy (foreshock indicator)
    bath_proxy         = magnitude - (mag_max_30d - 1.2)
                         If positive → this event could be a foreshock

  Magnitude acceleration (recent events bigger than monthly avg?)
    mag_accel_7d_30d   = mag_mean_7d - mag_mean_30d

  30d anomaly vs historical background
    count_30d_norm     = count_30d / (background_rate_yr * 30/365.25 + eps)

Usage : python3 src/train_30d_enhanced.py
Output:
  models/lgbm_30d_enhanced.txt
  reports/train_30d_enhanced_report.txt
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
MODEL_OUT    = Path("models/lgbm_30d_enhanced.txt")
REPORT_OUT   = Path("reports/train_30d_enhanced_report.txt")

TARGET       = "label_30d"
BASELINE_AUC = 0.8362       # from train_multi_horizon.py
TEST_YEAR    = 2010
RANDOM_STATE = 42
EPS          = 1e-9

DROP_ALWAYS  = [
    "datetime", "ref_lat", "ref_lon", "latitude", "longitude",
    "label_7d", "label_30d", "label_365d",
    "wsm_quality_enc",
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


# ── Feature engineering ───────────────────────────────────────────────────────

def add_30d_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    new_feats = []

    def safe_div(a, b):
        return a / (b + EPS)

    # 1. Omori-decay proxy: very short-term activity vs 7d average
    df["omori_proxy"] = safe_div(
        df["count_1d"],
        df["count_7d"] / 7.0
    ).clip(0, 100)
    new_feats.append("omori_proxy")

    # 2. Medium-scale seismic acceleration (7d vs 30d)
    df["rate_7d_vs_30d"] = safe_div(
        df["rate_7d"],
        df["rate_30d"]
    ).clip(0, 200)
    new_feats.append("rate_7d_vs_30d")

    df["energy_7d_vs_30d"] = safe_div(
        df["energy_7d"],
        df["energy_30d"] / 30.0 * 7.0
    ).clip(0, 200)
    new_feats.append("energy_7d_vs_30d")

    df["moment_7d_vs_30d"] = safe_div(
        df["moment_7d"],
        df["moment_30d"] / 30.0 * 7.0
    ).clip(0, 200)
    new_feats.append("moment_7d_vs_30d")

    # 3. Stable b-value trend (clipped — avoids the NaN-fill explosion)
    b30 = df["b_value_30d"].fillna(df["b_value_90d"]).fillna(1.0)
    b90 = df["b_value_90d"].fillna(1.0)
    df["b_trend_30d_90d"] = (b30 - b90).clip(-1, 1)
    new_feats.append("b_trend_30d_90d")

    # 4. Depth migration: are recent quakes shallower?
    depth_7d  = df["depth_mean_7d"].fillna(df["depth"])
    depth_30d = df["depth_mean_30d"].fillna(df["depth"])
    df["depth_migration"] = (depth_7d - depth_30d).clip(-50, 50)
    new_feats.append("depth_migration")

    # 5. Bath's law proxy: mainshock-foreshock indicator
    #    Bath: largest aftershock ≈ mainshock - 1.2
    #    If magnitude > mag_max_30d - 1.2, this could be a foreshock
    mag_max_30d = df["mag_max_30d"].fillna(df["magnitude"])
    df["bath_proxy"] = (df["magnitude"] - (mag_max_30d - 1.2)).clip(-3, 3)
    new_feats.append("bath_proxy")

    # 6. Magnitude acceleration: recent events bigger than monthly avg?
    mag_mean_7d  = df["mag_mean_7d"].fillna(df["magnitude"])
    mag_mean_30d = df["mag_mean_30d"].fillna(df["magnitude"])
    df["mag_accel_7d_30d"] = (mag_mean_7d - mag_mean_30d).clip(-2, 2)
    new_feats.append("mag_accel_7d_30d")

    # 7. 30d count anomaly vs historical background
    if "background_rate_yr" in df.columns:
        expected_30d = df["background_rate_yr"] * (30.0 / 365.25) + EPS
        df["count_30d_norm"] = (df["count_30d"] / expected_30d).clip(0, 500)
        new_feats.append("count_30d_norm")

    return df, new_feats


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(df, feature_cols, label=""):
    year       = pd.to_datetime(df["datetime"]).dt.year
    train_mask = year < TEST_YEAR
    test_mask  = year >= TEST_YEAR

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, TARGET]
    X_test  = df.loc[test_mask,  feature_cols]
    y_test  = df.loc[test_mask,  TARGET]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    print(f"\n   Train: {len(X_train):,} rows  pos={y_train.mean():.1%}  "
          f"spw={spw:.2f}")
    print(f"   Test : {len(X_test):,}  rows  pos={y_test.mean():.1%}")

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
    imp     = pd.Series(model.feature_importances_,
                        index=feature_cols).sort_values(ascending=False)

    print(f"\n   [{label}]")
    print(f"   ROC-AUC : {roc:.4f}   Avg Precision: {ap:.4f}   "
          f"Best iter: {model.best_iteration_}")
    print(f"\n   Confusion matrix:")
    print(f"             Predicted 0   Predicted 1")
    print(f"   Actual 0  {cm[0,0]:>10,}   {cm[0,1]:>10,}")
    print(f"   Actual 1  {cm[1,0]:>10,}   {cm[1,1]:>10,}")
    print(f"\n{rep}")

    return model, roc, ap, rep, imp


def print_top_features(imp, n=15, new_feats=None):
    max_imp = imp.max()
    print(f"\n   Top {n} features:")
    for i, (feat, val) in enumerate(imp.head(n).items(), 1):
        bar = "█" * int(val / max_imp * 30)
        mark = " ← NEW" if new_feats and feat in new_feats else ""
        print(f"   #{i:<3} {feat:<35} {val:>6.0f}  {bar}{mark}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    print("── Loading dataset_v3 ──────────────────────────────────────────")
    df = pd.read_csv(DATASET, low_memory=False)
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Add 30d-specific features ─────────────────────────────────────────────
    print("\n── Engineering 30d-specific features ──────────────────────────")
    df, new_feats = add_30d_features(df)
    print(f"   Added {len(new_feats)} features:")
    for f in new_feats:
        print(f"   {f:<30}  mean={df[f].mean():>8.3f}  "
              f"std={df[f].std():>8.3f}  NaN={df[f].isna().mean():.1%}")

    # ── Feature columns ───────────────────────────────────────────────────────
    base_cols     = [c for c in df.columns if c not in DROP_ALWAYS + new_feats]
    enhanced_cols = base_cols + new_feats

    print(f"\n   Baseline features : {len(base_cols)}")
    print(f"   Enhanced features : {len(enhanced_cols)}")

    # ── Train baseline (same features as train_multi_horizon) ─────────────────
    print("\n── Training baseline (label_30d, no new features) ──────────────")
    model_base, roc_base, ap_base, rep_base, imp_base = train_and_eval(
        df, base_cols, label=f"baseline — {len(base_cols)} features"
    )
    print_top_features(imp_base, n=15)

    # ── Train enhanced ────────────────────────────────────────────────────────
    print("\n── Training enhanced (label_30d + 30d-specific features) ───────")
    model_enh, roc_enh, ap_enh, rep_enh, imp_enh = train_and_eval(
        df, enhanced_cols, label=f"enhanced — {len(enhanced_cols)} features"
    )
    print_top_features(imp_enh, n=15, new_feats=new_feats)

    # ── Ranks of new features ─────────────────────────────────────────────────
    print("\n── New feature ranks in enhanced model ─────────────────────────")
    for f in new_feats:
        if f in imp_enh.index:
            rank = list(imp_enh.index).index(f) + 1
            val  = imp_enh[f]
            bar  = "█" * int(val / imp_enh.max() * 30)
            print(f"   #{rank:<3} {f:<30} {val:>6.0f}  {bar}")

    # ── Delta ─────────────────────────────────────────────────────────────────
    d_roc  = roc_enh - roc_base
    d_ap   = ap_enh  - ap_base
    print(f"\n── Results ─────────────────────────────────────────────────────")
    print(f"   Baseline  (train_multi_horizon)  ROC-AUC = {BASELINE_AUC:.4f}")
    print(f"   Baseline  (this run)             ROC-AUC = {roc_base:.4f}")
    print(f"   Enhanced  (+30d features)         ROC-AUC = {roc_enh:.4f}  "
          f"{'▲' if d_roc > 0 else '▼'} {abs(d_roc):.4f}")
    print(f"   Avg Precision  base={ap_base:.4f}  enh={ap_enh:.4f}  "
          f"{'▲' if d_ap > 0 else '▼'} {abs(d_ap):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_enh.booster_.save_model(str(MODEL_OUT))
    REPORT_OUT.write_text(
        f"Target         : {TARGET}\n"
        f"Baseline AUC   : {BASELINE_AUC:.4f}\n"
        f"Enhanced AUC   : {roc_enh:.4f}  ({'▲' if d_roc>0 else '▼'}{abs(d_roc):.4f})\n"
        f"Avg Precision  : base={ap_base:.4f}  enh={ap_enh:.4f}\n\n"
        f"New features   : {new_feats}\n\n"
        f"{rep_enh}\n\n"
        f"Top 20 features:\n{imp_enh.head(20).to_string()}\n"
    )
    print(f"\n   Model  → {MODEL_OUT}")
    print(f"   Report → {REPORT_OUT}")


if __name__ == "__main__":
    main()
