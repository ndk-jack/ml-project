import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# --- Chargement ---
df = pd.read_csv("data/clean_updated.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df = df.dropna(subset=["Year", "Month"])

# Encodage ordinal de Depth_category (ordre naturel : superficiel < intermédiaire < profond)
depth_enc = OrdinalEncoder(categories=[["superficiel", "intermédiaire", "profond"]])
df["Depth_category_enc"] = depth_enc.fit_transform(df[["Depth_category"]])

# Encodage one-hot de Season (pas d'ordre naturel)
df = pd.get_dummies(df, columns=["Season"], prefix="season", dtype=int)
season_cols = [c for c in df.columns if c.startswith("season_")]

FEATURES = [
    "Latitude", "Longitude", "Depth", "Year", "Month",
    "dist_fault_km", "IsCoastal", "Depth_category_enc",
] + season_cols

X    = df[FEATURES]
y_reg = df["Magnitude"]
y_clf = df["Dangerous"]

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)
print(f"Dataset     : {len(df):,} samples")
print(f"Train / Test: {len(X_train):,} / {len(X_test):,}")
print(f"Features ({len(FEATURES)}) : {FEATURES}\n")


# ── RÉGRESSION ───────────────────────────────────────────────────────────────

def eval_regressor(name, model):
    model.fit(X_train, yr_train)
    pred = model.predict(X_test)
    return {
        "Modèle": name,
        "RMSE": round(np.sqrt(mean_squared_error(yr_test, pred)), 4),
        "R²":   round(r2_score(yr_test, pred), 4),
    }, model, pred

reg_results, reg_models = [], {}

for name, model in [
    ("LinearRegression",            LinearRegression()),
    ("RandomForestRegressor",       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ("HistGradientBoostingReg",     HistGradientBoostingRegressor(max_iter=100, random_state=42)),
]:
    print(f"Entraînement : {name}...")
    metrics, fitted, preds = eval_regressor(name, model)
    reg_results.append(metrics)
    reg_models[name] = (fitted, preds)

df_reg = pd.DataFrame(reg_results).sort_values("RMSE")
print(f"\n{'─'*52}")
print("TABLEAU RÉGRESSION (trié par RMSE ↑)")
print(df_reg.to_string(index=False))

best_reg_name = df_reg.iloc[0]["Modèle"]
best_reg_model, best_reg_preds = reg_models[best_reg_name]
print(f"\nMeilleur régresseur : {best_reg_name}")
joblib.dump(best_reg_model, "outputs/best_regressor.joblib")


# ── CLASSIFICATION — GradientBoosting ────────────────────────────────────────

print(f"\n{'─'*52}")
print("Entraînement : GradientBoostingClassifier...")
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, yc_train)
y_proba = clf.predict_proba(X_test)[:, 1]

# --- Threshold Tuning ---
thresholds = np.arange(0.05, 0.51, 0.05)
thresh_rows = []
for t in thresholds:
    preds = (y_proba >= t).astype(int)
    thresh_rows.append({
        "Seuil":     round(t, 2),
        "Precision": round(precision_score(yc_test, preds, zero_division=0), 4),
        "Recall":    round(recall_score(yc_test, preds, zero_division=0), 4),
        "F1":        round(f1_score(yc_test, preds, zero_division=0), 4),
    })

df_thresh = pd.DataFrame(thresh_rows)
print(f"\n{'─'*52}")
print("TABLEAU THRESHOLD TUNING — GradientBoostingClassifier")
print(df_thresh.to_string(index=False))

best_thresh_row = df_thresh.loc[df_thresh["F1"].idxmax()]
best_threshold  = best_thresh_row["Seuil"]
print(f"\nSeuil optimal : {best_threshold}  "
      f"(Precision={best_thresh_row['Precision']}, "
      f"Recall={best_thresh_row['Recall']}, "
      f"F1={best_thresh_row['F1']})")

with open("outputs/best_threshold.txt", "w") as f:
    f.write(f"{best_threshold}\n")
print("Seuil sauvegardé dans outputs/best_threshold.txt")

joblib.dump(clf, "outputs/best_classifier.joblib")

# Prédictions finales avec seuil optimal
yc_pred_best = (y_proba >= best_threshold).astype(int)


# ── TABLEAU COMPARATIF FINAL ──────────────────────────────────────────────────

print(f"\n{'═'*60}")
print("TABLEAU COMPARATIF FINAL")
print(f"{'═'*60}")

summary = []
for row in reg_results:
    summary.append({
        "Modèle":  row["Modèle"],
        "Tâche":   "Régression",
        "Métrique": f"RMSE={row['RMSE']}  R²={row['R²']}",
    })
for t_row in thresh_rows:
    if t_row["Seuil"] == best_threshold:
        summary.append({
            "Modèle":  f"GradientBoosting (seuil={best_threshold})",
            "Tâche":   "Classification",
            "Métrique": f"P={t_row['Precision']}  R={t_row['Recall']}  F1={t_row['F1']}",
        })

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))


# ── PLOTS ─────────────────────────────────────────────────────────────────────

# Réel vs Prédit — meilleur régresseur
best_rmse = df_reg.iloc[0]["RMSE"]
best_r2   = df_reg.iloc[0]["R²"]
plt.figure(figsize=(6, 5))
plt.scatter(yr_test, best_reg_preds, alpha=0.2, s=5, color="steelblue")
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "r--", lw=1.5)
plt.xlabel("Magnitude réelle")
plt.ylabel("Magnitude prédite")
plt.title(f"{best_reg_name}  |  RMSE={best_rmse}  R²={best_r2}")
plt.tight_layout()
plt.savefig("outputs/regression_predictions.png", dpi=150)

# Matrice de confusion avec seuil optimal
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(yc_test, yc_pred_best),
    display_labels=["Non dangereux", "Dangereux"]
).plot(ax=ax, colorbar=False)
ax.set_title(f"GradientBoosting  |  seuil={best_threshold}  F1={best_thresh_row['F1']}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)

# Precision / Recall / F1 vs seuil
plt.figure(figsize=(8, 4))
plt.plot(df_thresh["Seuil"], df_thresh["Precision"], marker="o", label="Precision")
plt.plot(df_thresh["Seuil"], df_thresh["Recall"],    marker="s", label="Recall")
plt.plot(df_thresh["Seuil"], df_thresh["F1"],        marker="^", label="F1", lw=2)
plt.axvline(best_threshold, color="red", linestyle="--", label=f"Seuil optimal ({best_threshold})")
plt.xlabel("Seuil de décision")
plt.ylabel("Score")
plt.title("Threshold Tuning — GradientBoostingClassifier")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/threshold_tuning.png", dpi=150)

# Comparaison régression (barplot)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(df_reg["Modèle"], df_reg["RMSE"], color="steelblue")
axes[0].set_xlabel("RMSE (↓ meilleur)")
axes[0].set_title("Comparaison Régression — RMSE")
axes[0].invert_xaxis()
axes[1].barh(df_reg["Modèle"], df_reg["R²"], color="steelblue")
axes[1].set_xlabel("R² (↑ meilleur)")
axes[1].set_title("Comparaison Régression — R²")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)

print("\nModèles sauvegardés : outputs/best_regressor.joblib, outputs/best_classifier.joblib")
print("Seuil     sauvegardé : outputs/best_threshold.txt")
print("Graphiques sauvegardés dans outputs/")
