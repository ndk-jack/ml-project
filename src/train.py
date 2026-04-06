import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# --- Chargement ---
df = pd.read_csv("data/clean_updated.csv")

# Feature engineering : Year et Month depuis la colonne Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df = df.dropna(subset=["Year", "Month"])

FEATURES = ["Latitude", "Longitude", "Depth", "Year", "Month"]
X = df[FEATURES]
y_reg = df["Magnitude"]
y_clf = df["Dangerous"]

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)
print(f"Dataset     : {len(df):,} samples")
print(f"Train / Test: {len(X_train):,} / {len(X_test):,}")
print(f"Features    : {FEATURES}\n")


# ── RÉGRESSION ──────────────────────────────────────────────────────────────

def eval_regressor(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "Modèle": name,
        "RMSE": round(np.sqrt(mean_squared_error(y_te, pred)), 4),
        "R²": round(r2_score(y_te, pred), 4),
    }, model, pred

reg_results = []
reg_models = {}

for name, model in [
    ("LinearRegression",       LinearRegression()),
    ("RandomForestRegressor",  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
]:
    metrics, fitted, preds = eval_regressor(name, model, X_train, yr_train, X_test, yr_test)
    reg_results.append(metrics)
    reg_models[name] = (fitted, preds)
    print(f"[Régression] {name:30s}  RMSE={metrics['RMSE']}  R²={metrics['R²']}")

df_reg = pd.DataFrame(reg_results).sort_values("RMSE")
print(f"\n{'─'*50}")
print("TABLEAU RÉGRESSION")
print(df_reg.to_string(index=False))

best_reg_name = df_reg.iloc[0]["Modèle"]
best_reg_model, best_reg_preds = reg_models[best_reg_name]
print(f"\nMeilleur modèle régression : {best_reg_name}")
joblib.dump(best_reg_model, "outputs/best_regressor.joblib")


# ── CLASSIFICATION ───────────────────────────────────────────────────────────

def eval_classifier(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "Modèle": name,
        "Accuracy": round(accuracy_score(y_te, pred), 4),
        "F1-score": round(f1_score(y_te, pred), 4),
    }, model, pred

clf_results = []
clf_models = {}

print()
for name, model in [
    ("LogisticRegression",        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ("RandomForestClassifier",    RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
    ("GradientBoostingClassifier",GradientBoostingClassifier(n_estimators=100, random_state=42)),
]:
    metrics, fitted, preds = eval_classifier(name, model, X_train, yc_train, X_test, yc_test)
    clf_results.append(metrics)
    clf_models[name] = (fitted, preds)
    print(f"[Classification] {name:32s}  Acc={metrics['Accuracy']}  F1={metrics['F1-score']}")

df_clf = pd.DataFrame(clf_results).sort_values("F1-score", ascending=False)
print(f"\n{'─'*55}")
print("TABLEAU CLASSIFICATION")
print(df_clf.to_string(index=False))

best_clf_name = df_clf.iloc[0]["Modèle"]
best_clf_model, best_clf_preds = clf_models[best_clf_name]
print(f"\nMeilleur modèle classification : {best_clf_name}")
joblib.dump(best_clf_model, "outputs/best_classifier.joblib")


# ── PLOTS ─────────────────────────────────────────────────────────────────────

# Réel vs Prédit — meilleur régresseur
best_rmse = df_reg.iloc[0]["RMSE"]
best_r2   = df_reg.iloc[0]["R²"]
plt.figure(figsize=(6, 5))
plt.scatter(yr_test, best_reg_preds, alpha=0.3, s=8, color="steelblue")
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "r--", lw=1.5)
plt.xlabel("Magnitude réelle")
plt.ylabel("Magnitude prédite")
plt.title(f"{best_reg_name}\nRMSE={best_rmse}  R²={best_r2}")
plt.tight_layout()
plt.savefig("outputs/regression_predictions.png", dpi=150)

# Matrice de confusion — meilleur classifieur
best_f1 = df_clf.iloc[0]["F1-score"]
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(yc_test, best_clf_preds),
    display_labels=["Non dangereux", "Dangereux"]
).plot(ax=ax, colorbar=False)
ax.set_title(f"{best_clf_name}\nF1={best_f1}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)

# Comparaison des modèles (barplot)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].barh(df_reg["Modèle"], df_reg["RMSE"], color="steelblue")
axes[0].set_xlabel("RMSE (↓ meilleur)")
axes[0].set_title("Comparaison Régression")
axes[0].invert_xaxis()

axes[1].barh(df_clf["Modèle"], df_clf["F1-score"], color="seagreen")
axes[1].set_xlabel("F1-score (↑ meilleur)")
axes[1].set_title("Comparaison Classification")

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)

print("\nModèles sauvegardés : outputs/best_regressor.joblib, outputs/best_classifier.joblib")
print("Graphiques sauvegardés dans outputs/")
