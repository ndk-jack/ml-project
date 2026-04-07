import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import geopandas as gpd

COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)

# Dictionnaire de noms pour le top 15 zones (lat_bin, lon_bin) → nom
ZONE_NAMES = {
    ( 0,  125): "Sulawesi, Indonésie",
    (35,  140): "Honshu, Japon",
    (-25, -180): "Tonga",
    (-20, -180): "Fidji-Tonga",
    (-10,  150): "Papouasie-Nvl-Guinée",
    (  5,  125): "Mindanao, Philippines",
    ( 35,   70): "Hindu Kush, Afghanistan",
    (-25,  -70): "Atacama, Chili",
    (-35, -180): "Kermadec, NZ",
    (-15,  165): "Vanuatu",
    (-10,  125): "Timor, Indonésie",
    (-20, -175): "Tonga Sud",
    (-35,  -75): "Biobío, Chili",
    ( 20,  120): "Taïwan",
    (-20,  165): "Nouvelle-Calédonie",
}

df = pd.read_csv("data/clean_updated.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

dangerous = df[df["Dangerous"] == 1]
moderate  = df[df["Dangerous"] == 0]

print(f"Dataset chargé : {len(df):,} séismes")


# ── 1. Carte mondiale ─────────────────────────────────────────────────────────

print("Génération world_map.png...")
print("  Chargement des frontières pays...")
world = gpd.read_file(COUNTRIES_URL)

fig, ax = plt.subplots(figsize=(18, 9), facecolor="#1a1a2e")
ax.set_facecolor("#16213e")
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_aspect("equal")

# Frontières des pays
world.boundary.plot(ax=ax, linewidth=0.3, color="white", alpha=0.6)

# Grille subtile
ax.grid(color="#2a2a4a", linewidth=0.4, linestyle="--", alpha=0.5)
ax.set_xticks(range(-180, 181, 30))
ax.set_yticks(range(-90, 91, 30))
ax.tick_params(colors="#aaaacc", labelsize=7)

# Séismes modérés
size_mod = ((moderate["Magnitude"] - moderate["Magnitude"].min()) /
            (moderate["Magnitude"].max() - moderate["Magnitude"].min()) * 8 + 1)
ax.scatter(moderate["Longitude"], moderate["Latitude"],
           s=size_mod, c="#4a90d9", alpha=0.25, linewidths=0, rasterized=True,
           label="Modéré (< 6.0)")

# Séismes dangereux
size_dan = ((dangerous["Magnitude"] - dangerous["Magnitude"].min()) /
            (dangerous["Magnitude"].max() - dangerous["Magnitude"].min()) * 30 + 4)
ax.scatter(dangerous["Longitude"], dangerous["Latitude"],
           s=size_dan, c="#ff4444", alpha=0.6, linewidths=0, rasterized=True,
           label="Dangereux (≥ 6.0)")

# Noms des continents
continent_labels = [
    ("Amérique\ndu Nord",  -100,  45),
    ("Amérique\ndu Sud",    -60, -15),
    ("Europe",               15,  52),
    ("Afrique",              20,   5),
    ("Asie",                 90,  45),
    ("Océanie",             135, -25),
    ("Antarctique",           0, -80),
]
for name, lon, lat in continent_labels:
    ax.text(lon, lat, name, color="white", fontsize=7.5, alpha=0.65,
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", alpha=0.3, linewidth=0))

ax.set_title("540 000 séismes mondiaux 1900–2026", color="white", fontsize=16, pad=12)
ax.set_xlabel("Longitude", color="#aaaacc", fontsize=9)
ax.set_ylabel("Latitude", color="#aaaacc", fontsize=9)
ax.spines[:].set_color("#2a2a4a")

for mag, lbl in [(5.0, "M 5.0"), (7.0, "M 7.0"), (9.0, "M 9.0")]:
    ax.scatter([], [], s=(mag - 4) * 5, c="white", alpha=0.6, label=lbl, linewidths=0)
ax.legend(loc="lower left", fontsize=8, framealpha=0.4,
          labelcolor="white", facecolor="#1a1a2e", edgecolor="#444466",
          title="Magnitude", title_fontsize=8)

plt.tight_layout()
plt.savefig("outputs/world_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ world_map.png")


# ── 2. Évolution par décennie ─────────────────────────────────────────────────

print("Génération decades.png...")
df["Decade"] = (df["Year"] // 10 * 10).astype("Int64")
decade_counts = df.groupby("Decade").size().reset_index(name="Count")
decade_counts = decade_counts.dropna(subset=["Decade"])
decade_counts["Decade"] = decade_counts["Decade"].astype(int)

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#c0392b" if d >= 1960 else "#2980b9" for d in decade_counts["Decade"]]
bars = ax.bar(decade_counts["Decade"].astype(str), decade_counts["Count"],
              color=colors, edgecolor="white", linewidth=0.5, width=0.7)

# Annotation tournant 1960
idx_1960 = decade_counts[decade_counts["Decade"] == 1960].index
if len(idx_1960):
    pos = list(decade_counts["Decade"]).index(1960)
    ax.annotate(
        "Meilleure détection\naprès 1960\n(réseau sismique mondial)",
        xy=(pos, decade_counts.loc[idx_1960[0], "Count"]),
        xytext=(pos + 1.5, decade_counts["Count"].max() * 0.85),
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
        color="#e74c3c", fontsize=9, ha="left",
    )

# Valeurs sur les barres
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 500,
            f"{h:,.0f}", ha="center", va="bottom", fontsize=7, color="#555")

ax.set_title("Nombre de séismes enregistrés par décennie (M ≥ 4.0)", fontsize=13)
ax.set_xlabel("Décennie")
ax.set_ylabel("Nombre de séismes")
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
legend_patches = [
    mpatches.Patch(color="#2980b9", label="Avant 1960 (détection incomplète)"),
    mpatches.Patch(color="#c0392b", label="Après 1960 (réseau mondial)"),
]
ax.legend(handles=legend_patches, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/decades.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ decades.png")


# ── 3. Top 15 zones sismiques ─────────────────────────────────────────────────

print("Génération top_zones.png...")
df["lat_bin"] = (df["Latitude"]  // 5 * 5).astype(int)
df["lon_bin"] = (df["Longitude"] // 5 * 5).astype(int)

zone_counts = (df.groupby(["lat_bin", "lon_bin"])
                 .agg(count=("Magnitude", "size"),
                      mean_mag=("Magnitude", "mean"))
                 .reset_index()
                 .sort_values("count", ascending=False)
                 .head(15))

def zone_label(row):
    key = (int(row.lat_bin), int(row.lon_bin))
    if key in ZONE_NAMES:
        return ZONE_NAMES[key]
    lat = f"{abs(int(row.lat_bin))}°{'N' if row.lat_bin >= 0 else 'S'}"
    lon = f"{abs(int(row.lon_bin))}°{'E' if row.lon_bin >= 0 else 'W'}"
    return f"{lat} / {lon}"

zone_counts["label"] = zone_counts.apply(zone_label, axis=1)
zone_counts = zone_counts.sort_values("count")

fig, ax = plt.subplots(figsize=(10, 7))
cmap_vals = (zone_counts["mean_mag"] - zone_counts["mean_mag"].min()) / \
            (zone_counts["mean_mag"].max() - zone_counts["mean_mag"].min())
colors = plt.cm.YlOrRd(cmap_vals.values)

bars = ax.barh(zone_counts["label"], zone_counts["count"],
               color=colors, edgecolor="white", linewidth=0.4)

for bar, mag in zip(bars, zone_counts["mean_mag"]):
    w = bar.get_width()
    ax.text(w + 100, bar.get_y() + bar.get_height() / 2,
            f"M̄ {mag:.2f}", va="center", fontsize=8, color="#333")

sm = plt.cm.ScalarMappable(cmap="YlOrRd",
     norm=plt.Normalize(zone_counts["mean_mag"].min(), zone_counts["mean_mag"].max()))
sm.set_array([])
cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("Magnitude moyenne", fontsize=9)

ax.set_title("Top 15 zones sismiques — grille 5°×5°", fontsize=13)
ax.set_xlabel("Nombre de séismes")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("outputs/top_zones.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ top_zones.png")


# ── 4. Dashboard modèles ──────────────────────────────────────────────────────

print("Génération dashboard.png...")
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── 4a. R² régression ──
ax1 = fig.add_subplot(gs[0, 0])
models_reg = ["LinearRegression", "HistGradientBoosting", "RandomForest"]
r2_values   = [0.191, 0.361, 0.366]
colors_reg  = ["#95a5a6", "#3498db", "#2ecc71"]
bars = ax1.barh(models_reg, r2_values, color=colors_reg, edgecolor="white")
for bar, val in zip(bars, r2_values):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9)
ax1.set_xlim(0, 0.45)
ax1.axvline(0.366, color="#2ecc71", linestyle="--", alpha=0.4)
ax1.set_title("R² — Régression magnitude", fontsize=11)
ax1.set_xlabel("R²  (↑ meilleur)")

# ── 4b. F1 classification ──
ax2 = fig.add_subplot(gs[0, 1])
models_clf = ["LogisticReg\n(seuil=0.50)", "GradientBoosting\n(seuil=0.50)",
              "GradientBoosting\n(seuil=0.25)"]
f1_values  = [0.149, 0.420, 0.480]
colors_clf = ["#95a5a6", "#3498db", "#e74c3c"]
bars = ax2.barh(models_clf, f1_values, color=colors_clf, edgecolor="white")
for bar, val in zip(bars, f1_values):
    ax2.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9)
ax2.set_xlim(0, 0.60)
ax2.axvline(0.480, color="#e74c3c", linestyle="--", alpha=0.4)
ax2.set_title("F1-score — Classification (Dangereux)", fontsize=11)
ax2.set_xlabel("F1-score  (↑ meilleur)")

# ── 4c. Distribution magnitudes ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df["Magnitude"], bins=60, color="#3498db", edgecolor="white",
         linewidth=0.3, density=True)
ax3.axvline(6.0, color="#e74c3c", linestyle="--", lw=1.5, label="Seuil danger (6.0)")
ax3.axvline(df["Magnitude"].mean(), color="#f39c12", linestyle=":", lw=1.5,
            label=f"Moyenne ({df['Magnitude'].mean():.2f})")
ax3.set_title("Distribution des magnitudes", fontsize=11)
ax3.set_xlabel("Magnitude")
ax3.set_ylabel("Densité")
ax3.legend(fontsize=8)

# ── 4d. Distribution dist_fault_km ──
ax4 = fig.add_subplot(gs[1, 1])
clip = df["dist_fault_km"].clip(upper=500)
ax4.hist(clip, bins=60, color="#9b59b6", edgecolor="white", linewidth=0.3)
ax4.axvline(100, color="#e74c3c", linestyle="--", lw=1.5, label="100 km (IsCoastal)")
ax4.axvline(df["dist_fault_km"].median(), color="#f39c12", linestyle=":", lw=1.5,
            label=f"Médiane ({df['dist_fault_km'].median():.0f} km)")
ax4.set_title("Distance à la faille la plus proche\n(tronquée à 500 km)", fontsize=11)
ax4.set_xlabel("Distance (km)")
ax4.set_ylabel("Nombre de séismes")
ax4.legend(fontsize=8)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

fig.suptitle("Dashboard — Earthquake ML Project", fontsize=15, fontweight="bold", y=1.01)
plt.savefig("outputs/dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ dashboard.png")

print("\nTous les graphiques sauvegardés dans outputs/")
