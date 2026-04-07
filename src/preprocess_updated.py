import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0
FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/"
    "gem-global-active-faults/master/geojson/gem_active_faults.geojson"
)
COASTLINE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_coastline.geojson"
)


def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)


def chord_to_km(chord):
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0, 1))


def build_tree_from_geojson(url, label):
    """Télécharge un GeoJSON, extrait tous les points, retourne un cKDTree 3D."""
    print(f"Téléchargement : {label}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    geojson = resp.json()
    print(f"  {len(geojson['features'])} features chargées")

    coords = []
    for feature in geojson["features"]:
        geom = feature.get("geometry")
        if geom is None:
            continue
        gtype = geom["type"]
        if gtype == "LineString":
            coords.extend(geom["coordinates"])
        elif gtype == "MultiLineString":
            for line in geom["coordinates"]:
                coords.extend(line)
        elif gtype == "Polygon":
            for ring in geom["coordinates"]:
                coords.extend(ring)
        elif gtype == "MultiPolygon":
            for polygon in geom["coordinates"]:
                for ring in polygon:
                    coords.extend(ring)

    arr = np.array([c[:2] for c in coords])  # [lon, lat], ignore altitude
    print(f"  {len(arr):,} points extraits")
    x, y, z = latlon_to_xyz(arr[:, 1], arr[:, 0])
    return cKDTree(np.column_stack([x, y, z]))


def dist_km_to_tree(tree, lats, lons):
    eq_xyz = np.column_stack(latlon_to_xyz(lats, lons))
    chord, _ = tree.query(eq_xyz, workers=-1)
    return np.round(chord_to_km(chord), 2)


# ── 1. Nettoyage de base ──────────────────────────────────────────────────────

df = pd.read_csv("data/database_updated.csv")
shape_before = df.shape
print(f"Shape avant nettoyage : {shape_before}")

threshold = len(df) * 0.5
df = df.dropna(axis=1, thresh=int(threshold))
df = df[["Date", "Latitude", "Longitude", "Depth", "Magnitude", "Type"]]
df = df.dropna()
df = df[df["Type"].str.lower() == "earthquake"].drop(columns=["Type"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%m/%d/%Y")
df = df.dropna(subset=["Date"])
df["Dangerous"] = (df["Magnitude"] >= 6.0).astype(int)
df = df.reset_index(drop=True)
print(f"Shape après nettoyage de base : {df.shape}")


# ── 2. Feature : dist_fault_km ────────────────────────────────────────────────

fault_tree = build_tree_from_geojson(FAULTS_URL, "failles actives GEM")
print("Calcul dist_fault_km...")
df["dist_fault_km"] = dist_km_to_tree(fault_tree, df["Latitude"].values, df["Longitude"].values)

print(f"  Moyenne={df['dist_fault_km'].mean():.1f} km  "
      f"Médiane={df['dist_fault_km'].median():.1f} km  "
      f"Min={df['dist_fault_km'].min():.1f}  Max={df['dist_fault_km'].max():.1f} km")


# ── 3. Feature : IsCoastal ────────────────────────────────────────────────────

coast_tree = build_tree_from_geojson(COASTLINE_URL, "côtes Natural Earth 50m")
print("Calcul IsCoastal (seuil 100 km)...")
dist_coast = dist_km_to_tree(coast_tree, df["Latitude"].values, df["Longitude"].values)
df["IsCoastal"] = (dist_coast < 100).astype(int)

n_coastal = df["IsCoastal"].sum()
print(f"  Séismes côtiers (<100 km) : {n_coastal:,} ({n_coastal/len(df)*100:.1f}%)")


# ── 4. Feature : Season ───────────────────────────────────────────────────────

month = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.month
season_map = {
    12: "Hiver", 1: "Hiver",  2: "Hiver",
     3: "Printemps", 4: "Printemps", 5: "Printemps",
     6: "Été",  7: "Été",  8: "Été",
     9: "Automne", 10: "Automne", 11: "Automne",
}
df["Season"] = month.map(season_map)
print("\n=== Distribution Season ===")
print(df["Season"].value_counts().to_string())


# ── 5. Feature : Depth_category ──────────────────────────────────────────────

def depth_category(d):
    if d < 70:
        return "superficiel"
    elif d <= 300:
        return "intermédiaire"
    else:
        return "profond"

df["Depth_category"] = df["Depth"].apply(depth_category)
print("\n=== Distribution Depth_category ===")
print(df["Depth_category"].value_counts().to_string())


# ── 6. Résumé et sauvegarde ───────────────────────────────────────────────────

print("\n=== Distribution de 'Dangerous' ===")
counts = df["Dangerous"].value_counts()
print(f"  Non dangereux (0) : {counts[0]:>7,}  ({counts[0]/len(df)*100:.1f}%)")
print(f"  Dangereux     (1) : {counts[1]:>7,}  ({counts[1]/len(df)*100:.1f}%)")

df.to_csv("data/clean_updated.csv", index=False)

print(f"\n=== Résumé final ===")
print(f"Shape avant  : {shape_before}")
print(f"Shape après  : {df.shape}")
print(f"Colonnes     : {list(df.columns)}")
print(f"Période      : {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
print("\nDataset sauvegardé dans data/clean_updated.csv")
