import pandas as pd

df = pd.read_csv("data/database.csv")
shape_before = df.shape
print(f"Shape avant nettoyage : {shape_before}")

# Supprimer les colonnes avec > 50% de NaN
threshold = len(df) * 0.5
df = df.dropna(axis=1, thresh=int(threshold))

# Garder uniquement les colonnes utiles
cols = ["Date", "Latitude", "Longitude", "Depth", "Magnitude", "Type"]
df = df[cols]

# Supprimer les lignes restantes avec NaN
df = df.dropna()

# Filtrer uniquement les séismes (exclure explosions nucléaires, etc.)
df = df[df["Type"] == "Earthquake"].drop(columns=["Type"])

# Colonne cible classification
df["Dangerous"] = (df["Magnitude"] >= 6.0).astype(int)

print("\n=== Distribution de 'Dangerous' ===")
counts = df["Dangerous"].value_counts()
print(f"  Non dangereux (0) : {counts[0]:>6}  ({counts[0]/len(df)*100:.1f}%)")
print(f"  Dangereux     (1) : {counts[1]:>6}  ({counts[1]/len(df)*100:.1f}%)")

df.to_csv("data/clean_database.csv", index=False)

print(f"\n=== Résumé ===")
print(f"Shape avant  : {shape_before}")
print(f"Shape après  : {df.shape}")
print(f"Colonnes     : {list(df.columns)}")
print("\nDataset sauvegardé dans data/clean_database.csv")
