import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("data/database.csv")

print("=== 5 premières lignes ===")
print(df.head())

print("\n=== Shape & colonnes ===")
print(f"Shape : {df.shape}")
print(f"Colonnes : {list(df.columns)}")
print("\nTypes :")
print(df.dtypes)

print("\n=== Valeurs manquantes ===")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

print("\n=== Statistiques sur la magnitude ===")
mag = df["Magnitude"]
print(f"Mean  : {mag.mean():.3f}")
print(f"Min   : {mag.min():.3f}")
print(f"Max   : {mag.max():.3f}")
print(f"Std   : {mag.std():.3f}")

plt.figure(figsize=(10, 5))
plt.hist(mag.dropna(), bins=50, color="steelblue", edgecolor="white")
plt.title("Distribution des magnitudes (USGS 1965–2016)")
plt.xlabel("Magnitude")
plt.ylabel("Nombre de séismes")
plt.tight_layout()
plt.savefig("outputs/magnitude_distribution.png", dpi=150)
print("\nHistogramme sauvegardé dans outputs/magnitude_distribution.png")
