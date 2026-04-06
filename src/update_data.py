import requests
import pandas as pd
import time
from datetime import date
from io import StringIO

API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
MIN_MAGNITUDE = 4.0
START_YEAR = 1900
END_YEAR = date.today().year
MAX_RETRIES = 3
RETRY_DELAY = 5  # secondes


def fetch_year(year):
    """Télécharge les séismes pour une année donnée avec retry automatique."""
    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime": f"{year}-12-31" if year < END_YEAR else str(date.today()),
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time-asc",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            return df
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                print(f"  Erreur (tentative {attempt}/{MAX_RETRIES}) : {e}. Retry dans {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Echec après {MAX_RETRIES} tentatives pour {year} : {e}")
                return pd.DataFrame()
        except Exception as e:
            print(f"  Erreur inattendue pour {year} : {e}")
            return pd.DataFrame()


def normalize_columns(df):
    """Renomme les colonnes de l'API USGS vers un format cohérent."""
    rename = {
        "time": "Date",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "depth": "Depth",
        "mag": "Magnitude",
        "magType": "Magnitude Type",
        "type": "Type",
        "id": "ID",
        "status": "Status",
        "locationSource": "Location Source",
        "magSource": "Magnitude Source",
        "horizontalError": "Horizontal Error",
        "depError": "Depth Error",
        "magError": "Magnitude Error",
        "magNst": "Magnitude Seismic Stations",
        "nst": "Depth Seismic Stations",
        "gap": "Azimuthal Gap",
        "dmin": "Horizontal Distance",
        "rms": "Root Mean Square",
        "net": "Source",
    }
    df = df.rename(columns=rename)
    # Sépare Date et Time
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df["Time"] = dt.dt.strftime("%H:%M:%S")
        df["Date"] = dt.dt.strftime("%m/%d/%Y")
    return df


# --- Téléchargement par tranche annuelle ---
all_frames = []
total_years = END_YEAR - START_YEAR + 1

print(f"Téléchargement des séismes M >= {MIN_MAGNITUDE} de {START_YEAR} à {END_YEAR}...\n")

for year in range(START_YEAR, END_YEAR + 1):
    df_year = fetch_year(year)
    n = len(df_year)
    print(f"  Année {year}... ({n} séismes)")
    if n > 0:
        all_frames.append(df_year)

# --- Combinaison et nettoyage ---
print("\nCombination des données...")
df_all = pd.concat(all_frames, ignore_index=True)
df_all = normalize_columns(df_all)

before_dedup = len(df_all)
df_all = df_all.drop_duplicates(subset=["ID"])
after_dedup = len(df_all)
print(f"Doublons supprimés : {before_dedup - after_dedup}")

# --- Sauvegarde ---
output_path = "data/database_updated.csv"
df_all.to_csv(output_path, index=False)

# --- Résumé final ---
print(f"\n{'='*45}")
print(f"RÉSUMÉ FINAL")
print(f"{'='*45}")
print(f"Total séismes      : {len(df_all):,}")
print(f"Période couverte   : {df_all['Date'].iloc[0]} → {df_all['Date'].iloc[-1]}")
print(f"Magnitude min      : {df_all['Magnitude'].min()}")
print(f"Magnitude max      : {df_all['Magnitude'].max()}")
print(f"Fichier sauvegardé : {output_path}")
