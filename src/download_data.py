import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("usgs/earthquake-database")
print("Downloaded to:", path)

# Copier dans notre dossier data/
shutil.copy(path + "/database.csv", "data/database.csv")
print("✅ Fichier copié dans data/database.csv")
