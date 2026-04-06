Analyse le dataset du projet séismes USGS.

1. Charge `data/clean_database.csv` avec pandas
2. Affiche la shape, les types de colonnes et les statistiques descriptives complètes
3. Affiche la distribution de la colonne `Dangerous` (0 vs 1) avec les pourcentages
4. Affiche les corrélations entre `Latitude`, `Longitude`, `Depth` et `Magnitude`
5. Identifie les outliers sur `Depth` (valeurs > 3 écarts-types)
6. Génère et sauvegarde dans `outputs/` :
   - Un heatmap de corrélation (`correlation_heatmap.png`)
   - Un scatter plot Depth vs Magnitude coloré par Dangerous (`depth_vs_magnitude.png`)
7. Résume les insights clés pour le modeling (features les plus corrélées, déséquilibre de classes, plage des valeurs)
