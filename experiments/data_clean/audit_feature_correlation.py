from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "features_clean" / "dataset_v5_dedup.csv"
EXCLUDE = {"datetime", "label_7d", "label_30d"}

def main():
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print("dataset_path", DATASET_PATH)

    feature_cols = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    sample = df[feature_cols].sample(min(20000, len(df)), random_state=42)
    corr = sample.corr(numeric_only=True).abs()

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and v >= 0.95:
                pairs.append((cols[i], cols[j], float(v)))

    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    print("high_corr_pairs_ge_0.95")
    for a, b, v in pairs[:100]:
        print(f"{a} | {b} | {v:.6f}")

if __name__ == "__main__":
    main()
