from pathlib import Path
import pandas as pd

DATASET_PATH = Path("/Users/nazlidecker/ml-project/data/features/dataset_v3.csv")

def main():
    df = pd.read_csv(DATASET_PATH, nrows=5000, low_memory=False)

    print("dataset_path", DATASET_PATH)
    print("rows_sample", len(df))
    print("column_count", len(df.columns))

    print("\npossible_target_columns")
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ["target", "label", "follow", "y_", "_7d", "_30d", "_365d"]):
            print(c)

    print("\npossible_time_columns")
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ["time", "date", "datetime", "event_"]):
            print(c)

    print("\nnull_rate_top20")
    nulls = df.isna().mean().sort_values(ascending=False).head(20)
    for k, v in nulls.items():
        print(f"{k}: {v:.4f}")

    print("\nbinary_like_columns")
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) > 0 and len(vals) <= 5:
            s = set(vals.tolist())
            if s.issubset({0, 1, 0.0, 1.0, True, False}):
                print(c, sorted(s))

if __name__ == "__main__":
    main()
