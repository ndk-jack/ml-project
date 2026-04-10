from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "features_clean" / "dataset_v5_dedup.csv"
TIME_COL = "datetime"
TARGETS = ["label_7d", "label_30d"]

def main():
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("dataset_path", DATASET_PATH)
    print("rows", len(df))
    print("column_count", len(df.columns))

    print("\nconstant_columns")
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    for c in constant_cols:
        print(c)

    print("\nlow_cardinality_columns_top30")
    low_card = []
    for c in df.columns:
        n = df[c].nunique(dropna=False)
        low_card.append((c, n))
    low_card = sorted(low_card, key=lambda x: x[1])[:30]
    for c, n in low_card:
        print(f"{c}: {n}")

    print("\nnon_numeric_columns")
    for c in df.columns:
        if c in [TIME_COL] + TARGETS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            print(c, str(df[c].dtype))

    print("\nnull_rate_top30")
    nulls = df.isna().mean().sort_values(ascending=False).head(30)
    for k, v in nulls.items():
        print(f"{k}: {v:.6f}")

if __name__ == "__main__":
    main()
