from pathlib import Path
import pandas as pd

DATASET_PATH = Path("/Users/nazlidecker/ml-project/data/features/dataset_v3.csv")

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
TEST_END = "2026-12-31"

TARGETS = ["label_7d", "label_30d", "label_365d"]
TIME_COL = "datetime"

def positive_rate(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return float(s.mean())

def main():
    usecols = [TIME_COL] + TARGETS
    df = pd.read_csv(DATASET_PATH, usecols=usecols, low_memory=False)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COL]).copy()

    print("dataset_path", DATASET_PATH)
    print("rows", len(df))
    print("datetime_min", df[TIME_COL].min())
    print("datetime_max", df[TIME_COL].max())

    train = df[df[TIME_COL] <= TRAIN_END].copy()
    val = df[(df[TIME_COL] > TRAIN_END) & (df[TIME_COL] <= VAL_END)].copy()
    test = df[(df[TIME_COL] > VAL_END) & (df[TIME_COL] <= TEST_END)].copy()

    print("\nsplit_sizes")
    print("train", len(train))
    print("validation", len(val))
    print("test", len(test))

    for name, part in [("train", train), ("validation", val), ("test", test)]:
        print(f"\npositive_rates_{name}")
        for target in TARGETS:
            print(target, positive_rate(part[target]))

if __name__ == "__main__":
    main()
