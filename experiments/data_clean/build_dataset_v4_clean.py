from pathlib import Path
import json
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = PROJECT_ROOT / "experiments" / "data_clean" / "config" / "cleaning_rules_v1.yaml"


def resolve_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def main():
    rules = yaml.safe_load(RULES_PATH.read_text())

    src = resolve_path(rules["dataset_source"])
    dst = resolve_path(rules["dataset_output"])
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, low_memory=False)

    time_col = rules["time_column"]
    df[time_col] = pd.to_datetime(df[time_col], format="mixed", errors="coerce", utc=True)

    for col in rules.get("drop_columns_exact", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    if rules.get("drop_if_all_null", False):
        all_null_cols = [c for c in df.columns if df[c].isna().all()]
        if all_null_cols:
            df = df.drop(columns=all_null_cols)
    else:
        all_null_cols = []

    if rules.get("drop_if_constant", False):
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
    else:
        constant_cols = []

    target_cols = rules["target_columns"]
    keep = []
    for c in df.columns:
        if c == time_col or c in target_cols:
            keep.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)

    df = df[keep].copy()
    df = df.dropna(subset=[time_col])

    df.to_csv(dst, index=False)

    manifest = {
        "rules_path": str(RULES_PATH),
        "source": str(src),
        "output": str(dst),
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "datetime_min": str(df[time_col].min()),
        "datetime_max": str(df[time_col].max()),
        "dropped_all_null_columns": all_null_cols,
        "dropped_constant_columns": constant_cols,
        "target_columns": target_cols,
    }

    manifest_path = PROJECT_ROOT / "experiments" / "data_clean" / "manifests" / "dataset_v4_clean_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("dataset_output", dst)
    print("rows", len(df))
    print("column_count", len(df.columns))
    print("manifest", manifest_path)


if __name__ == "__main__":
    main()
