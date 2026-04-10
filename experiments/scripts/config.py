from pathlib import Path
import argparse
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "experiments" / "config" / "benchmark.yaml"


def resolve_path(value: str) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def build_sqlite_uri(db_path: str) -> str:
    db = Path(db_path)
    if not db.is_absolute():
        db = (PROJECT_ROOT / db).resolve()
    return f"sqlite:///{db}"


def resolve_benchmark_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_BENCHMARK_PATH))
    args, _ = parser.parse_known_args()
    return Path(args.config).resolve()


def load_benchmark_config() -> dict:
    benchmark_path = resolve_benchmark_path()
    with benchmark_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "tracking_db_path" in cfg:
        cfg["tracking_uri"] = build_sqlite_uri(cfg["tracking_db_path"])

    if "dataset" in cfg and "path" in cfg["dataset"]:
        cfg["dataset"]["path"] = resolve_path(cfg["dataset"]["path"])

    cfg["_benchmark_path"] = str(benchmark_path)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg
