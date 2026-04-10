from pathlib import Path
import argparse
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "experiments" / "config" / "benchmark.yaml"


def resolve_benchmark_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_BENCHMARK_PATH))
    args, _ = parser.parse_known_args()
    return Path(args.config).resolve()


def load_benchmark_config() -> dict:
    benchmark_path = resolve_benchmark_path()
    with benchmark_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_benchmark_path"] = str(benchmark_path)
    return cfg
