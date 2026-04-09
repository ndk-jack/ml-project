from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = PROJECT_ROOT / "experiments" / "config" / "benchmark.yaml"


def load_benchmark_config() -> dict:
    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
