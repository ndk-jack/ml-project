# Experiments

This folder contains the reproducible ML experimentation workflow for `ml-project`.

## Principles

- MLflow stores runs, params, metrics, artifacts, and benchmark outputs.
- GitHub stores only code, configs, docs, and retained champion assets.
- Every run must log:
  - benchmark_id
  - dataset_version
  - dataset_path
  - feature_set_version
  - split_version
  - git_sha
  - horizon
  - params
  - metrics
  - artifacts

## Official ML working base

- Official benchmark: `benchmark_v2`
- Dataset: `data/features_clean/dataset_v5_dedup.csv`
- Active targets:
  - `label_7d`
  - `label_30d`
- Active feature set candidate: `candidate_feature_set_v1`
- Best params reference: `experiments/config/best_params_v2.yaml`

All new ML experiments must use `--config experiments/config/benchmark_v2.yaml`.

## Local tracking

MLflow uses a SQLite backend stored at the project root:

- `mlflow.db`

Start UI locally with:

```bash
cd ~/ml-project
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:////$(pwd)/mlflow.db --host 127.0.0.1 --port 5001
```

## Folder structure

```
experiments/
├── config/
│   ├── benchmark_v2.yaml          # Official benchmark config (canonical)
│   ├── best_params_v2.yaml        # Optuna best params per target
│   └── benchmark_v1.yaml          # Legacy — do not use for new runs
├── scripts/
│   ├── config.py                  # Shared config loader (path-portable)
│   ├── baseline_naive.py          # Class-frequency naive baseline
│   ├── train_lightgbm_benchmark.py
│   ├── train_lightgbm_best_params.py
│   ├── train_lightgbm_compact_best_params.py  # Active candidate
│   ├── train_lightgbm_minimal_features.py
│   ├── optuna_search.py
│   ├── feature_importance_report.py
│   ├── feature_ablation_v2.py
│   ├── feature_selection_report.py
│   ├── compare_baseline_vs_challenger.py
│   ├── compare_compact_vs_full.py
│   └── compare_feature_sets.py
├── reports/
│   ├── feature_selection/         # candidate_feature_set_v1 report
│   ├── ablation/                  # Geo / background_rate ablation results
│   ├── compact_bestparams/        # Active candidate reports
│   └── model_decision_v2.md       # Decision note for benchmark_v2
├── manifests/
│   └── candidate_manifest_v2.json
└── data_clean/                    # Dataset cleaning pipeline (v4→v5_dedup)
```

## Run order (benchmark_v2)

```bash
source .venv/bin/activate
CONFIG=experiments/config/benchmark_v2.yaml

python experiments/scripts/baseline_naive.py --config $CONFIG
python experiments/scripts/train_lightgbm_benchmark.py --config $CONFIG
python experiments/scripts/optuna_search.py --config $CONFIG --target label_7d --trials 20
python experiments/scripts/optuna_search.py --config $CONFIG --target label_30d --trials 20
python experiments/scripts/train_lightgbm_compact_best_params.py --config $CONFIG
python experiments/scripts/compare_compact_vs_full.py --config $CONFIG
```
