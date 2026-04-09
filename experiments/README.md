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

## Local tracking

MLflow tracking URI:

file:/Users/nazlidecker/ml-project/mlruns

Start UI locally with:

```bash
cd /Users/nazlidecker/ml-project
source .venv/bin/activate
mlflow ui --backend-store-uri file:/Users/nazlidecker/ml-project/mlruns --host 127.0.0.1 --port 5001
```

## First milestone

The first milestone is not tuning. It is:
1. freeze benchmark config
2. log the current champion baseline
3. validate dataset manifest and benchmark metadata
