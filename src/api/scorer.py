"""
scorer.py — Load trained LightGBM models and score a feature vector.

Responsibilities:
  - Load lgbm_7d.txt, lgbm_30d.txt, lgbm_365d.txt at startup
  - Load training medians for NaN imputation (from dataset_v3.csv stats)
  - Expose score(feats_series) → dict with prob_7d, prob_30d, prob_365d
  - Compute a human-readable risk level per horizon
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from feature_engine import FEATURE_COLS

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR         = PROJECT_ROOT / "models"
FEATURES_DIR       = PROJECT_ROOT / "data" / "features"
RUNTIME_ASSETS_DIR = PROJECT_ROOT / "data" / "external"

MODEL_FILES = {
    "7d":   MODELS_DIR / "lgbm_7d.txt",
    "30d":  MODELS_DIR / "lgbm_30d.txt",
    "365d": MODELS_DIR / "lgbm_365d.txt",
}

# Fallback medians (from training dataset_v3) — used if dataset_v3.csv unavailable
FALLBACK_MEDIANS = {
    "magnitude": 4.6, "depth": 33.0, "elapsed_since_last_s": 86400.0,
    "dist_to_plate_boundary_km": 150.0,
    "count_1d": 0.0, "rate_1d": 0.0, "energy_1d": 0.0, "moment_1d": 0.0,
    "b_value_1d": 1.0, "mag_mean_1d": 4.5, "mag_max_1d": 4.5,
    "depth_mean_1d": 33.0, "depth_std_1d": 0.0,
    "count_7d": 2.0, "rate_7d": 0.29, "energy_7d": 1e6, "moment_7d": 1e15,
    "b_value_7d": 1.0, "mag_mean_7d": 4.5, "mag_max_7d": 4.8,
    "depth_mean_7d": 33.0, "depth_std_7d": 10.0,
    "count_30d": 8.0, "rate_30d": 0.27, "energy_30d": 5e6, "moment_30d": 5e15,
    "b_value_30d": 1.0, "mag_mean_30d": 4.5, "mag_max_30d": 5.0,
    "depth_mean_30d": 33.0, "depth_std_30d": 12.0,
    "count_90d": 20.0, "rate_90d": 0.22, "energy_90d": 1e7, "moment_90d": 1e16,
    "b_value_90d": 1.0, "mag_mean_90d": 4.5, "mag_max_90d": 5.2,
    "depth_mean_90d": 35.0, "depth_std_90d": 15.0,
    "count_7d_m2": 5.0, "energy_7d_m2": 2e6, "b_value_7d_m2": 1.0,
    "count_30d_m2": 20.0, "energy_30d_m2": 8e6, "b_value_30d_m2": 1.0,
    "count_90d_m2": 60.0, "energy_90d_m2": 2e7, "b_value_90d_m2": 1.0,
    "accel_count": 1.0, "accel_energy": 1.0, "mag_excess": 0.0,
    "dist_to_nearest_fault_km": 85.0, "fault_slip_type_enc": 0,
    "stress_regime_enc": 1, "shmax_sin": 0.0, "shmax_cos": 1.0,
    "wsm_dist_km": 150.0, "background_rate_yr": 12.6, "normalized_rate_30d": 5.0,
}

# Risk thresholds — tuned to be informative for non-technical users
RISK_LEVELS = [
    (0.70, "🔴 Très élevé"),
    (0.50, "🟠 Élevé"),
    (0.35, "🟡 Modéré"),
    (0.20, "🟢 Faible"),
    (0.00, "⚪ Très faible"),
]


def _risk_label(prob: float) -> str:
    for threshold, label in RISK_LEVELS:
        if prob >= threshold:
            return label
    return "⚪ Très faible"


class Scorer:
    """Loads models once and scores feature vectors."""

    def __init__(self):
        self.models: dict[str, lgb.Booster] = {}
        self.medians: dict[str, float] = {}
        self._ready = False

        self.using_fallback_medians: bool = False
        self.medians_source: str | None = None
        self.last_imputed_feature_count: int = 0
        self.last_imputed_features: list[str] = []

    def initialize(self):
        """Load models and compute imputation medians. Call once at startup."""
        self._load_models()
        self._load_medians()
        self._ready = True
        logger.info("Scorer ready.")

    @property
    def ready(self) -> bool:
        return self._ready and bool(self.models)

    def score(self, feats: pd.Series) -> dict:
        """
        Score a single event.

        Parameters
        ----------
        feats : pd.Series — all computed features (superset is fine)

        Returns
        -------
        dict with keys: prob_7d, prob_30d, prob_365d,
                        risk_7d, risk_30d, risk_365d,
                        features_used (int)

        Each model selects its own expected features via model.feature_name(),
        so mismatches between training and inference are handled gracefully.
        """
        if not self.ready:
            raise RuntimeError("Scorer not initialized.")

        result = {}
        total_used = 0
        imputed_features_all: set[str] = set()
        total_imputed_count = 0

        for horizon, model in self.models.items():
            # Use the exact feature list the model was trained with
            expected = model.feature_name()
            vec = pd.Series({col: feats.get(col, np.nan) for col in expected})

            # Impute NaN with training medians
            nan_mask = vec.isna()
            missing_cols = list(vec.index[nan_mask])
            nan_count = len(missing_cols)

            for col in missing_cols:
                vec[col] = self.medians.get(col, 0.0)

            imputed_features_all.update(missing_cols)
            total_imputed_count += nan_count

            X = vec.values.reshape(1, -1)
            prob = float(model.predict(X)[0])

            result[f"prob_{horizon}"] = round(prob, 4)
            result[f"risk_{horizon}"] = _risk_label(prob)
            total_used = max(total_used, len(expected) - nan_count)

        self.last_imputed_feature_count = total_imputed_count
        self.last_imputed_features = sorted(imputed_features_all)

        if total_imputed_count > 0:
            logger.info(
                "Score imputation used %s values across %s unique features.",
                total_imputed_count,
                len(self.last_imputed_features),
            )

        result["features_used"] = total_used
        result["imputed_feature_count"] = total_imputed_count
        result["imputed_unique_features"] = len(self.last_imputed_features)

        # Fill missing horizons as None
        for h in ["7d", "30d", "365d"]:
            if f"prob_{h}" not in result:
                result[f"prob_{h}"] = None
                result[f"risk_{h}"] = "N/A (model not loaded)"

        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_models(self):
        for horizon, path in MODEL_FILES.items():
            if path.exists():
                try:
                    self.models[horizon] = lgb.Booster(model_file=str(path))
                    logger.info(f"Model {horizon} loaded from {path}")
                except Exception as e:
                    logger.error(f"Failed to load model {horizon}: {e}")
            else:
                logger.warning(f"Model file not found: {path}")

        if not self.models:
            raise FileNotFoundError(
                f"No model files found in {MODELS_DIR}. "
                "Run train_multi_horizon.py first."
            )

    def _load_medians(self):
        """Load training medians from packaged JSON artifact, fallback to dataset_v3.csv, then constants."""
        medians_json = RUNTIME_ASSETS_DIR / "feature_medians_v3.json"
        dataset = FEATURES_DIR / "dataset_v3.csv"

        if medians_json.exists():
            try:
                logger.info(f"Loading imputation medians from {medians_json}…")
                self.medians = json.loads(medians_json.read_text())
                self.medians = {k: float(v) for k, v in self.medians.items()}
                self.using_fallback_medians = False
                self.medians_source = str(medians_json)
                logger.info(
                    f"Medians loaded for {len(self.medians)} features from {medians_json}."
                )
                return
            except Exception as e:
                logger.warning(f"Could not read median artifact {medians_json}: {e}")

        if dataset.exists():
            try:
                logger.info("Computing imputation medians from dataset_v3.csv…")
                df = pd.read_csv(dataset, usecols=FEATURE_COLS, low_memory=False)
                self.medians = df.median(numeric_only=True).to_dict()
                self.medians = {k: float(v) for k, v in self.medians.items()}
                self.using_fallback_medians = False
                self.medians_source = str(dataset)
                logger.info(
                    f"Medians computed for {len(self.medians)} features from {dataset}."
                )
                return
            except Exception as e:
                logger.warning(f"Could not read dataset_v3.csv: {e}")

        logger.warning("Using fallback medians.")
        self.medians = FALLBACK_MEDIANS.copy()
        self.using_fallback_medians = True
        self.medians_source = "FALLBACK_MEDIANS"


# Singleton
scorer = Scorer()
