"""
scorer.py — Hybrid MLP/LightGBM scorer.

- label_7d  : MLP Keras + temperature scaling (T=0.6663)
- label_30d : MLP Keras + temperature scaling (T=1.0357)
- label_365d: LightGBM (conservé, pas de MLP champion pour ce horizon)

Pipeline d'inférence MLP :
  feats_s → sélection 67 features → imputer.transform() → scaler.transform()
          → keras.predict() → temperature_scale() → prob calibrée

Interface identique à l'ancien scorer : drop-in replacement.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR          = Path(os.getenv("MODEL_DATA_DIR",    "data/external"))
MLP_7D_PATH        = Path(os.getenv("MLP_7D_PATH",       str(_DATA_DIR / "mlp_full_medium__label_7d.keras")))
MLP_30D_PATH       = Path(os.getenv("MLP_30D_PATH",      str(_DATA_DIR / "mlp_full_medium__label_30d.keras")))
PREP_7D_PATH       = Path(os.getenv("PREP_7D_PATH",      str(_DATA_DIR / "mlp_full_medium__label_7d__preprocessor.joblib")))
PREP_30D_PATH      = Path(os.getenv("PREP_30D_PATH",     str(_DATA_DIR / "mlp_full_medium__label_30d__preprocessor.joblib")))
LGBM_365D_PATH     = Path(os.getenv("LGBM_365D_PATH",    str(_DATA_DIR / "lgbm_365d.txt")))
FEATURES_META_PATH = Path(os.getenv("FEATURES_META_PATH",str(_DATA_DIR / "features_metadata.json")))
TEMP_SCALE_PATH    = Path(os.getenv("TEMP_SCALE_PATH",   str(_DATA_DIR / "temperature_scaling.json")))
MEDIANS_PATH       = Path(os.getenv("MEDIANS_PATH",      str(_DATA_DIR / "feature_medians_v3.json")))


def _risk_label(p: Optional[float]) -> tuple[str, str]:
    if p is None:         return "⚪ Inconnu",     "unknown"
    if p < 0.20:          return "🟢 Très faible", "very_low"
    if p < 0.35:          return "🟡 Faible",      "low"
    if p < 0.50:          return "🟡 Modéré",      "moderate"
    if p < 0.70:          return "🟠 Élevé",       "high"
    return                       "🔴 Très élevé",  "very_high"


def _t_scale(p: float, T: float) -> float:
    """p_cal = sigmoid(logit(p) / T)."""
    p = max(1e-7, min(1 - 1e-7, p))
    return float(1.0 / (1.0 + math.exp(-math.log(p / (1 - p)) / T)))


class HybridMLPScorer:

    def __init__(self):
        self.models:        dict = {}
        self.preprocessors: dict = {}
        self.ready:         bool = False
        self._feature_names: list = []
        self._temperatures:  dict = {}
        self._medians:       dict = {}

    def initialize(self) -> None:
        import tensorflow as tf

        # Features metadata
        meta = json.loads(FEATURES_META_PATH.read_text())
        self._feature_names = meta["feature_names"]
        logger.info(f"Features : {len(self._feature_names)} colonnes")

        # Temperature scaling
        self._temperatures = json.loads(TEMP_SCALE_PATH.read_text())["temperatures"]
        logger.info(f"Temperature scaling : {self._temperatures}")

        # Medians (fallback pour features absentes de feats_s)
        if MEDIANS_PATH.exists():
            self._medians = json.loads(MEDIANS_PATH.read_text())

        # MLP 7d + 30d
        for horizon, keras_path, prep_path in [
            ("7d",  MLP_7D_PATH,  PREP_7D_PATH),
            ("30d", MLP_30D_PATH, PREP_30D_PATH),
        ]:
            if not keras_path.exists():
                raise FileNotFoundError(f"Modèle manquant : {keras_path}")
            if not prep_path.exists():
                raise FileNotFoundError(f"Préprocesseur manquant : {prep_path}")
            self.models[horizon]        = tf.keras.models.load_model(keras_path)
            self.preprocessors[horizon] = joblib.load(prep_path)
            logger.info(f"MLP {horizon} chargé : {keras_path.name}")

        # LightGBM 365d (optionnel — conservé de l'ancienne version)
        if LGBM_365D_PATH.exists():
            import lightgbm as lgb
            self.models["365d"] = (
                lgb.Booster(model_file=str(LGBM_365D_PATH))
                if LGBM_365D_PATH.suffix == ".txt"
                else joblib.load(LGBM_365D_PATH)
            )
            logger.info(f"LightGBM 365d chargé : {LGBM_365D_PATH.name}")
        else:
            logger.warning("LightGBM 365d absent — prob_365d sera None")

        self.ready = True
        logger.info(f"HybridMLPScorer prêt. Modèles : {list(self.models.keys())}")

    def _preprocess(self, feats_s: pd.Series, horizon: str) -> np.ndarray:
        """
        Sélectionne les 67 features dans le bon ordre,
        applique imputer → scaler, retourne un array (1, 67).
        """
        prep         = self.preprocessors[horizon]
        feature_cols = prep["feature_columns"]

        row = {f: feats_s.get(f, np.nan) for f in feature_cols}
        df  = pd.DataFrame([row], columns=feature_cols)

        x_imp = prep["imputer"].transform(df)
        x_scl = prep["scaler"].transform(x_imp)
        return x_scl.astype(np.float32)

    def score(self, feats_s: pd.Series) -> dict:
        if not self.ready:
            raise RuntimeError("Scorer non initialisé. Appeler initialize() d'abord.")

        # MLP 7d / 30d
        probs: dict = {}
        for horizon, label_key in [("7d", "label_7d"), ("30d", "label_30d")]:
            x   = self._preprocess(feats_s, horizon)
            raw = float(self.models[horizon].predict(x, verbose=0)[0][0])
            probs[horizon] = _t_scale(raw, self._temperatures.get(label_key, 1.0))

        # LightGBM 365d
        prob_365d: Optional[float] = None
        if "365d" in self.models:
            import lightgbm as lgb
            m = self.models["365d"]
            if isinstance(m, lgb.Booster):
                feat_names = m.feature_name()
                x365 = np.array(
                    [feats_s.get(f, self._medians.get(f, 0.0)) for f in feat_names],
                    dtype=np.float32,
                ).reshape(1, -1)
                prob_365d = float(m.predict(x365)[0])
            else:
                feat_names = getattr(m, "feature_names_in_", self._feature_names)
                x365 = pd.DataFrame(
                    [[feats_s.get(f, self._medians.get(f, 0.0)) for f in feat_names]],
                    columns=feat_names,
                )
                prob_365d = float(m.predict_proba(x365)[0][1])

        risk_7d,   risk_7d_code   = _risk_label(probs["7d"])
        risk_30d,  risk_30d_code  = _risk_label(probs["30d"])
        risk_365d, risk_365d_code = _risk_label(prob_365d)

        return {
            "prob_7d":        round(probs["7d"], 4),
            "prob_30d":       round(probs["30d"], 4),
            "prob_365d":      round(prob_365d, 4) if prob_365d is not None else None,
            "risk_7d":        risk_7d,
            "risk_30d":       risk_30d,
            "risk_365d":      risk_365d,
            "risk_7d_code":   risk_7d_code,
            "risk_30d_code":  risk_30d_code,
            "risk_365d_code": risk_365d_code,
            "features_used":  len(self._feature_names),
        }


# Singleton — import existant dans main.py inchangé
scorer = HybridMLPScorer()