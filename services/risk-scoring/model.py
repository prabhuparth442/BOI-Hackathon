import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from feature_extractor import extract_features, extract_xgboost_features, FEATURE_NAMES, XGBOOST_FEATURE_NAMES

logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/xgb_risk_model.json"))

# Severity scores per class — must match training order in notebook
# CLASSES = ["Benign", "Riskware", "Adware", "SMS", "Banking"]
SEVERITY_VEC = np.array([5, 35, 55, 75, 95], dtype=np.float32)
CLASS_NAMES = ["Benign", "Riskware", "Adware", "SMS", "Banking"]

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    if MODEL_PATH.exists():
        import xgboost as xgb
        _model = xgb.Booster()
        _model.load_model(str(MODEL_PATH))
        logger.info("Loaded XGBoost model from %s", MODEL_PATH)
    else:
        logger.warning("No trained model found at %s — using heuristic scoring", MODEL_PATH)
        _model = "heuristic"

    return _model


def predict_score(features: np.ndarray, _raw_data: dict = None) -> float:
    model = _load_model()

    if model == "heuristic":
        return _heuristic_score(features)

    try:
        import xgboost as xgb

        # XGBoost must receive exactly the 12 features it was trained on.
        # Pass the raw data dict so we can extract the correct 12-feature slice.
        # 'features' here may be 16-wide (heuristic); we rebuild the 12-wide version.
        xgb_features = extract_xgboost_features(_raw_data) if _raw_data else features[:12]
        dmatrix = xgb.DMatrix(xgb_features.reshape(1, -1), feature_names=XGBOOST_FEATURE_NAMES)

        # model was trained with multi:softprob — output shape is (n_samples, n_classes)
        best_iteration = getattr(model, 'best_iteration', 0)
        kwargs = {}
        if best_iteration > 0:
            kwargs["iteration_range"] = (0, best_iteration + 1)
        proba = model.predict(dmatrix, **kwargs)
        proba = np.array(proba).reshape(-1, len(CLASS_NAMES))

        # weighted sum: proba[0] @ severity_vec gives a continuous 0-100 risk score
        risk_score = float(proba[0] @ SEVERITY_VEC)
        risk_score = round(min(max(risk_score, 0.0), 100.0), 1)

        return risk_score

    except Exception as exc:
        logger.warning("XGBoost prediction failed, falling back to heuristic: %s", exc)
        return _heuristic_score(features)


def predict_class(features: np.ndarray, _raw_data: dict = None) -> Dict:
    model = _load_model()

    if model == "heuristic":
        return {"class": "Unknown", "probabilities": {}}

    try:
        import xgboost as xgb

        xgb_features = extract_xgboost_features(_raw_data) if _raw_data else features[:12]
        dmatrix = xgb.DMatrix(xgb_features.reshape(1, -1), feature_names=XGBOOST_FEATURE_NAMES)
        best_iteration = getattr(model, 'best_iteration', 0)
        kwargs = {}
        if best_iteration > 0:
            kwargs["iteration_range"] = (0, best_iteration + 1)
        proba = model.predict(dmatrix, **kwargs)
        proba = np.array(proba).reshape(-1, len(CLASS_NAMES))[0]

        pred_idx = int(np.argmax(proba))
        return {
            "class": CLASS_NAMES[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, proba)},
        }
    except Exception as exc:
        logger.warning("Class prediction failed: %s", exc)
        return {"class": "Unknown", "probabilities": {}}


def explain_score(features: np.ndarray) -> List[Dict]:
    model = _load_model()

    if model == "heuristic":
        return _heuristic_explanation(features)

    try:
        import shap
        import xgboost as xgb

        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(features.reshape(1, -1))

        sv = np.asarray(raw_shap) if not isinstance(raw_shap, list) else np.stack(raw_shap, axis=0)

        # Collapse every axis except the one matching len(FEATURE_NAMES)
        feat_axis = sv.shape.index(len(FEATURE_NAMES))
        other_axes = tuple(a for a in range(sv.ndim) if a != feat_axis)

        # Use absolute mean for magnitude, signed mean for direction.
        # Previous bug: computed np.abs().mean() for both — direction was always
        # "increases_risk" because abs values are never negative.
        mean_abs_shap  = np.abs(sv).mean(axis=other_axes)   # magnitude
        mean_sign_shap = sv.mean(axis=other_axes)            # direction (signed)

        return [
            {
                "feature": FEATURE_NAMES[i],
                "value": float(features[i]),
                "shap_value": round(float(mean_abs_shap[i]), 4),
                "direction": (
                    "increases_risk" if mean_sign_shap[i] > 0
                    else "decreases_risk" if mean_sign_shap[i] < 0
                    else "neutral"
                ),
            }
            for i in range(len(FEATURE_NAMES))
        ]
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return _heuristic_explanation(features)


def _heuristic_score(features: np.ndarray) -> float:
    weights = np.array([
        4.0,   # dangerous_perm_count
        3.0,   # suspicious_api_count
        8.0,   # yara_match_count
        10.0,  # obfuscation_detected
        8.0,   # dynamic_code_loading
        2.0,   # hardcoded_url_count
        12.0,  # malicious_ioc_count
        15.0,  # sms_intercepted
        15.0,  # accessibility_abuse
        6.0,   # c2_connection_count
        5.0,   # runtime_downloads
        10.0,  # ai_confidence
        # QuarkEngine behavioral features — high weights because these are
        # confirmed criminal API sequences, not just static indicators
        10.0,  # quark_crime_count
        20.0,  # quark_max_confidence  ← strongest single signal
        8.0,   # quark_banking_crime
        8.0,   # quark_sms_crime
    ], dtype=np.float32)

    normalizers = np.array(
        [10, 10, 5, 1, 1, 10, 5, 1, 1, 5, 3, 1,
         10, 1, 1, 1],   # quark: crime_count/10, max_conf already 0-1, binary flags
        dtype=np.float32
    )
    normalized = np.clip(features / normalizers, 0, 1)
    raw = round(min(float(np.dot(normalized, weights)), 100.0), 1)

    # Static signals alone (permissions + suspicious APIs + obfuscation +
    # dynamic code loading) CAN push above 40 — many real malware samples
    # don't trigger dynamic sandbox signals (emulator detection, needs user
    # interaction, etc.).  Only cap at 40 if the APK has ZERO suspicious
    # static indicators too (truly clean apps like Hello World).
    has_any_signal = (
        features[0] > 3    # 4+ dangerous permissions
        or features[1] > 2  # 3+ suspicious APIs
        or features[2] > 0  # yara_match_count
        or features[3] > 0  # obfuscation_detected
        or features[4] > 0  # dynamic_code_loading
        or features[6] > 0  # malicious_ioc_count
        or features[7] > 0  # sms_intercepted
        or features[8] > 0  # accessibility_abuse
        or features[9] > 0  # c2_connection_count
        or features[10] > 0 # runtime_downloads
        or features[12] > 0 # quark_crime_count
        or features[13] > 0 # quark_max_confidence
    )
    if not has_any_signal:
        raw = min(raw, 40.0)

    return raw



def _heuristic_explanation(features: np.ndarray) -> List[Dict]:
    contrib_weights = [4, 3, 8, 10, 8, 2, 12, 15, 15, 6, 5, 10, 10, 20, 8, 8]
    return [
        {
            "feature": FEATURE_NAMES[i],
            "value": float(features[i]),
            "shap_value": round(float(features[i]) * contrib_weights[i] / 100, 4),
            "direction": "increases_risk" if features[i] > 0 else "neutral",
        }
        for i in range(len(FEATURE_NAMES))
    ]
