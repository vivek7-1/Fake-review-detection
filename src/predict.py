"""Load the saved pipeline and score a single review."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.features import FeatureExtractionError
from src.labels import ID_TO_NAME, decode_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fake_review_pipeline.pkl"


def load_pipeline(model_path: str | Path | None = None):
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline not found at {path}. Train it with: python -m src.train"
        )
    return joblib.load(path)


def predict_review(text: str, pipeline=None, threshold: float = 0.5) -> dict:
    if text is None or not str(text).strip():
        raise FeatureExtractionError("Review text is empty.")
    pipe = pipeline if pipeline is not None else load_pipeline()
    probabilities = pipe.predict_proba([text])[0]
    label_id = 1 if probabilities[1] >= threshold else 0
    return {
        "label_id": int(label_id),
        "label_code": "OR" if label_id == 1 else "CG",
        "label_name": decode_label(label_id),
        "threshold": float(threshold),
        "probabilities": {
            ID_TO_NAME[0]: float(probabilities[0]),
            ID_TO_NAME[1]: float(probabilities[1]),
        },
    }
