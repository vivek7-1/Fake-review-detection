"""Train the final leakage-free pipeline and write metrics.

Usage (from the project root):

    python -m src.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.labels import ID_TO_NAME, encode_label
from src.nltk_setup import setup_nltk
from src.pipeline import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "Raw" / "fakeReviewData.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fake_review_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text_" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain text_ and label columns.")
    df = df[["text_", "label"]].copy()
    df["text_"] = df["text_"].astype(str)
    df = df[df["text_"].str.strip().astype(bool)].drop_duplicates()
    df["label_id"] = df["label"].map(encode_label)
    return df.reset_index(drop=True)


def evaluate(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    setup_nltk()
    print(f"Loading {DATA_PATH}")
    df = load_dataset()
    X = df["text_"].tolist()
    y = df["label_id"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"Split: train={len(X_train)} test={len(X_test)} "
        f"test_size={TEST_SIZE} random_state={RANDOM_STATE} stratify=True"
    )

    pipeline = build_pipeline()
    print("Fitting TF-IDF + numeric features + LogisticRegression on TRAIN only...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    metrics.update(
        {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "label_mapping": ID_TO_NAME,
            "architecture": "tfidf_processed_text + 12_raw_numeric_features + StandardScaler + LogisticRegression",
            "note": (
                "These scores are from the corrected leakage-free pipeline. "
                "They are not the old 0.91 TF-IDF row in Notebooks/progress_log.csv, "
                "which used a different experiment and fit transforms on all rows."
            ),
        }
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "confusion_matrix")}, indent=2))

    # Sanity: scaler/vectorizer exist and dummy-zero path is not used.
    combiner = pipeline.named_steps["features"]
    assert hasattr(combiner, "vectorizer_") and hasattr(combiner, "scaler_")
    n_numeric = combiner.scaler_.mean_.shape[0]
    print(f"Numeric features scaled: {n_numeric} (punctuation_count included)")
    print(f"TF-IDF vocabulary size: {len(combiner.vectorizer_.vocabulary_)}")
    print(f"Scaler fitted mean of punctuation_count: {combiner.scaler_.mean_[-1]:.4f}")


if __name__ == "__main__":
    main()
