from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import (
    FEATURE_ORDER,
    FeatureExtractionError,
    extract_feature_dict,
    punctuation_count,
)
from src.labels import encode_label, decode_label
from src.pipeline import ReviewFeatureCombiner, build_pipeline
from src.preprocessing import preprocess_text


def test_label_mapping_matches_raw_codes():
    assert encode_label("CG") == 0
    assert encode_label("OR") == 1
    assert decode_label(0) == "Computer Generated"
    assert decode_label(1) == "Original"


def test_punctuation_count_varies_on_raw_text():
    punctuated = "Love this! Well made, sturdy, and very comfortable."
    plain = "Love this well made sturdy and very comfortable"
    assert punctuation_count(punctuated) > 0
    assert punctuation_count(plain) == 0
    assert punctuation_count(punctuated) != punctuation_count(plain)


def test_features_use_raw_punctuation_not_processed_text():
    raw = "Amazing product!!! I'd buy it again."
    feats = extract_feature_dict(raw)
    assert feats["punctuation_count"] > 0
    processed = preprocess_text(raw)
    assert "!" not in processed
    assert feats["punctuation_count"] == punctuation_count(raw)


def test_empty_text_raises():
    try:
        extract_feature_dict("   ")
    except FeatureExtractionError:
        return
    raise AssertionError("Expected FeatureExtractionError")


def test_preprocess_is_deterministic():
    text = "I've had mine for 4 years. LOVE it!"
    assert preprocess_text(text) == preprocess_text(text)
    assert preprocess_text(text) == preprocess_text(text).lower()


def test_tfidf_and_scaler_fit_on_train_only():
    texts = [
        "Great product, highly recommend!",
        "Terrible quality. Broke immediately.",
        "Okay item. Does the job.",
        "Worst purchase ever!!!",
        "Fantastic build quality and comfort.",
        "Do not buy this, waste of money.",
    ]
    y = np.array([1, 0, 1, 0, 1, 0])
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.33, random_state=42, stratify=y
    )
    combiner = ReviewFeatureCombiner(max_features=50)
    combiner.fit(X_train, y_train)
    train_vocab = set(combiner.vectorizer_.vocabulary_)
    test_only = preprocess_text("xylophone zebra quark")
    # Transform must use the train-fitted vectorizer, not refit.
    _ = combiner.transform(X_test)
    assert set(combiner.vectorizer_.vocabulary_) == train_vocab
    assert combiner.scaler_.mean_.shape[0] == len(FEATURE_ORDER)
    assert combiner.scaler_.mean_[-1] > 0  # punctuation_count mean on train


def test_end_to_end_pipeline_predicts():
    texts = [
        "Love this! Well made, sturdy, and very comfortable.",
        "Missing information on how to use it, but it is a great product.",
        "This pillow saved my back. I love the look and feel of this pillow.",
        "Broke after one day. Cheap materials and awful stitching!!!",
        "Perfect size, great upgrade from the original I've owned.",
        "Do not waste your money. Completely useless junk.",
    ]
    y = np.array([1, 0, 1, 0, 1, 0])
    pipe = build_pipeline(max_features=80)
    pipe.fit(texts, y)
    pred = pipe.predict(["Love this! Well made and comfortable."])
    proba = pipe.predict_proba(["Love this! Well made and comfortable."])
    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1) < 1e-6


def test_no_dummy_zero_vector_in_app_source():
    app = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    assert "np.zeros((1, 11))" not in app
    assert "dummy" not in app
