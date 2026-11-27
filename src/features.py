"""Style / statistical features computed from the ORIGINAL review text.

N5 trained on these 11 columns (in this order), all of which had been
computed on already-cleaned `processed_text`:

    lexical_diversity, avg_word_length, sentiment_polarity, subjectivity,
    flesch_reading_ease, sentence_length, named_entity_count,
    noun_count, verb_count, adj_count, adv_count

N2 also computed `punctuation_count` but N5 dropped it because it was 0
everywhere (punctuation had already been stripped). This module keeps that
feature, calculated on the raw review, as the 12th numeric column.

Decision per feature (all use original text):

- punctuation_count: raw (must see punctuation)
- named_entity_count: raw (needs capitalization)
- flesch_reading_ease: raw (needs original sentences)
- POS counts / lexical diversity / avg word length / sentence_length: raw
- sentiment_polarity / subjectivity: raw (TextBlob on the user's wording)

Missing or empty text raises FeatureExtractionError. Nothing is silently
zero-filled.
"""

from __future__ import annotations

import string
from typing import Iterable

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from src.nltk_setup import setup_nltk

FEATURE_ORDER = (
    "lexical_diversity",
    "avg_word_length",
    "sentiment_polarity",
    "subjectivity",
    "flesch_reading_ease",
    "sentence_length",
    "named_entity_count",
    "noun_count",
    "verb_count",
    "adj_count",
    "adv_count",
    "punctuation_count",
)

# The 11 features that model.pkl (N5 training) was trained on.
# N5 dropped punctuation_count because it was always 0 on already-cleaned text.
MODEL_PKL_FEATURE_ORDER = FEATURE_ORDER[:11]


class FeatureExtractionError(ValueError):
    """Raised when a numeric feature cannot be computed."""


def _require_text(text) -> str:
    if text is None:
        raise FeatureExtractionError("Review text is missing.")
    if not isinstance(text, str):
        raise FeatureExtractionError(
            f"Review text must be a string, got {type(text).__name__}."
        )
    if not text.strip():
        raise FeatureExtractionError("Review text is empty.")
    return text


def punctuation_count(text: str) -> int:
    return sum(1 for char in text if char in string.punctuation)


def lexical_diversity(text: str) -> float:
    words = word_tokenize(text)
    if not words:
        raise FeatureExtractionError("lexical_diversity: no tokens in review.")
    return len(set(words)) / len(words)


def average_word_length(text: str) -> float:
    words = word_tokenize(text)
    if not words:
        raise FeatureExtractionError("avg_word_length: no tokens in review.")
    return sum(len(word) for word in words) / len(words)


def sentence_length(text: str) -> int:
    words = word_tokenize(text)
    if not words:
        raise FeatureExtractionError("sentence_length: no tokens in review.")
    return len(words)


def named_entity_count(text: str) -> int:
    tokens = word_tokenize(text)
    if not tokens:
        raise FeatureExtractionError("named_entity_count: no tokens in review.")
    tagged = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tagged, binary=True)
    return sum(1 for subtree in tree if getattr(subtree, "label", None))


def _pos_counts(text: str) -> dict:
    tokens = word_tokenize(text)
    if not tokens:
        raise FeatureExtractionError("POS counts: no tokens in review.")
    tagged = nltk.pos_tag(tokens)
    counts = {"noun_count": 0, "verb_count": 0, "adj_count": 0, "adv_count": 0}
    for _, tag in tagged:
        if tag.startswith("NN"):
            counts["noun_count"] += 1
        elif tag.startswith("VB"):
            counts["verb_count"] += 1
        elif tag.startswith("JJ"):
            counts["adj_count"] += 1
        elif tag.startswith("RB"):
            counts["adv_count"] += 1
    return counts


def sentiment_polarity(text: str) -> float:
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception as exc:
        raise FeatureExtractionError(f"sentiment_polarity failed: {exc}") from exc


def subjectivity_score(text: str) -> float:
    try:
        return float(TextBlob(text).sentiment.subjectivity)
    except Exception as exc:
        raise FeatureExtractionError(f"subjectivity failed: {exc}") from exc


def flesch_reading_ease_score(text: str) -> float:
    try:
        from textstat import flesch_reading_ease

        return float(flesch_reading_ease(text))
    except Exception as exc:
        raise FeatureExtractionError(f"flesch_reading_ease failed: {exc}") from exc


def extract_feature_dict(text) -> dict:
    setup_nltk()
    text = _require_text(text)
    tokens = word_tokenize(text)
    if not tokens:
        raise FeatureExtractionError("No tokens in review.")
    tagged = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tagged, binary=True)
    pos = {"noun_count": 0, "verb_count": 0, "adj_count": 0, "adv_count": 0}
    for _, tag in tagged:
        if tag.startswith("NN"):
            pos["noun_count"] += 1
        elif tag.startswith("VB"):
            pos["verb_count"] += 1
        elif tag.startswith("JJ"):
            pos["adj_count"] += 1
        elif tag.startswith("RB"):
            pos["adv_count"] += 1
    values = {
        "lexical_diversity": len(set(tokens)) / len(tokens),
        "avg_word_length": sum(len(word) for word in tokens) / len(tokens),
        "sentiment_polarity": sentiment_polarity(text),
        "subjectivity": subjectivity_score(text),
        "flesch_reading_ease": flesch_reading_ease_score(text),
        "sentence_length": len(tokens),
        "named_entity_count": sum(
            1 for subtree in tree if getattr(subtree, "label", None)
        ),
        "noun_count": pos["noun_count"],
        "verb_count": pos["verb_count"],
        "adj_count": pos["adj_count"],
        "adv_count": pos["adv_count"],
        "punctuation_count": punctuation_count(text),
    }
    missing = [name for name in FEATURE_ORDER if name not in values]
    if missing:
        raise FeatureExtractionError(f"Missing features: {missing}")
    return {name: values[name] for name in FEATURE_ORDER}


def extract_feature_vector(text) -> np.ndarray:
    values = extract_feature_dict(text)
    return np.array([values[name] for name in FEATURE_ORDER], dtype=float)


def extract_feature_matrix(texts: Iterable) -> np.ndarray:
    texts = list(texts)
    rows = []
    n = len(texts)
    for i, text in enumerate(texts):
        if n >= 1000 and i % 1000 == 0:
            print(f"  numeric features {i}/{n}", flush=True)
        rows.append(extract_feature_vector(text))
    return np.vstack(rows)


class ReviewFeatureExtractor:
    """sklearn transformer: raw reviews -> numeric feature matrix."""

    def fit(self, X, y=None):
        setup_nltk()
        return self

    def transform(self, X):
        return extract_feature_matrix(X)
