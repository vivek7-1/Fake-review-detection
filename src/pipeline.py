"""sklearn Pipeline that trains and serves with the same transformers.

Raw review
  ├─> preprocess_text() ─> TfidfVectorizer          (fit on TRAIN only)
  └─> extract_feature_matrix() ─> StandardScaler    (fit on TRAIN only)
        └─> hstack sparse TF-IDF + scaled numerics
              └─> LogisticRegression
"""

from __future__ import annotations

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_ORDER, extract_feature_matrix
from src.preprocessing import preprocess_text


class ReviewFeatureCombiner(BaseEstimator, TransformerMixin):
    """Combines TF-IDF of processed text with scaled original-text features.

    Fitted attributes (`vectorizer_`, `scaler_`) are created in `fit` so they
    are fit on the current train fold only. Pipeline.fit uses `fit_transform`
    to avoid computing features twice.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        texts = list(X)
        processed = []
        n = len(texts)
        for i, text in enumerate(texts):
            if n >= 1000 and i % 1000 == 0:
                print(f"  preprocess {i}/{n}", flush=True)
            processed.append(preprocess_text(text))
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features)
        tfidf = self.vectorizer_.fit_transform(processed)
        numeric = extract_feature_matrix(texts)
        self.scaler_ = StandardScaler()
        numeric_scaled = self.scaler_.fit_transform(numeric)
        self.feature_names_ = list(FEATURE_ORDER)
        return hstack([tfidf, csr_matrix(numeric_scaled)])

    def transform(self, X):
        if not hasattr(self, "vectorizer_") or not hasattr(self, "scaler_"):
            raise RuntimeError("ReviewFeatureCombiner is not fitted.")
        texts = list(X)
        processed = [preprocess_text(text) for text in texts]
        tfidf = self.vectorizer_.transform(processed)
        numeric = self.scaler_.transform(extract_feature_matrix(texts))
        return hstack([tfidf, csr_matrix(numeric)])


def build_pipeline(max_features: int = 5000) -> Pipeline:
    return Pipeline(
        [
            ("features", ReviewFeatureCombiner(max_features=max_features)),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=200,
                ),
            ),
        ]
    )
