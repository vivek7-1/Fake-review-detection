"""Deprecated.

The production vectorizer lives inside models/fake_review_pipeline.pkl.
Do not fit a standalone TfidfVectorizer on the full dataset.

Train the real pipeline:

    python -m src.train
"""

raise SystemExit(
    "src/temp.py is deprecated. It used to fit TF-IDF on all rows and save "
    "tfidf_vectorizer.pkl, which caused train/serve mismatch. "
    "Run: python -m src.train"
)
