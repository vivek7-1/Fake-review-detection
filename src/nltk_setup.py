"""Portable NLTK data directory for Windows, Linux, and Streamlit Cloud."""

from __future__ import annotations

import os
from pathlib import Path

import nltk

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
    ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
    ("corpora/words", "words"),
)


def nltk_data_dir() -> Path:
    env = os.environ.get("NLTK_DATA")
    if env:
        path = Path(env)
    else:
        path = _PROJECT_ROOT / ".nltk_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_nltk() -> Path:
    data_dir = nltk_data_dir()
    data_path = str(data_dir)
    if data_path not in nltk.data.path:
        nltk.data.path.insert(0, data_path)
    os.environ["NLTK_DATA"] = data_path
    for resource_path, package in _REQUIRED:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, download_dir=data_path, quiet=True)
    return data_dir
