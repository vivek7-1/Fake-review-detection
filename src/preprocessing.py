"""Single preprocessing function for TF-IDF (training, evaluation, and inference).

This matches the N1 lemmatization configuration that produced
`preprocessed_lemmatization.csv`:

    use_spell_correction = False
    remove_stopwords = False
    use_stemming = False
    use_lemmatization = True
    expand_contractions = True

Order applied to each review:

1. Drop non-string / missing values -> empty string
2. Remove non-ASCII (emojis)
3. Expand contractions (I've -> I have)
4. Strip HTML tags
5. Lowercase
6. Convert standalone digits to words (4 -> four)
7. Remove punctuation
8. Word-tokenize
9. WordNet lemmatize using sentence-level POS tags

Spell correction (SymSpell) is intentionally omitted. N1 already set
`use_spell_correction=False` for every saved preprocessed dataset. Applying
SymSpell only at inference, or rewriting generated-text typos, would change
the TF-IDF vocabulary relative to training.

Do not use this function for punctuation / style / NER features. Those are
computed from the original review in `features.py`.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.nltk_setup import setup_nltk

_HTML_RE = re.compile(r"<[^>]+>")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_NUMBER_RE = re.compile(r"\b\d+\b")
# WordNet POS codes (avoid importing nltk.corpus.wordnet at module load).
_POS_MAP = {
    "J": "a",
    "N": "n",
    "V": "v",
    "R": "r",
}


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    setup_nltk()
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _inflect_engine():
    import inflect

    return inflect.engine()


def _expand_contractions(text: str) -> str:
    import contractions

    return contractions.fix(text)


def _convert_numbers(text: str) -> str:
    engine = _inflect_engine()

    def _replace(match: re.Match) -> str:
        words = engine.number_to_words(match.group())
        return words if isinstance(words, str) else " ".join(words)

    return _NUMBER_RE.sub(_replace, text)


def _wordnet_pos(tag: str):
    return _POS_MAP.get(tag[:1], "n")


def preprocess_text(text) -> str:
    """Normalize a review for TF-IDF. Stateless — safe to use on train and test."""
    setup_nltk()
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    if not text.strip():
        return ""

    text = _NON_ASCII_RE.sub("", text)
    text = _expand_contractions(text)
    text = _HTML_RE.sub("", text)
    text = text.lower()
    text = _convert_numbers(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    if not words:
        return ""

    tagged = nltk.pos_tag(words)
    lemmatizer = _lemmatizer()
    lemmas = [
        lemmatizer.lemmatize(word, _wordnet_pos(tag)) for word, tag in tagged
    ]
    return " ".join(lemmas)


class TextPreprocessor:
    """sklearn-compatible transformer wrapping `preprocess_text`."""

    def fit(self, X, y=None):
        setup_nltk()
        return self

    def transform(self, X):
        return [preprocess_text(text) for text in X]
