import streamlit as st

st.set_page_config(
    page_title="Fake Review Detection - Prediction",
    page_icon="🤖",
    layout="wide",
)

import os
import sys

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so ``from src.*`` imports work
# regardless of the working directory (local dev, Docker, Streamlit Cloud).
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Download/verify NLTK data before any module that touches the corpora.
from src.nltk_setup import setup_nltk

setup_nltk()

import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import joblib
import numpy as np
import pandas as pd
import altair as alt

from src.preprocessing import preprocess_text
from src.features import extract_feature_vector, MODEL_PKL_FEATURE_ORDER

# Number of numeric features the legacy model.pkl was trained on.
_N_NUMERIC = len(MODEL_PKL_FEATURE_ORDER)  # 11


# ---------------------------------------------------------------------------
# Load Model & Vectorizer
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_vectorizer():
    model_path = os.path.join(_APP_DIR, "model.pkl")
    vectorizer_path = os.path.join(_APP_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(
            "Model files not found.  Please ensure 'model.pkl' and "
            "'tfidf_vectorizer.pkl' exist in the src/ directory."
        )
        return None, None
    m = joblib.load(model_path)
    v = joblib.load(vectorizer_path)
    return m, v


model, vectorizer = load_model_vectorizer()
if model is None or vectorizer is None:
    st.stop()

# ---------------------------------------------------------------------------
# Single Prediction
# ---------------------------------------------------------------------------
st.title("Fake Review Detection - Single Prediction")

review_text = st.text_area("Enter review text:", height=150)
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("Predict"):
    if not review_text.strip():
        st.error("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            # 1. Pre-process exactly as during training (lemmatization pipeline)
            processed = preprocess_text(review_text)
            x_tfidf = vectorizer.transform([processed])

            # 2. Extract the 11 numeric features the model was trained on
            #    (everything in FEATURE_ORDER except punctuation_count).
            numeric = extract_feature_vector(review_text)[:_N_NUMERIC].reshape(
                1, -1
            )

            # 3. Combine sparse TF-IDF + dense numeric features
            feats = np.hstack((x_tfidf.toarray(), numeric))

            try:
                probabilities = model.predict_proba(feats)[0]
            except Exception:
                probabilities = None
                prediction = model.predict(feats)[0]
            else:
                prediction = 1 if probabilities[1] >= threshold else 0

            result = "Computer Generated" if prediction == 0 else "Original"
            st.success(f"Prediction: {result}")

            if probabilities is not None:
                df_probs = pd.DataFrame(
                    {
                        "Label": ["Computer Generated", "Original"],
                        "Probability": [probabilities[0], probabilities[1]],
                    }
                )
                bar_chart = (
                    alt.Chart(df_probs)
                    .mark_bar(color="#66b3ff")
                    .encode(
                        x=alt.X("Label", axis=alt.Axis(labelAngle=0)),
                        y="Probability",
                    )
                    .properties(
                        title="Prediction Probability Distribution",
                        width=300,
                        height=400,
                    )
                )
                st.altair_chart(bar_chart, use_container_width=True)
