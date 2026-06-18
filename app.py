"""
Fake News Detector — Streamlit MVP frontend
===========================================

This is the user-facing interface for the project. It loads the single end-to-end
model exported by the notebook (`models/fake_news_model.keras`), which already
contains the text-vectorization step inside it, so this app only has to pass raw
text in and read a probability out.

Run from the project root:
    streamlit run app/app.py

The notebook must have been run at least once so the model file exists.
"""
import os
import re
import string

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# --- Locate the exported model (works regardless of where streamlit is launched) ---
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.normpath(os.path.join(HERE, "..", "models", "fake_news_model.keras"))
MODEL_PATH = os.environ.get("FAKE_NEWS_MODEL", DEFAULT_MODEL)


# --- The SAME standardization function used in training -----------------------
# It must be registered (and/or passed via custom_objects) so the saved model,
# which references it inside its TextVectorization layer, can be deserialized.
@keras.utils.register_keras_serializable(package="fake_news")
def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_tag = tf.strings.regex_replace(lowercase, "<[^>]+>", "")
    return tf.strings.regex_replace(no_tag, "[%s]" % re.escape(string.punctuation), "")


@st.cache_resource(show_spinner="Loading model…")
def load_fake_news_model(path):
    return keras.models.load_model(path, custom_objects={"standardization": standardization})


def predict(model, text):
    """Return p(fake) for a single raw string. Note the tf.constant wrapper:
    the model's first layer expects a string tensor, not a numpy string array."""
    return float(model.predict(tf.constant([text]), verbose=0).ravel()[0])


# --- Page setup ---------------------------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.caption("An RNN/LSTM model that flags likely fake news for human review. "
           "Paste a headline or article below.")

# Guard: model must exist
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found at:\n\n`{MODEL_PATH}`\n\n"
        "Run the notebook `notebooks/fake_news_detection.ipynb` end-to-end first — "
        "its final section exports the model the app needs."
    )
    st.stop()

model = load_fake_news_model(MODEL_PATH)

# --- Sidebar controls ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold for flagging as FAKE", 0.05, 0.95, 0.50, 0.05,
        help="Lower this to catch more fake news (higher recall, more false alarms). "
             "Raise it to flag only the most confident cases (higher precision).",
    )
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "The article text is lowercased, cleaned, and turned into a sequence of word "
        "tokens. A recurrent network reads the sequence and outputs a probability that "
        "the article is fake."
    )

# --- Example buttons ----------------------------------------------------------
EXAMPLES = {
    "📄 Looks legitimate":
        "According to officials, the central bank confirmed that quarterly inflation "
        "figures were consistent with earlier estimates. Analysts said the data matched "
        "seasonal expectations and that no immediate policy change was required.",
    "🚨 Looks like clickbait":
        "SHOCKING the government is HIDING the TRUTH about the economy and you WONT "
        "believe what happens next. Share this before they DELETE it forever — the "
        "mainstream media is TERRIFIED of this story.",
}

st.write("**Try an example:**")
ex_cols = st.columns(len(EXAMPLES))
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.article_text = text

# --- Main input ---------------------------------------------------------------
article = st.text_area(
    "Article or headline text",
    key="article_text",
    height=200,
    placeholder="Paste the news text here…",
)

analyze = st.button("Analyze", type="primary", use_container_width=True)

# --- Prediction ---------------------------------------------------------------
if analyze:
    if not article.strip():
        st.warning("Please enter or select some text first.")
    else:
        p_fake = predict(model, article)
        is_fake = p_fake >= threshold
        confidence = p_fake if is_fake else 1 - p_fake

        if is_fake:
            st.error(f"### 🚨 Likely FAKE")
        else:
            st.success(f"### ✅ Likely REAL")

        c1, c2 = st.columns(2)
        c1.metric("Confidence", f"{confidence * 100:.1f}%")
        c2.metric("P(fake)", f"{p_fake:.3f}", help=f"Flagged as fake when ≥ {threshold:.2f}")

        st.progress(p_fake, text=f"Fake-news probability: {p_fake:.0%}")
        st.caption(
            "This model is a triage assistant: it ranks content for human reviewers and "
            "is not a final verdict on truthfulness."
        )
