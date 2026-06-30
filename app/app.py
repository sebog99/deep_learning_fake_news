"""
Fake News Detector — Streamlit frontend.

Loads the end-to-end model exported by the notebook (`models/fake_news_model.keras`),
which contains the text-vectorization step inside it, so the app passes raw text in
and reads a fake-probability out.

    streamlit run app/app.py
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


# --- Standardization function (must match the notebook's exactly) -------------
# Registered and passed via custom_objects so the saved model, which references it
# inside its TextVectorization layer, can be deserialized.
@keras.utils.register_keras_serializable(package="fake_news")
def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_tag = tf.strings.regex_replace(lowercase, "<[^>]+>", "")
    no_punct = tf.strings.regex_replace(no_tag, "[%s]" % re.escape(string.punctuation), "")
    # Must match the notebook exactly: drop non-ASCII, then collapse whitespace.
    ascii_only = tf.strings.regex_replace(no_punct, r"[^\x00-\x7f]+", "")
    return tf.strings.regex_replace(ascii_only, r"\s+", " ")


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
           "Paste the **full article text** below — the model was trained on complete "
           "articles, so a short headline or a sentence or two is unreliable.")

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
        "WASHINGTON - The Federal Reserve left interest rates unchanged on Wednesday and "
        "signaled it was in no hurry to adjust borrowing costs, citing steady economic growth "
        "and a labor market that remains resilient. In a statement following its two-day "
        "meeting, the central bank said inflation had eased over the past year but remained "
        "somewhat above its 2 percent target. Officials noted that consumer spending had "
        "continued to expand at a solid pace, while business investment showed signs of "
        "moderating. The Fed chair said policymakers would continue to assess incoming data, "
        "the evolving outlook, and the balance of risks before making any changes. Economists "
        "said the decision was widely expected and that the central bank was likely to hold "
        "rates steady through the next quarter as it waited for clearer evidence on the "
        "direction of prices and employment. Markets showed little reaction, with major stock "
        "indexes ending the session nearly flat and government bond yields holding steady.",
    "🚨 Looks like clickbait":
        "SHOCKING the government is HIDING the TRUTH about the economy and you WONT "
        "believe what happens next. Share this before they DELETE it forever - the "
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
    "Article text",
    key="article_text",
    height=200,
    placeholder="Paste the full article text here…",
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
