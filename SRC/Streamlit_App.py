from pathlib import Path

import pandas as pd
import streamlit as st

from Inference import load_model_bundle, predict_sentiment


st.set_page_config(
    page_title="Sentiment Feedback Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 183, 77, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(76, 175, 80, 0.14), transparent 24%),
            linear-gradient(180deg, #fffdf8 0%, #f6f1e8 100%);
        color: #1c1c1c;
    }
    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border: 1px solid rgba(28, 28, 28, 0.08);
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 18px 60px rgba(24, 24, 24, 0.08);
        backdrop-filter: blur(12px);
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        color: #8c6239;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .title {
        font-size: clamp(2.2rem, 4vw, 4.2rem);
        line-height: 1;
        margin: 0;
        font-weight: 800;
        color: #181512;
    }
    .subtitle {
        font-size: 1.05rem;
        color: rgba(24, 21, 18, 0.82);
        max-width: 62ch;
        margin-top: 0.85rem;
    }
    .panel {
        border-radius: 24px;
        border: 1px solid rgba(28, 28, 28, 0.08);
        background: rgba(255, 255, 255, 0.86);
        box-shadow: 0 14px 36px rgba(24, 24, 24, 0.06);
        padding: 1.25rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,244,229,0.95));
        border: 1px solid rgba(28, 28, 28, 0.08);
    }
    .muted {
        color: rgba(24, 21, 18, 0.68);
        font-size: 0.94rem;
    }
    .pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
        background: #f0ece6;
        color: #34281f;
    }
    .positive {
        background: linear-gradient(135deg, #dff5e5, #b9e6c2);
        color: #0d4d21;
    }
    .negative {
        background: linear-gradient(135deg, #ffe2e2, #f6b8b8);
        color: #7e1e1e;
    }
</style>
"""


SAMPLES = [
    "The product feels thoughtfully designed and works smoothly.",
    "I was disappointed by the slow response and confusing layout.",
    "The experience was okay, but nothing really stood out.",
]


@st.cache_resource(show_spinner=False)
def warm_up_model():
    return load_model_bundle()


def render_prediction(text: str):
    prediction = predict_sentiment(text, return_scores=True)
    render_prediction_result(prediction)
    return prediction


def render_prediction_result(prediction):
    positive_score = prediction["probabilities"].get("Positive", 0.0)
    negative_score = prediction["probabilities"].get("Negative", 0.0)
    confidence = prediction["confidence"]
    label = prediction["label"]

    accent_class = "positive" if label == "Positive" else "negative"

    col_left, col_right = st.columns([1.15, 0.85], gap="large")
    with col_left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown(f"<div class='pill {accent_class}'>{label} sentiment</div>", unsafe_allow_html=True)
        st.markdown(f"### Confidence: {confidence:.1%}")
        st.progress(confidence)
        st.caption("The score reflects the model's top-class probability, not a guarantee.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("**Score breakdown**")
        st.metric("Positive", f"{positive_score:.1%}")
        st.metric("Negative", f"{negative_score:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    if "sentiment_text" not in st.session_state:
        st.session_state.sentiment_text = ""
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    with st.sidebar:
        st.markdown("## Sentiment Feedback Studio")
        st.write("A public-ready interface for testing the trained DistilBERT sentiment model.")
        st.markdown("### Quick tips")
        st.write("Use a short sentence, review, or product feedback for best results.")
        st.write("Long paragraphs are supported, but the model was trained on review-style text.")
        st.markdown("### Example prompts")
        for sample in SAMPLES:
            st.markdown(f"<span class='pill'>{sample}</span>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='hero'>
            <div class='eyebrow'>NLP sentiment interface</div>
            <h1 class='title'>Type any sentence and get instant sentiment feedback.</h1>
            <p class='subtitle'>Type a sentence, phrase, or short review and the app will return the predicted sentiment with a confidence breakdown.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        warm_up_model()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### Try it now")
        st.caption("Type any sentence or short review here, then click Analyze sentiment.")
        st.text_area(
            "Type your sentence here",
            height=180,
            placeholder="Enter a sentence, review, or short paragraph...",
            label_visibility="visible",
            key="sentiment_text",
        )
        submitted = st.button("Analyze sentiment")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            try:
                current_text = st.session_state.sentiment_text.strip()
                if not current_text:
                    raise ValueError("Text cannot be empty.")
                st.session_state.last_prediction = render_prediction(current_text)
                st.session_state.sentiment_text = ""
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### What this app gives you")
        st.write("- A single-screen, distraction-free layout")
        st.write("- Clear prediction confidence and label breakdown")
        st.write("- Fast reuse of the trained model without retraining")
        st.markdown("### Suggested usage")
        st.write("Ask about product feedback, support messages, or short review snippets.")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.last_prediction is not None:
        st.markdown("### Latest result")
        render_prediction_result(st.session_state.last_prediction)

    upload = st.file_uploader("Optional CSV batch preview", type=["csv"])
    if upload is not None:
        frame = pd.read_csv(upload)
        text_columns = [column for column in frame.columns if frame[column].dtype == object]
        if not text_columns:
            st.warning("No text columns found in the uploaded file.")
            return

        selected_column = st.selectbox("Text column", text_columns)
        preview = frame.head(25).copy()
        preview["prediction"] = preview[selected_column].fillna("").astype(str).apply(
            lambda value: predict_sentiment(value) if value.strip() else ""
        )
        st.dataframe(preview, use_container_width=True)


if __name__ == "__main__":
    main()