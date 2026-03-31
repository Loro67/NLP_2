import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header, metric_card, probability_chart, info_box, empty_state,
    star_rating_badge, PRIMARY, ACCENT_GREEN, ACCENT_RED,
)
from utils.model_loader import load_classifier, load_bert_sentiment_pipeline
from utils.preprocessing import clean_text, prepare_for_tfidf


def predict_with_lr(text: str, tfidf, clf):
    """Run TF-IDF + LR inference on raw text. Returns (label_0idx, probs array)."""
    cleaned = prepare_for_tfidf(text)
    vec     = tfidf.transform([cleaned])
    probs   = clf.predict_proba(vec)[0]
    pred    = int(np.argmax(probs))
    return pred, probs


def predict_with_bert(text: str, pipe):
    """Run BERT multilingual sentiment inference. Returns (label_0idx, probs array)."""
    truncated = " ".join(text.split()[:400])
    result    = pipe(truncated, truncation=True, max_length=512)[0]
    label_str = result["label"]
    score     = result["score"]

    # BERT model outputs "1 star" … "5 stars"
    label_idx = int(label_str.split()[0]) - 1   # → 0…4

    # Approximate probability distribution (peak at predicted, distribute rest)
    probs = np.full(5, (1 - score) / 4)
    probs[label_idx] = score
    return label_idx, probs


def render():
    page_header(
        "", "Prediction",
    )

    with st.spinner("Loading classifier …"):
        tfidf, clf = load_classifier()

    model_available = tfidf is not None and clf is not None
    bert_pipe       = None

    if not model_available:
        info_box(
            "Local TF-IDF + LR classifier not found. "
            "Falling back to <b>BERT multilingual</b> sentiment model. "
            "Re-train and export <code>lr_classifier_en.pkl</code> from notebook 5 "
            "to use the faster local model.",
            kind="warning",
        )
        with st.spinner("Loading BERT fallback model …"):
            bert_pipe = load_bert_sentiment_pipeline()

    if not model_available and bert_pipe is None:
        info_box(
            "No classifier available. Please run notebook 5 and export the model.",
            kind="error",
        )
        return

    # ── Input ─────────────────────────────────────────────────────────────────
    st.markdown("#### Enter a review")
    example_texts = {
        "— choose an example —": "",
        "Negative (1★)": (
            "Très déçu par ce service. Ma réclamation a été refusée sans raison valable. "
            "Le service client ne répond pas aux emails. Je déconseille fortement cette assurance."
        ),
        "Neutral (3★)": (
            "L'assurance est correcte dans l'ensemble. Les remboursements prennent un peu de temps "
            "mais arrivent finalement. Le prix a augmenté cette année ce qui est dommage."
        ),
        "Positive (5★)": (
            "Excellent service ! Ma réclamation a été traitée en 48 heures. "
            "Le conseiller était très professionnel et à l'écoute. Je recommande vivement."
        ),
    }

    col_ex, col_lang = st.columns([3, 1])
    with col_ex:
        selected = st.selectbox("Load an example", list(example_texts.keys()))
    with col_lang:
        lang = st.radio("Language hint", ["French", "English"], horizontal=True)

    default_text = example_texts[selected]
    user_text    = st.text_area(
        "Review text",
        value=default_text,
        height=150,
        placeholder="Paste or type your review here …",
        label_visibility="collapsed",
    )

    st.markdown(
        f"<small style='color:#888;'>{len(user_text.split())} words</small>",
        unsafe_allow_html=True,
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    btn = st.button("Predict Rating", type="primary", use_container_width=True)

    if btn:
        if len(user_text.strip()) < cfg.MIN_REVIEW_CHARS:
            info_box(f"Please enter at least {cfg.MIN_REVIEW_CHARS} characters.", kind="warning")
            return

        with st.spinner("Running inference …"):
            try:
                if model_available:
                    pred_idx, probs = predict_with_lr(user_text, tfidf, clf)
                    model_name = "TF-IDF + Logistic Regression"
                else:
                    pred_idx, probs = predict_with_bert(user_text, bert_pipe)
                    model_name = "BERT multilingual (fallback)"

                star      = pred_idx + 1
                confidence = float(probs[pred_idx])

                # Sentiment bucket
                if star <= 2:
                    sentiment = "Negative"
                    sent_color = ACCENT_RED
                elif star == 3:
                    sentiment = "Neutral"
                    sent_color = "#FB8C00"
                else:
                    sentiment = "Positive"
                    sent_color = ACCENT_GREEN

            except Exception as e:
                info_box(f"Prediction error: {e}", kind="error")
                return

        st.divider()
        st.markdown("### Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Predicted Rating", star_rating_badge(star), color=PRIMARY)
        with col2:
            metric_card("Confidence", f"{confidence*100:.1f}%", color=PRIMARY)
        with col3:
            metric_card("Sentiment", sentiment, color=sent_color)

        # Probability distribution chart
        star_labels = [f"{i} star{'s' if i > 1 else ''}" for i in range(1, 6)]
        probability_chart(star_labels, probs.tolist(), "Class Probability Distribution")

        # Model info
        with st.expander("Model details"):
            st.markdown(f"**Model:** {model_name}")
            st.markdown("**Pipeline:** clean text → tokenize → TF-IDF transform → LR predict")
            st.markdown("**Training task:** 5-class star rating prediction (0-indexed classes 0–4)")
            prob_df_data = {
                "Rating": [f"{i} stars" for i in range(1, 6)],
                "Probability": [f"{p:.4f}" for p in probs],
                "Percentage":  [f"{p*100:.2f}%" for p in probs],
            }
            import pandas as pd
            st.dataframe(pd.DataFrame(prob_df_data), hide_index=True, use_container_width=True)
