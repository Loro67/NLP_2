import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header, info_box, results_table, empty_state, metric_card,
    PRIMARY, ACCENT_GREEN,
)
from utils.model_loader import (
    load_dataset, load_tfidf_vectorizers,
    load_word2vec_models, build_sentence_vectors,
)
from utils.retrieval import tfidf_search, bm25_search, embedding_search
from utils.preprocessing import truncate_text


def render():
    page_header(
        "","Information Retrieval"
    )
    df            = load_dataset()
    tfidf_en, _   = load_tfidf_vectorizers()
    w2v_en, _     = load_word2vec_models()

    if df.empty:
        info_box("Dataset not loaded. Please run the notebooks first.", kind="error")
        return

    col_query, col_method = st.columns([3, 1])
    with col_query:
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g. 'refused claim no response' or 'price increase expensive'",
            label_visibility="collapsed",
        )
    with col_method:
        method = st.selectbox(
            "Search method",
            ["TF-IDF Cosine", "BM25", "Embedding (Word2Vec)"],
            label_visibility="collapsed",
        )

    # Filters
    with st.expander("Filters", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            insurers = ["All"] + sorted(df[cfg.COL_INSURER].dropna().unique().tolist())
            f_insurer = st.selectbox("Insurer", insurers)
        with fcol2:
            ratings = ["All"] + [str(i) for i in range(1, 6)]
            f_rating = st.selectbox("Min. Star Rating", ratings)
        with fcol3:
            top_k = st.slider("Top-K results", min_value=3, max_value=30,
                               value=cfg.TOP_K_RETRIEVAL, step=1)

        if cfg.COL_THEME in df.columns:
            themes = ["All"] + sorted(df[cfg.COL_THEME].dropna().unique().tolist())
            f_theme = st.selectbox("Theme", themes)
        else:
            f_theme = "All"

    # Apply pre-filters to corpus
    corpus = df.copy()
    if f_insurer != "All":
        corpus = corpus[corpus[cfg.COL_INSURER] == f_insurer]
    if f_rating != "All":
        corpus = corpus[corpus[cfg.COL_RATING] >= int(f_rating)]
    if f_theme != "All" and cfg.COL_THEME in corpus.columns:
        corpus = corpus[corpus[cfg.COL_THEME] == f_theme]

    st.markdown(
        f"<small style='color:#888;'>Searching in <b>{len(corpus):,}</b> documents</small>",
        unsafe_allow_html=True,
    )

    if st.button("Search", type="primary", use_container_width=True):
        if not query.strip():
            info_box("Please enter a search query.", kind="warning")
            return

        if corpus.empty:
            empty_state("No documents match the selected filters.")
            return

        with st.spinner(f"Running {method} search …"):
            try:
                if method == "TF-IDF Cosine":
                    results = tfidf_search(
                        query, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k=top_k
                    )

                elif method == "BM25":
                    results = bm25_search(
                        query, corpus, cfg.COL_TEXT_EN, top_k=top_k
                    )

                else:  # Embedding
                    if w2v_en is None:
                        info_box(
                            "Word2Vec model not found. Falling back to TF-IDF search.",
                            kind="warning",
                        )
                        results = tfidf_search(query, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k)
                    else:
                        # Build or load sentence vectors
                        corpus_vecs = build_sentence_vectors(corpus)
                        if corpus_vecs is None:
                            info_box(
                                "Could not build sentence vectors. Falling back to TF-IDF.",
                                kind="warning",
                            )
                            results = tfidf_search(query, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k)
                        else:
                            idf_map = {}
                            if tfidf_en is not None:
                                idf_map = dict(
                                    zip(tfidf_en.get_feature_names_out(), tfidf_en.idf_)
                                )
                            results = embedding_search(
                                query, corpus, corpus_vecs, w2v_en, idf_map, top_k=top_k
                            )

            except Exception as e:
                info_box(f"Search error: {e}", kind="error")
                return

        st.divider()

        if results.empty:
            empty_state("No results found. Try a different query or method.")
            return

        st.markdown(f"### Top {len(results)} Results")

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Results found", str(len(results)), color=PRIMARY)
        with col2:
            avg_rating = results[cfg.COL_RATING].mean() if cfg.COL_RATING in results else 0
            metric_card("Avg. rating", f"{'⭐' * round(avg_rating)}", color=PRIMARY)
        with col3:
            avg_score = results["score"].mean() if "score" in results else 0
            metric_card("Avg. similarity", f"{avg_score:.3f}", color=PRIMARY)

        # Interactive table
        display_cols = [cfg.COL_INSURER, cfg.COL_RATING, "score"]
        if cfg.COL_THEME in results.columns:
            display_cols.insert(2, cfg.COL_THEME)
        if cfg.COL_PRODUCT in results.columns:
            display_cols.insert(2, cfg.COL_PRODUCT)

        results_table(
            results,
            columns=display_cols,
            column_labels={
                cfg.COL_INSURER: "Insurer",
                cfg.COL_RATING:  "Rating",
                cfg.COL_THEME:   "Theme",
                cfg.COL_PRODUCT: "Product",
                "score":         "Relevance Score",
            },
        )

        # Expandable review cards
        st.markdown("### Review Details")
        for i, (_, row) in enumerate(results.iterrows()):
            stars = "⭐" * int(row.get(cfg.COL_RATING, 0))
            score = row.get("score", 0)
            insurer = str(row.get(cfg.COL_INSURER, "—"))
            theme   = str(row.get(cfg.COL_THEME, "—")) if cfg.COL_THEME in row else "—"

            with st.expander(
                f"**#{i+1}** [{insurer}] {stars} | Score: {score:.3f} | Theme: {theme}"
            ):
                text = str(row.get(cfg.COL_TEXT_EN, ""))
                st.markdown(text)

                if str(row.get(cfg.COL_TEXT_FR, "")).strip():
                    st.markdown("---")
                    st.markdown("**Original French:**")
                    st.markdown(str(row.get(cfg.COL_TEXT_FR, "")))

                summary = str(row.get(cfg.COL_SUMMARY, ""))
                if summary and summary != "nan":
                    st.info(f"**Extractive summary:** {summary}")
