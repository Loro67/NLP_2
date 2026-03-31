import streamlit as st
import sys
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header, info_box, metric_card, empty_state, PRIMARY, ACCENT_GREEN
)
from utils.model_loader import (
    load_dataset, load_tfidf_vectorizers,
    load_word2vec_models, build_sentence_vectors,
)
from utils.retrieval import (
    tfidf_search, bm25_search, embedding_search, build_rag_context,
)

# Configuration API
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def call_gemini_rag(prompt: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

def build_rag_prompt(question: str, context: str) -> str:
    return f"""You are an expert analyst of French insurance customer reviews.
Use the following retrieved reviews to answer the user's question truthfully and concisely.

GUIDELINES:
- Use ONLY the provided context.
- If the context doesn't contain the answer, say "The provided reviews do not contain enough information."
- Do not make up facts.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""


def _display_result(answer: str, retrieved):
    """Rendu visuel de la réponse et des métriques."""
    st.divider()
    
    st.markdown("### Result")
    st.markdown(
        f"""
        <div style="
            background:#E8F5E9;
            border-left:5px solid {ACCENT_GREEN};
            padding:1.5rem;
            border-radius:10px;
            font-size:1.05rem;
            line-height:1.6;
            color: #1B5E20;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Métriques en dessous de la réponse
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Reviews Analyzed", str(len(retrieved)), color=PRIMARY)
    with m2:
        avg_r = retrieved[cfg.COL_RATING].mean()
        metric_card("Avg. Context Rating", f"{avg_r:.1f} ⭐", color=PRIMARY)
    with m3:
        metric_card("Model", "Gemini 2.5 Flash", color="#4285F4")

    # Sources détaillées
    with st.expander("View Source Documents (Evidence)"):
        for i, (_, row) in enumerate(retrieved.iterrows()):
            stars = "⭐" * int(row.get(cfg.COL_RATING, 0))
            insurer = row.get(cfg.COL_INSURER, "Unknown")
            st.markdown(f"**Doc {i+1} | {insurer} | {stars}**")
            st.caption(row.get(cfg.COL_TEXT_EN, ""))
            st.divider()

def render():
    page_header(
        "", "RAG Analysis",
    )

    df = load_dataset()
    tfidf_en, _ = load_tfidf_vectorizers()
    w2v_en, _ = load_word2vec_models()

    if df.empty:
        info_box("Dataset not loaded.", kind="error")
        return

    st.markdown("#### Retrieval Configuration")
    
    # On crée deux colonnes pour les paramètres
    c1, c2 = st.columns(2)
    
    with c1:
        with st.container(border=True):
            retrieval_method = st.selectbox(
                "Search Method",
                ["BM25 (Recommended)", "TF-IDF Cosine", "Embedding (Word2Vec)"],
                help="BM25 est excellent pour les mots-clés, l'Embedding pour le sens global."
            )
    
    with c2:
        with st.container(border=True):
            top_k = st.slider(
                "Reviews to Analyze (k)", 
                min_value=5, 
                max_value=200, 
                value=20, 
                step=5,
                help="Plus le nombre est élevé, plus Gemini a de contexte pour répondre."
            )

    with st.expander("Filter the knowledge base (optional)", expanded=False):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            insurers = ["All"] + sorted(df[cfg.COL_INSURER].dropna().unique().tolist())
            f_insurer = st.selectbox("Limit to insurer", insurers)
        with fcol2:
            products = ["All"] + sorted(df[cfg.COL_PRODUCT].dropna().unique().tolist())
            f_product = st.selectbox("Limit to product", products)

    corpus = df.copy()
    if f_insurer != "All":
        corpus = corpus[corpus[cfg.COL_INSURER] == f_insurer]
    if f_product != "All":
        corpus = corpus[corpus[cfg.COL_PRODUCT] == f_product]

    st.divider()

    st.markdown("#### Ask a question about the reviews")
    
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the main points of dissatisfaction regarding the price?",
        label_visibility="collapsed"
    )

    if st.button("Generate Answer", type="primary", use_container_width=True):
        if len(question.strip()) < 5:
            st.warning("Please enter a question (at least 5 characters).")
            return

        # 1. Retrieval
        with st.spinner(f"Searching {len(corpus)} reviews via {retrieval_method}..."):
            try:
                if "BM25" in retrieval_method:
                    retrieved = bm25_search(question, corpus, cfg.COL_TEXT_EN, top_k=top_k)
                elif "TF-IDF" in retrieval_method:
                    retrieved = tfidf_search(question, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k=top_k)
                else:
                    corpus_vecs = build_sentence_vectors(corpus)
                    idf_map = dict(zip(tfidf_en.get_feature_names_out(), tfidf_en.idf_)) if tfidf_en else {}
                    retrieved = embedding_search(question, corpus, corpus_vecs, w2v_en, idf_map, top_k=top_k)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                return

        if retrieved.empty:
            empty_state("No relevant reviews found for this filter/query.")
            return

        # 2. RAG Flow
        with st.spinner("The model is generating the answer"):
            # On construit le contexte (on peut monter la limite de mots car Gemini encaisse bien)
            context = build_rag_context(retrieved, cfg.COL_TEXT_EN, max_words=2000)
            prompt = build_rag_prompt(question, context)
            answer = call_gemini_rag(prompt)

        # 3. Display Result
        _display_result(answer, retrieved)