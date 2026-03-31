import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header, info_box, metric_card, empty_state, PRIMARY,
)
from utils.model_loader import (
    load_dataset, load_tfidf_vectorizers,
    load_word2vec_models, build_sentence_vectors,
)
from utils.retrieval import (
    tfidf_search, bm25_search, embedding_search, build_rag_context,
)


def call_llm(prompt: str, api_key: str, base_url: str, model: str, max_tokens: int) -> str:
    """
    Call an OpenAI-compatible LLM endpoint.
    Works with OpenAI, Azure OpenAI, Ollama, or any compatible provider.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert analyst of French insurance customer reviews. "
                        "Answer the user's question truthfully and concisely using ONLY "
                        "the provided context. If the context does not contain enough "
                        "information, say so clearly. Do not make up facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except ImportError:
        return (
            "The `openai` Python package is not installed. "
            "Run `pip install openai` and restart the app."
        )
    except Exception as e:
        return f"LLM call failed: {e}"


def build_rag_prompt(question: str, context: str) -> str:
    """Build the RAG prompt injecting retrieved context."""
    return f"""You are analysing customer reviews of French insurance companies.

RETRIEVED CONTEXT (use ONLY this to answer):
{context}

USER QUESTION:
{question}

Answer the question based solely on the context above. Be concise and specific.
If the answer is not in the context, say: "The provided reviews do not contain enough information to answer this question."
"""


def render():
    page_header(
        "RAG — Retrieval-Augmented Generation",
        "Ask a natural language question and get a grounded answer backed by real reviews.",
    )

    df          = load_dataset()
    tfidf_en, _ = load_tfidf_vectorizers()
    w2v_en, _   = load_word2vec_models()

    if df.empty:
        info_box("Dataset not loaded. Please run the notebooks first.", kind="error")
        return

    with st.expander("LLM Configuration", expanded=not cfg.OPENAI_API_KEY):
        st.markdown("Configure the LLM endpoint. Works with OpenAI, Ollama, or any compatible API.")

        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input(
                "API Key",
                value=cfg.OPENAI_API_KEY or "",
                type="password",
                help="Your OpenAI (or compatible) API key. Leave empty for local Ollama.",
            )
        with col2:
            api_base = st.text_input(
                "API Base URL",
                value=cfg.OPENAI_API_BASE,
                help="For Ollama: http://localhost:11434/v1",
            )

        col3, col4 = st.columns(2)
        with col3:
            model_id = st.text_input("Model", value=cfg.RAG_LLM_MODEL)
        with col4:
            max_tokens = st.slider("Max tokens", 100, 1024, cfg.RAG_MAX_TOKENS, 50)

        top_k = st.slider(
            "Number of retrieved chunks (k)", min_value=1, max_value=15,
            value=cfg.RAG_TOP_K, step=1,
        )
        retrieval_method = st.selectbox(
            "Retrieval method",
            ["TF-IDF Cosine", "BM25", "Embedding (Word2Vec)"],
        )

    with st.expander("Filter the knowledge base", expanded=False):
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

    st.markdown("#### Ask a question about the insurance reviews")

    example_questions = {
        "— choose an example —": "",
        "What do customers complain about most?":
            "What are the most common complaints customers have about their insurance?",
        "Price concerns":
            "What do customers say about price increases and value for money?",
        "Claims process":
            "How do customers describe the claims handling process?",
        "Best insurers":
            "Which insurers receive the most positive reviews and why?",
        "Customer service":
            "How is the quality of customer service described by reviewers?",
    }

    selected_ex = st.selectbox("Load an example question", list(example_questions.keys()))
    question    = st.text_input(
        "Your question",
        value=example_questions[selected_ex],
        placeholder="Ask anything about the insurance reviews corpus …",
        label_visibility="collapsed",
    )

    go = st.button("Ask", type="primary", use_container_width=True)

    if go:
        if len(question.strip()) < 5:
            info_box("Please enter a question.", kind="warning")
            return

        if not api_key and "ollama" not in api_base.lower() and "localhost" not in api_base.lower():
            info_box(
                "Please provide an API key, or set the base URL to a local Ollama endpoint.",
                kind="warning",
            )
            return

        # Step 1: retrieve
        with st.spinner(f"Retrieving top-{top_k} relevant documents …"):
            try:
                if retrieval_method == "TF-IDF Cosine":
                    retrieved = tfidf_search(
                        question, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k=top_k
                    )
                elif retrieval_method == "BM25":
                    retrieved = bm25_search(
                        question, corpus, cfg.COL_TEXT_EN, top_k=top_k
                    )
                else:
                    if w2v_en is None:
                        info_box("Word2Vec model not found. Falling back to TF-IDF.", kind="warning")
                        retrieved = tfidf_search(question, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k)
                    else:
                        corpus_vecs = build_sentence_vectors(corpus)
                        idf_map = {}
                        if tfidf_en is not None:
                            idf_map = dict(zip(tfidf_en.get_feature_names_out(), tfidf_en.idf_))
                        if corpus_vecs is not None:
                            retrieved = embedding_search(
                                question, corpus, corpus_vecs, w2v_en, idf_map, top_k=top_k
                            )
                        else:
                            retrieved = tfidf_search(question, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k)

            except Exception as e:
                info_box(f"Retrieval failed: {e}", kind="error")
                return

        if retrieved.empty:
            empty_state("No relevant documents found for this query.")
            return

        # Step 2: build context
        context = build_rag_context(retrieved, cfg.COL_TEXT_EN, max_words=800)
        prompt  = build_rag_prompt(question, context)

        # Step 3: call LLM
        with st.spinner("Generating answer with LLM …"):
            answer = call_llm(
                prompt,
                api_key  = api_key or "no-key",
                base_url = api_base,
                model    = model_id,
                max_tokens = max_tokens,
            )

        st.divider()
        st.markdown("### Generated Answer")
        st.markdown(
            f"""
            <div style="
                background:#E8F5E9;
                border-left:4px solid #43A047;
                padding:1rem 1.2rem;
                border-radius:8px;
                font-size:1rem;
                line-height:1.7;
            ">
            {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Retrieved docs", str(len(retrieved)), color=PRIMARY)
        with col2:
            avg_r = retrieved[cfg.COL_RATING].mean() if cfg.COL_RATING in retrieved.columns else 0
            metric_card("Avg. doc rating", f"⭐ {avg_r:.1f}", color=PRIMARY)
        with col3:
            metric_card("LLM model", model_id, color=PRIMARY)

        # Source documents
        st.markdown("### Source Documents")
        st.caption("These are the retrieved reviews used as context for the LLM.")

        for i, (_, row) in enumerate(retrieved.iterrows()):
            stars   = "⭐" * int(row.get(cfg.COL_RATING, 0))
            insurer = str(row.get(cfg.COL_INSURER, "—"))
            score   = row.get("score", 0)
            theme   = str(row.get(cfg.COL_THEME, "—")) if cfg.COL_THEME in row else "—"
            text    = str(row.get(cfg.COL_TEXT_EN, ""))

            with st.expander(
                f"Doc {i+1} | [{insurer}] {stars} | Score: {score:.3f} | {theme}"
            ):
                st.write(text)

        # Show the prompt (for transparency)
        with st.expander("View full RAG prompt sent to LLM"):
            st.code(prompt, language="text")
