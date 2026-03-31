import streamlit as st
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header, info_box, metric_card, empty_state, PRIMARY,
)
from utils.model_loader import (
    load_dataset, load_insurer_summaries, load_summarizer,
)

def run_abstractive_summary(text: str, summarizer) -> str:
    """Utilise DistilBART comme dans le Notebook Step 2."""
    word_count = len(text.split())
    if word_count < 20:
        return text
    
    # Paramètres optimisés selon le notebook
    max_l = min(150, max(30, word_count // 2))
    try:
        # Nettoyage rapide pour éviter les erreurs d'encodage lors de l'inférence
        clean_input = text.encode("utf-8", errors="ignore").decode("utf-8")
        result = summarizer(
            clean_input,
            min_length=15,
            max_length=max_l,
            truncation=True,
            do_sample=False
        )
        return result[0]["summary_text"].strip()
    except Exception as e:
        return f"Summarization error: {e}"

def run_translation(text: str) -> str:
    """Traduction FR->EN (Helsinki-NLP) avec gestion d'erreurs d'encodage."""
    try:
        from transformers import pipeline
        # On force l'utilisation du CPU pour éviter les crashs de mémoire sur Streamlit
        translator = pipeline("translation", model=cfg.TRANSLATION_MODEL, device=-1)
        # On limite à 512 tokens pour éviter les erreurs de séquence trop longue
        result = translator(text[:1000], max_length=512, truncation=True)
        return result[0]["translation_text"]
    except Exception as e:
        return f"Translation error: {e}"


def render():
    page_header("", "Summarization & Translation")

    # Chargement des données avec gestion d'erreur robuste
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    mode = st.radio(
        "Mode",
        ["Enter custom text", "Select a review"],
        horizontal=True, label_visibility="collapsed"
    )
    st.divider()

    if mode == "Enter custom text":
        text_input = st.text_area("French or English text", height=200, placeholder="Paste review here...")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Summarize (DistilBART)", type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    model = load_summarizer()
                    if model:
                        res = run_abstractive_summary(text_input, model)
                        st.subheader("Summary")
                        st.success(res)
        with c2:
            if st.button("Translate FR → EN", use_container_width=True):
                with st.spinner("Translating..."):
                    res = run_translation(text_input)
                    st.subheader("Translation")
                    st.info(res)

    elif mode == "Select a review":
        if df.empty:
            empty_state("No data available.")
            return

        # Filtres simples pour éviter les erreurs d'index
        ins_list = ["All"] + sorted(df[cfg.COL_INSURER].unique().tolist())
        sel_ins = st.selectbox("Insurer", ins_list)
        
        filtered = df if sel_ins == "All" else df[df[cfg.COL_INSURER] == sel_ins]
        
        if not filtered.empty:
            # On prend les 50 premiers pour la performance
            options = filtered.head(50)
            choice = st.selectbox("Pick a review", range(len(options)), 
                                 format_func=lambda i: f"{options.iloc[i][cfg.COL_TEXT_EN][:60]}...")
            
            row = options.iloc[choice]
            st.markdown(f"**Original:** {row[cfg.COL_TEXT_EN]}")
            
            # Affichage du résumé extractif déjà présent (avis_summary dans ton notebook)
            st.markdown("---")
            st.markdown("#### Extractive Summary (TF-IDF)")
            # On utilise .get() pour éviter le KeyError si la colonne manque
            ext_summary = row.get("avis_summary", "No summary pre-generated.")
            st.info(ext_summary)