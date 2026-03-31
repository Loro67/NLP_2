from __future__ import annotations
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config as cfg


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the review dataset with strict format handling."""
    path = cfg.CLEANED_DATA
    # Système de fallback si le fichier principal n'existe pas
    if not path.exists():
        for fallback in ["reviews_step2.parquet"]:
            p = cfg.DATA_DIR / fallback
            if p.exists():
                path = p
                break
        else:
            st.warning(f"No dataset found at {path}")
            return pd.DataFrame()

    try:
        # 1. Si c'est un fichier Parquet
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
    
        # --- Nettoyage post-chargement ---
        for col in [cfg.COL_TEXT_EN, cfg.COL_TEXT_FR, cfg.COL_INSURER]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        
        # S'assurer que la colonne de résumé du Notebook Step 2 existe
        if "avis_summary" in df.columns:
            df[cfg.COL_SUMMARY] = df["avis_summary"]
        
        return df

    except Exception as e:
        st.error(f"Critical error loading {path.name}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_insurer_summaries() -> pd.DataFrame:
    """Load per-insurer abstractive summaries with robust delimiter/encoding detection."""
    path = cfg.INSURER_SUMMARIES
    if not path.exists():
        return pd.DataFrame(columns=["assureur", "summary_overall",
                                      "summary_complaints", "avg_rating", "n_reviews"])
    
    # Liste des encodages et séparateurs à tester
    encodings = ['utf-8', 'latin-1']
    separators = [',', ';', '\t'] # Virgule, Point-virgule, Tabulation

    for enc in encodings:
        for sep in separators:
            try:
                # On essaie de lire un petit morceau pour vérifier la structure
                df = pd.read_csv(path, encoding=enc, sep=sep)
                
                # Vérification critique : est-ce que les colonnes attendues sont là ?
                # Si le séparateur est mauvais, on aura souvent une seule colonne avec un nom bizarre
                if "assureur" in df.columns or "summary_overall" in df.columns:
                    return df
            except Exception:
                continue
                
    # Si tout a échoué, on essaie une lecture brute très permissive
    try:
        return pd.read_csv(path, sep=None, engine='python', encoding='latin-1')
    except Exception as e:
        st.error(f"Failed to parse {path.name}: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_tfidf_vectorizers() -> Tuple[Optional[object], Optional[object]]:
    """Load fitted TF-IDF vectorizers (EN and FR) from Step 3."""
    tfidf_en = tfidf_fr = None
    for attr, path in [("tfidf_en", cfg.TFIDF_EN_PKL),
                        ("tfidf_fr", cfg.TFIDF_FR_PKL)]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                if attr == "tfidf_en":
                    tfidf_en = obj
                else:
                    tfidf_fr = obj
            except Exception:
                pass
    return tfidf_en, tfidf_fr

@st.cache_resource(show_spinner=False)
def load_classifier() -> Tuple[Optional[object], Optional[object]]:
    """
    Load the TF-IDF + LR classifier saved from Step 5.

    Returns (tfidf, classifier) or (None, None) if not found.
    The app will fall back to a live HuggingFace BERT pipeline if missing.
    """
    # Try dedicated classifier pkl first
    for tfidf_path, clf_path in [
        (cfg.LR_TFIDF_EN, cfg.LR_CLASSIFIER_EN),
        # Backward-compatibility names exported by older notebooks
        (cfg.MODEL_DIR / "tfidf_en.pkl", cfg.MODEL_DIR / "lr_rating_en.pkl"),
    ]:
        if tfidf_path.exists() and clf_path.exists():
            try:
                with open(tfidf_path, "rb") as f:
                    tfidf = pickle.load(f)
                with open(clf_path, "rb") as f:
                    clf = pickle.load(f)
                return tfidf, clf
            except Exception:
                pass

    # Fallback: use the Step-3 TF-IDF + rebuild a thin LR from dataset
    return None, None


@st.cache_resource(show_spinner=False)
def load_bert_sentiment_pipeline():
    """Load the BERT multilingual sentiment pipeline (fallback classifier)."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model=cfg.BERT_SENTIMENT,
            framework="pt",
            device=-1,   # CPU
        )
        return pipe
    except Exception as e:
        st.warning(f"Could not load BERT sentiment model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Load the DistilBART summarization pipeline."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "summarization",
            model=cfg.SUMMARIZER_MODEL,
            framework="pt",
            device=-1,
        )
        return pipe
    except Exception as e:
        st.warning(f"Could not load summarization model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    """Load the extractive QA pipeline (RoBERTa-SQuAD2)."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "question-answering",
            model=cfg.QA_MODEL,
            framework="pt",
            device=-1,
        )
        return pipe
    except Exception as e:
        st.warning(f"Could not load QA model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_word2vec_models():
    """Load EN and FR Word2Vec models from Step 4."""
    w2v_en = w2v_fr = None
    try:
        from gensim.models import Word2Vec
        if cfg.W2V_EN_MODEL.exists():
            w2v_en = Word2Vec.load(str(cfg.W2V_EN_MODEL))
        if cfg.W2V_FR_MODEL.exists():
            w2v_fr = Word2Vec.load(str(cfg.W2V_FR_MODEL))
    except Exception:
        pass
    return w2v_en, w2v_fr

@st.cache_data(show_spinner=False)
def build_sentence_vectors(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Build TF-IDF weighted sentence vectors from Word2Vec embeddings.
    Returns (n_docs, embed_dim) float32 array or None if models unavailable.
    """
    w2v_en, _ = load_word2vec_models()
    tfidf_en, _ = load_tfidf_vectorizers()
    if w2v_en is None or tfidf_en is None:
        return None

    wv      = w2v_en.wv
    idf_map = dict(zip(tfidf_en.get_feature_names_out(), tfidf_en.idf_))
    dim     = wv.vector_size

    def sentence_vec(tokens):
        if not isinstance(tokens, list):
            return np.zeros(dim, dtype=np.float32)
        vecs, weights = [], []
        for w in tokens:
            if w in wv:
                vecs.append(wv[w])
                weights.append(idf_map.get(w, 1.0))
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.average(vecs, axis=0, weights=weights).astype(np.float32)

    col = cfg.COL_TEXT_EN
    if "tokens_en" in df.columns:
        token_col = df["tokens_en"]
    else:
        # Fallback: tokenize from text
        from utils.preprocessing import simple_tokenize
        token_col = df[col].apply(simple_tokenize)

    vecs = np.vstack([sentence_vec(t) for t in token_col])
    return vecs
