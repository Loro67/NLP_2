import re
import string
from typing import List

def clean_text(text: str) -> str:
    """
    Lightweight cleaning: normalize whitespace, strip URLs and odd symbols.
    Preserves punctuation and accented characters for readability.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove unusual symbols/emojis but keep punctuation and Unicode word chars
    text = re.sub(r"[^\w\s.,!?;:'\"()\-]", "", text)

    # Normalize whitespace and line breaks
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)

    # Fix repeated characters (e.g. "trèèèès" → "trèès")
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # Fix repeated punctuation
    text = re.sub(r"([!?.])[\1]{2,}", r"\1", text)

    return text.strip()



STOP_EN_BASIC = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "i", "me", "my", "we", "our", "you", "your", "he", "his",
    "she", "her", "they", "their", "it", "its", "this", "that", "these",
    "those", "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "just", "as", "if", "because", "although", "while",
    "when", "where", "who", "which", "what", "how", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "there", "here",
}


def simple_tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Simple regex tokenizer — no external model required.
    Returns lowercase alpha tokens longer than 2 characters.
    """
    text = clean_text(text).lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", text)
    tokens = [t for t in tokens if len(t) > 2]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_EN_BASIC]
    return tokens


def tokens_to_string(tokens: list) -> str:
    """Join a token list back into a space-separated string."""
    if isinstance(tokens, list):
        return " ".join(str(t) for t in tokens)
    return str(tokens) if tokens else ""


def prepare_for_tfidf(text: str) -> str:
    """
    Prepare raw user input for TF-IDF vectorization.
    Tokenizes and joins so the vocabulary matches notebook 3 output.
    """
    tokens = simple_tokenize(text)
    return " ".join(tokens)


def truncate_text(text: str, max_words: int = 200) -> str:
    """Truncate text to max_words words."""
    words = str(text).split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"
