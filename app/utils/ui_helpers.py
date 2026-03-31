import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

PRIMARY      = "#1565C0"
SECONDARY    = "#0D47A1"
ACCENT_GREEN = "#43A047"
ACCENT_RED   = "#E53935"
ACCENT_AMBER = "#FB8C00"
BG_CARD      = "#F5F8FF"


def page_header(icon: str, title: str, subtitle: str = ""):
    """Render a styled page header."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: white;
        ">
            <h1 style="margin:0; font-size:1.8rem;">{icon} {title}</h1>
            {"<p style='margin:0.4rem 0 0; opacity:0.85;'>" + subtitle + "</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "", color: str = PRIMARY):
    """A simple styled metric card."""
    st.markdown(
        f"""
        <div style="
            background:{BG_CARD};
            border-left: 4px solid {color};
            padding: 0.8rem 1.2rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        ">
            <div style="font-size:0.8rem; color:#555; text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
            <div style="font-size:1.6rem; font-weight:700; color:{color};">{value}</div>
            {"<div style='font-size:0.8rem; color:#888;'>" + delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_bar(label: str, confidence: float, color: str = PRIMARY):
    """Render a labeled progress bar for confidence/probability."""
    pct = round(confidence * 100, 1)
    bar_html = f"""
    <div style="margin-bottom:0.5rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:2px;">
            <span>{label}</span><span><b>{pct}%</b></span>
        </div>
        <div style="background:#e0e0e0; border-radius:4px; height:10px;">
            <div style="background:{color}; width:{pct}%; height:10px; border-radius:4px; transition:width 0.4s;"></div>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def star_rating_badge(rating: int) -> str:
    """Return a star string for ratings 1–5."""
    filled = "⭐" * rating
    return filled if rating else "—"


def highlight_text(text: str, words: list, color: str = "#FFF59D") -> str:
    """
    Return HTML with specified words highlighted.
    Used in QA and XAI pages to show relevant spans.
    """
    import re
    if not words or not text:
        return text

    pattern = r"(" + "|".join(re.escape(w) for w in words if w) + r")"
    highlighted = re.sub(
        pattern,
        rf'<mark style="background:{color}; padding:0 2px; border-radius:3px;">\1</mark>',
        text,
        flags=re.IGNORECASE,
    )
    return highlighted


def results_table(
    df: pd.DataFrame,
    columns: list,
    column_labels: dict = None,
    max_rows: int = 10,
):
    """
    Render a clean styled dataframe as an interactive table.

    Args:
        df: results dataframe
        columns: list of column names to display
        column_labels: {col: display_label}
        max_rows: max rows to display
    """
    show_cols = [c for c in columns if c in df.columns]
    display   = df[show_cols].head(max_rows).copy()

    if column_labels:
        display = display.rename(columns=column_labels)

    # Format score column
    if "score" in display.columns:
        display["score"] = display["score"].apply(lambda x: f"{x:.3f}")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


def info_box(text: str, kind: str = "info"):
    """Styled info / warning / success / error box."""
    colors = {
        "info":    ("#E3F2FD", "#1565C0"),
        "warning": ("#FFF8E1", "#F57F17"),
        "success": ("#E8F5E9", "#2E7D32"),
        "error":   ("#FFEBEE", "#C62828"),
    }
    bg, border = colors.get(kind, colors["info"])
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left:4px solid {border};
            padding:0.75rem 1rem;
            border-radius:6px;
            margin-bottom:0.8rem;
        ">
            &nbsp;&nbsp;{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str = "No results found.", icon: str = "🔍"):
    st.markdown(
        f"""
        <div style="text-align:center; padding:2rem; color:#888;">
            <div style="font-size:2.5rem;">{icon}</div>
            <p style="font-size:1rem; margin-top:0.5rem;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_chart(labels: list, probs: list, title: str = "Probability Distribution"):
    """
    Render a Plotly horizontal bar chart for class probabilities.
    """
    try:
        import plotly.graph_objects as go

        colors = [PRIMARY if p == max(probs) else "#90CAF9" for p in probs]
        fig = go.Figure(go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(range=[0, 1.1], tickformat=".0%"),
            height=max(200, len(labels) * 45),
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        for label, prob in zip(labels, probs):
            confidence_bar(label, prob)
