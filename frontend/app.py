"""Scatterbrain — Legal Knowledge Graph.

Streamlit multi-page application entry point.

This module serves as the landing page and navigation hub for the Scatterbrain
application.  Streamlit automatically discovers the pages in the ``pages/``
sub-directory and renders them in the sidebar navigation.

Requirements: 1.1, 5.1
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# Must be the first Streamlit call in the entry-point script.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Scatterbrain — Legal Knowledge Graph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Landing page content
# ---------------------------------------------------------------------------

st.title("⚖️🧠🤖 Scatterbrain")
st.subheader("Legal Knowledge Graph")

st.markdown(
    """
    Welcome to **Scatterbrain**, a legal document intelligence application for Currency and Exchanges for the SARB.

    Use the sidebar to navigate between pages:

    - **📄 Upload** — Upload PDF, DOCX, or TXT documents for processing.
      The system will extract entities and relationships and store them in a
      knowledge graph.
    - **💬 Chat** — Ask natural language questions about your uploaded
      documents.  Responses are grounded in the extracted knowledge graph
      using a graph-RAG pattern.
    """
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.info(
        "**Getting started**\n\n"
        "1. Go to the **Upload** page and upload one or more legal documents.\n"
        "2. Wait for processing to complete (status will update automatically).\n"
        "3. Switch to the **Chat** page and start asking questions."
    )

with col2:
    st.info(
        "**Supported formats**\n\n"
        "- PDF (`.pdf`)\n"
        "- Word documents (`.docx`)\n"
        "- Plain text (`.txt`)"
    )
