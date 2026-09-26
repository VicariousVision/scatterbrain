"""Scatterbrain — Document RAG System.

Streamlit multi-page application entry point.

This module serves as the landing page and navigation hub for the Scatterbrain
RAG application.  Streamlit automatically discovers the pages in the ``pages/``
sub-directory and renders them in the sidebar navigation.
"""

from __future__ import annotations

import streamlit as st

# Page configuration - must be the first Streamlit call
st.set_page_config(
    page_title="Scatterbrain RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Landing page content
st.title("🧠 Scatterbrain")
st.subheader("Document Intelligence with RAG")

st.markdown(
    """
    Welcome to **Scatterbrain**, a document intelligence system powered by Retrieval-Augmented Generation (RAG).

    Use the sidebar to navigate between pages:

    - **📄 Upload** — Upload PDF or TXT documents for processing.
      Documents are parsed, chunked, embedded, and stored in a vector database.
    - **💬 Chat** — Ask natural language questions about your uploaded documents.
      Responses are grounded in the most relevant document chunks retrieved from the vector store.
    """
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.info(
        "**Getting started**\n\n"
        "1. Go to the **Upload** page and upload one or more documents.\n"
        "2. Wait for processing to complete (parsing → chunking → embedding).\n"
        "3. Switch to the **Chat** page and start asking questions."
    )

with col2:
    st.info(
        "**Supported formats**\n\n"
        "- PDF (`.pdf`)\n"
        "- Plain text (`.txt`)\n\n"
        "**Tech stack**\n\n"
        "- Ollama (qwen3.5:0.8b + nomic-embed-text)\n"
        "- SQLite + sqlite-vec (vector store)\n"
        "- Custom text chunking"
    )
