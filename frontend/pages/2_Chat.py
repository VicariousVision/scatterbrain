"""Chat Page — Scatterbrain Legal Knowledge Graph.

Provides a conversational interface for querying the legal knowledge graph.
Users can select the LLM backend (Ollama / DeepSeek / OpenRouter) via a
sidebar dropdown.  The selected backend is sent to the FastAPI backend on
every query and controls both Cypher generation and answer synthesis.

Requirements: 5.1, 5.2, 5.6, 5.7, 5.8
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the frontend package root is on the path when running via
# ``streamlit run frontend/app.py`` from the project root.
_frontend_root = Path(__file__).resolve().parent.parent
if str(_frontend_root) not in sys.path:
    sys.path.insert(0, str(_frontend_root))

import api_client  # noqa: E402  (import after sys.path fixup)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("💬 Chat with Your Documents")

# ---------------------------------------------------------------------------
# Session state initialisation
# Requirement 5.1 — maintain Chat Session message history
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_backend" not in st.session_state:
    st.session_state["selected_backend"] = "ollama"

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    # ------------------------------------------------------------------
    # Backend selector
    # ------------------------------------------------------------------
    _BACKEND_OPTIONS: dict[str, str] = {
        "🖥️  Ollama (local)": "ollama",
        "🤖  DeepSeek": "deepseek",
        "🌐  OpenRouter (free-tier first)": "openrouter",
    }

    selected_label = st.selectbox(
        "LLM Backend",
        options=list(_BACKEND_OPTIONS.keys()),
        index=list(_BACKEND_OPTIONS.values()).index(
            st.session_state["selected_backend"]
        ),
        help=(
            "Choose the LLM used for both Cypher generation and answer synthesis.\n\n"
            "- **Ollama** — local model, no API key needed\n"
            "- **DeepSeek** — requires DEEPSEEK_CHAT_API_KEY in .env\n"
            "- **OpenRouter** — requires OPENROUTER_API_KEY in .env; "
            "free-tier models are used first, paid fallback only when all "
            "free-tier slots are exhausted"
        ),
    )
    selected_backend: str = _BACKEND_OPTIONS[selected_label]
    st.session_state["selected_backend"] = selected_backend

    # Show a hint if the user picks an external backend.
    if selected_backend == "deepseek":
        st.info(
            "DeepSeek backend selected.\n\n"
            "Make sure **DEEPSEEK_CHAT_API_KEY** is set in your `.env` file."
        )
    elif selected_backend == "openrouter":
        st.info(
            "OpenRouter backend selected.\n\n"
            "Make sure **OPENROUTER_API_KEY** is set in your `.env` file.\n\n"
            "Free-tier models are tried first; the paid fallback is only used "
            "when all free slots are rate-limited."
        )

    st.divider()
    st.header("Session")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    st.caption(f"{len(st.session_state['messages'])} message(s) in history")

# ---------------------------------------------------------------------------
# Message history panel
# Requirements 5.1, 5.6 — scrollable panel displaying conversation
# ---------------------------------------------------------------------------

st.subheader("Conversation")

# Render all messages stored in session state
for message in st.session_state["messages"]:
    role = message.get("role", "user")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# ---------------------------------------------------------------------------
# Query input and submission
# Requirements 5.2, 5.6, 5.7, 5.8
# ---------------------------------------------------------------------------

query = st.chat_input("Ask a question about your documents…")

if query:
    # Immediately display the user's message in the history panel
    with st.chat_message("user"):
        st.markdown(query)

    # Requirement 5.7 — show loading indicator while backend processes the query
    with st.chat_message("assistant"):
        backend_label = selected_label.split("  ", 1)[-1]  # strip emoji prefix
        with st.spinner(f"Thinking… ({backend_label})"):
            try:
                # Requirement 5.2 — send query + current history + backend to backend
                result = api_client.chat_query(
                    query=query,
                    history=st.session_state["messages"],
                    backend=selected_backend,
                )

                assistant_response: str = result.get("response", "")
                used_backend: str = result.get("backend", selected_backend)
                generated_cypher: str | None = result.get("generated_cypher")
                cypher_source: str | None = result.get("cypher_source")

                # Display the assistant's reply with a small backend badge.
                st.markdown(assistant_response)
                st.caption(f"_Answered by: **{used_backend}**_")

                # Show the generated Cypher in a collapsible expander.
                if generated_cypher:
                    source_label = (
                        "keyword fallback" if cypher_source == "keyword_fallback"
                        else "Text2Cypher"
                    )
                    with st.expander(f"🔍 Graph query ({source_label})", expanded=False):
                        st.code(generated_cypher, language="cypher")

                # Requirement 5.6 — append both messages to session state
                st.session_state["messages"].append(
                    {"role": "user", "content": query}
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": assistant_response}
                )

            except Exception as exc:  # noqa: BLE001
                # Requirement 5.8 — display error without clearing history
                error_msg = str(exc)
                if "400" in error_msg:
                    st.error(
                        f"Configuration error: {error_msg}\n\n"
                        "Check that the required API key is set in your `.env` file."
                    )
                elif "503" in error_msg:
                    st.error(
                        "Ollama is unavailable. Make sure `ollama serve` is running."
                    )
                else:
                    st.error(f"Error: {error_msg}")
