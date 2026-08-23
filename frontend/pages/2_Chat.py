"""Chat Page — Scatterbrain Legal Knowledge Graph.

Provides a conversational interface for querying the legal knowledge graph.
Users can submit natural language questions and receive LLM responses grounded
in the extracted graph data.

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
        with st.spinner("Thinking…"):
            try:
                # Requirement 5.2 — send query + current history to backend
                result = api_client.chat_query(
                    query=query,
                    history=st.session_state["messages"],
                )

                assistant_response: str = result.get("response", "")

                # Display the assistant's reply
                st.markdown(assistant_response)

                # Requirement 5.6 — append both messages to session state
                st.session_state["messages"].append(
                    {"role": "user", "content": query}
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": assistant_response}
                )

            except Exception as exc:  # noqa: BLE001
                # Requirement 5.8 — display error without clearing history
                st.error(f"Error: {exc}")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    st.caption(f"{len(st.session_state['messages'])} message(s) in history")
