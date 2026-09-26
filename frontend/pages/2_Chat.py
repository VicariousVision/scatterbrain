"""Chat Page — Scatterbrain RAG.

Provides a conversational interface for querying uploaded documents using RAG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the frontend package root is on the path
_frontend_root = Path(__file__).resolve().parent.parent
if str(_frontend_root) not in sys.path:
    sys.path.insert(0, str(_frontend_root))

import api_client  # noqa: E402

st.title("💬 Chat with Your Documents")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    st.caption(f"{len(st.session_state['messages'])} message(s) in history")
    
    st.divider()
    st.info(
        "**How it works:**\n\n"
        "1. Upload documents in the Upload page\n"
        "2. Ask questions here\n"
        "3. The system retrieves relevant chunks from the vector database\n"
        "4. The LLM generates an answer based only on retrieved context"
    )

# ---------------------------------------------------------------------------
# Message history panel
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
# ---------------------------------------------------------------------------

query = st.chat_input("Ask a question about your documents…")

if query:
    # Immediately display the user's message in the history panel
    with st.chat_message("user"):
        st.markdown(query)

    # Show loading indicator while backend processes the query
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = api_client.chat_query(
                    query=query,
                    history=st.session_state["messages"],
                )

                assistant_response: str = result.get("response", "")
                retrieved_chunks: int = result.get("retrieved_chunks", 0)

                # Display the assistant's reply
                st.markdown(assistant_response)
                st.caption(f"_Retrieved {retrieved_chunks} relevant chunks_")

                # Append both messages to session state
                st.session_state["messages"].append(
                    {"role": "user", "content": query}
                )
                st.session_state["messages"].append(
                    {"role": "assistant", "content": assistant_response}
                )

            except Exception as exc:
                error_msg = str(exc)
                if "503" in error_msg:
                    st.error(
                        "Ollama is unavailable. Make sure `ollama serve` is running."
                    )
                else:
                    st.error(f"Error: {error_msg}")
