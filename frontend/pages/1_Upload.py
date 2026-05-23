"""Upload Page — Scatterbrain Legal Knowledge Graph.

Allows users to upload PDF, DOCX, or TXT documents, monitors processing
status, and displays the full document list with per-document graph summaries.

Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 7.1, 7.2, 7.4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Ensure the frontend package root is on the path when running via
# ``streamlit run frontend/app.py`` from the project root.
_frontend_root = Path(__file__).resolve().parent.parent
if str(_frontend_root) not in sys.path:
    sys.path.insert(0, str(_frontend_root))

import api_client  # noqa: E402  (import after sys.path fixup)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Upload Documents — Scatterbrain", layout="wide")
st.title("📄 Upload Documents")

# ---------------------------------------------------------------------------
# Section 1: File upload
# ---------------------------------------------------------------------------

st.header("Upload a Document")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT",
)

if uploaded_file is not None:
    filename = uploaded_file.name

    # Requirement 1.3 — validate file type before calling the API
    if not api_client.validate_file_type(filename):
        st.error(
            f"File type not supported: **{Path(filename).suffix}**. "
            "Please upload a PDF, DOCX, or TXT file."
        )
    else:
        if st.button("Upload and Process", type="primary"):
            file_bytes = uploaded_file.read()

            # Requirement 1.2 — send file to backend
            try:
                with st.spinner("Uploading document…"):
                    upload_result = api_client.upload_document(file_bytes, filename)

                document_id: str = upload_result["document_id"]
                st.info(f"Document accepted. ID: `{document_id}`")

                # Requirements 1.5, 1.6, 1.7 — poll until completed or failed
                status_placeholder = st.empty()
                poll_interval_seconds = 2
                max_polls = 150  # 5 minutes maximum

                with st.spinner("Processing document…"):
                    for _ in range(max_polls):
                        status_record = api_client.get_document_status(document_id)
                        current_status: str = status_record.get("status", "processing")

                        status_placeholder.caption(
                            f"Status: **{current_status}**"
                        )

                        if current_status == "completed":
                            break
                        if current_status == "failed":
                            break

                        time.sleep(poll_interval_seconds)

                # Display final notification
                if current_status == "completed":
                    st.success(
                        f"✅ **{filename}** processed successfully."
                    )
                elif current_status == "failed":
                    error_detail = status_record.get("error") or "Unknown error."
                    st.error(
                        f"❌ Processing failed for **{filename}**: {error_detail}"
                    )
                else:
                    st.warning(
                        "Processing is taking longer than expected. "
                        "Check the document list below for the latest status."
                    )

                # Refresh the page state so the document list updates
                st.rerun()

            except Exception as exc:  # noqa: BLE001
                st.error(f"Upload failed: {exc}")

# ---------------------------------------------------------------------------
# Section 2: Document list
# ---------------------------------------------------------------------------

st.divider()
st.header("Uploaded Documents")

# Requirements 7.1, 7.2 — fetch and display document list
try:
    documents: list[dict] = api_client.list_documents()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not fetch document list: {exc}")
    documents = []

if not documents:
    st.info("No documents uploaded yet.")
else:
    # Build a display-friendly list
    display_rows = [
        {
            "document_id": doc.get("document_id", ""),
            "filename": doc.get("filename", ""),
            "uploaded_at": doc.get("uploaded_at", ""),
            "status": doc.get("status", ""),
        }
        for doc in documents
    ]

    # Requirement 7.1 — show filename, timestamp, status
    st.dataframe(
        display_rows,
        column_order=["filename", "uploaded_at", "status", "document_id"],
        column_config={
            "filename": st.column_config.TextColumn("Filename"),
            "uploaded_at": st.column_config.TextColumn("Uploaded At"),
            "status": st.column_config.TextColumn("Status"),
            "document_id": st.column_config.TextColumn("Document ID"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------------------
    # Section 3: Per-document graph summary
    # ---------------------------------------------------------------------------

    st.subheader("Graph Summary")

    # Requirement 7.4 — let user select a document and show entity/relationship counts
    doc_options = {
        f"{doc.get('filename', 'Unknown')} ({doc.get('document_id', '')[:8]}…)": doc.get(
            "document_id", ""
        )
        for doc in documents
        if doc.get("status") == "completed"
    }

    if not doc_options:
        st.info("No completed documents available for graph summary.")
    else:
        selected_label = st.selectbox(
            "Select a completed document to view its graph summary:",
            options=list(doc_options.keys()),
        )

        if selected_label:
            selected_doc_id = doc_options[selected_label]

            try:
                with st.spinner("Fetching graph summary…"):
                    summary = api_client.get_graph_summary(selected_doc_id)

                node_count: int = summary.get("node_count", 0)
                edge_count: int = summary.get("edge_count", 0)

                col1, col2 = st.columns(2)
                col1.metric("Entities (nodes)", node_count)
                col2.metric("Relationships (edges)", edge_count)

            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not fetch graph summary: {exc}")
