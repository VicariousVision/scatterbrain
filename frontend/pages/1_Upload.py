"""Upload Page — Scatterbrain RAG.

Allows users to upload PDF or TXT documents, monitors processing
status, and displays the full document list.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Ensure the frontend package root is on the path
_frontend_root = Path(__file__).resolve().parent.parent
if str(_frontend_root) not in sys.path:
    sys.path.insert(0, str(_frontend_root))

import api_client  # noqa: E402

st.set_page_config(page_title="Upload Documents — Scatterbrain", layout="wide")
st.title("📄 Upload Documents")

# ---------------------------------------------------------------------------
# Section 1: File upload
# ---------------------------------------------------------------------------

st.header("Upload a Document")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "txt"],
    help="Supported formats: PDF, TXT",
)

if uploaded_file is not None:
    filename = uploaded_file.name

    if not api_client.validate_file_type(filename):
        st.error(
            f"File type not supported: **{Path(filename).suffix}**. "
            "Please upload a PDF or TXT file."
        )
    else:
        if st.button("Upload and Process", type="primary"):
            file_bytes = uploaded_file.read()

            try:
                with st.spinner("Uploading document…"):
                    upload_result = api_client.upload_document(file_bytes, filename)

                document_id: str = upload_result["document_id"]
                st.info(f"Document accepted. ID: `{document_id}`")

                # Poll until completed or failed
                status_placeholder = st.empty()
                poll_interval_seconds = 2
                max_polls = 150  # 5 minutes maximum

                with st.spinner("Processing document (parsing, chunking, embedding)…"):
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
                        f"✅ **{filename}** processed successfully. "
                        f"Chunks embedded and stored in vector database."
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

                # Refresh the page state
                st.rerun()

            except Exception as exc:
                st.error(f"Upload failed: {exc}")

# ---------------------------------------------------------------------------
# Section 2: Document list
# ---------------------------------------------------------------------------

st.divider()
st.header("Uploaded Documents")

try:
    documents: list[dict] = api_client.list_documents()
except Exception as exc:
    st.error(f"Could not fetch document list: {exc}")
    documents = []

if not documents:
    st.info("No documents uploaded yet.")
else:
    # Build a display-friendly list
    for doc in documents:
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{doc.get('filename', 'Unknown')}**")
            st.caption(f"ID: {doc.get('document_id', '')[:8]}…")
        
        with col2:
            status = doc.get('status', 'unknown')
            if status == 'completed':
                st.success(f"✅ {status}")
            elif status == 'failed':
                st.error(f"❌ {status}")
            else:
                st.info(f"⏳ {status}")
            
            if doc.get('error'):
                st.caption(f"Error: {doc['error']}")
        
        with col3:
            if st.button("Delete", key=f"delete_{doc.get('document_id')}", type="secondary"):
                try:
                    api_client.delete_document(doc.get('document_id'))
                    st.success("Deleted!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
        
        st.divider()
