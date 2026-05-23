"""HTTP client for the Scatterbrain backend API.

All functions read ``BACKEND_URL`` from the environment, defaulting to
``http://localhost:8000``.

Requirements: 1.2, 1.3, 5.2, 7.2, 7.4, 8.3
"""

from __future__ import annotations

import os

import httpx

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

# Supported file extensions for upload validation (Requirement 1.3)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".txt"})


def validate_file_type(filename: str) -> bool:
    """Return True if *filename* has a supported extension, False otherwise.

    Supported extensions are ``.pdf``, ``.docx``, and ``.txt``.

    This validation must be performed before calling :func:`upload_document`
    so that unsupported file types are rejected on the client side and the
    upload API is never called for them.

    Requirements: 1.3

    Parameters
    ----------
    filename:
        The name of the file to validate (e.g. ``"contract.pdf"``).

    Returns
    -------
    bool
        ``True`` when the extension is supported, ``False`` otherwise.
    """
    ext = os.path.splitext(filename)[-1].lower()
    return ext in SUPPORTED_EXTENSIONS


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """Upload a document to the backend for processing.

    Sends a multipart/form-data POST request to ``/documents/upload``.
    Returns the parsed JSON response body, which contains a ``document_id``
    field on success (HTTP 202).

    Requirements: 1.2

    Parameters
    ----------
    file_bytes:
        Raw bytes of the file to upload.
    filename:
        Original filename, used as the ``filename`` field in the multipart
        form and to set the MIME type.

    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code.
    """
    with httpx.Client() as client:
        response = client.post(
            f"{BACKEND_URL}/documents/upload",
            files={"file": (filename, file_bytes, "application/octet-stream")},
        )
        response.raise_for_status()
        return response.json()


def get_document_status(document_id: str) -> dict:
    """Fetch the current status record for a document.

    Sends a GET request to ``/documents/{document_id}`` and returns the
    parsed JSON body containing ``document_id``, ``filename``,
    ``uploaded_at``, and ``status`` fields.

    Requirements: 7.2

    Parameters
    ----------
    document_id:
        The UUID string returned by :func:`upload_document`.

    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code (e.g. 404 when the
        document does not exist).
    """
    with httpx.Client() as client:
        response = client.get(f"{BACKEND_URL}/documents/{document_id}")
        response.raise_for_status()
        return response.json()


def list_documents() -> list:
    """Return the list of all uploaded documents.

    Sends a GET request to ``/documents`` and returns the parsed JSON array.
    Each element contains ``document_id``, ``filename``, ``uploaded_at``,
    and ``status`` fields.

    Requirements: 7.2

    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code.
    """
    with httpx.Client() as client:
        response = client.get(f"{BACKEND_URL}/documents/")
        response.raise_for_status()
        return response.json()


def get_graph_summary(document_id: str) -> dict:
    """Fetch the entity/relationship counts for a processed document.

    Sends a GET request to ``/documents/{document_id}/graph-summary`` and
    returns the parsed JSON body containing ``node_count`` and
    ``edge_count`` fields.

    Requirements: 7.4

    Parameters
    ----------
    document_id:
        The UUID string of the document whose graph summary is requested.

    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code.
    """
    with httpx.Client() as client:
        response = client.get(
            f"{BACKEND_URL}/documents/{document_id}/graph-summary"
        )
        response.raise_for_status()
        return response.json()


def chat_query(query: str, history: list) -> dict:
    """Submit a chat query to the backend and return the LLM response.

    Sends a POST request to ``/chat/query`` with a 120-second timeout.
    Returns the parsed JSON body containing ``response`` (the LLM answer
    text) and ``history`` (the updated message history).

    Requirements: 5.2

    Parameters
    ----------
    query:
        The user's natural-language question.
    history:
        The current Chat Session message history as a list of
        ``{"role": "user"|"assistant", "content": str}`` dicts.

    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code (e.g. 503 when Ollama
        is unavailable).
    """
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{BACKEND_URL}/chat/query",
            json={"query": query, "history": history},
        )
        response.raise_for_status()
        return response.json()
