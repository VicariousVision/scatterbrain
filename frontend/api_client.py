"""HTTP client for the Scatterbrain RAG backend API.

All functions read ``BACKEND_URL`` from the environment, defaulting to
``http://localhost:8000``.
"""

from __future__ import annotations

import os

import httpx

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

# Supported file extensions for upload validation
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


def validate_file_type(filename: str) -> bool:
    """Return True if *filename* has a supported extension, False otherwise.
    
    Supported extensions are ``.pdf`` and ``.txt``.
    
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
    
    Parameters
    ----------
    file_bytes:
        Raw bytes of the file to upload.
    filename:
        Original filename, used as the ``filename`` field in the multipart form.
        
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
    
    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code.
    """
    with httpx.Client() as client:
        response = client.get(f"{BACKEND_URL}/documents/")
        response.raise_for_status()
        return response.json()


def delete_document(document_id: str) -> None:
    """Delete a document and all its chunks.
    
    Sends a DELETE request to ``/documents/{document_id}``.
    
    Parameters
    ----------
    document_id:
        The UUID string of the document to delete.
        
    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code.
    """
    with httpx.Client() as client:
        response = client.delete(f"{BACKEND_URL}/documents/{document_id}")
        response.raise_for_status()


def chat_query(query: str, history: list) -> dict:
    """Submit a chat query to the backend and return the LLM response.
    
    Sends a POST request to ``/chat/query`` with a 180-second timeout.
    Returns the parsed JSON body containing ``response`` (the LLM answer
    text) and ``history`` (the updated message history).
    
    Parameters
    ----------
    query:
        The user's natural-language question.
    history:
        The current chat session message history as a list of
        ``{"role": "user"|"assistant", "content": str}`` dicts.
        
    Raises
    ------
    httpx.HTTPStatusError
        If the backend returns a non-2xx status code (e.g. 503 when Ollama
        is unavailable).
    """
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{BACKEND_URL}/chat/query",
            json={"query": query, "history": history},
        )
        response.raise_for_status()
        return response.json()
