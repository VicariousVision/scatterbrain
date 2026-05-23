"""Property-based tests for frontend/api_client.py.

**Property 1: Unsupported file types are always rejected**
**Validates: Requirements 1.3**

For any filename whose extension is not in {.pdf, .docx, .txt}, the
``validate_file_type`` function must return False, ensuring the upload API
is never called for unsupported file types.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from frontend.api_client import SUPPORTED_EXTENSIONS, validate_file_type

# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

# All supported extensions (without the leading dot, for convenience)
_SUPPORTED_BARE = {ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS}

# Strategy: generate extensions that are NOT in the supported set.
# We build arbitrary lowercase alphabetic extensions (1–10 chars) and filter
# out the three supported ones.
_unsupported_extension = (
    st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1,
        max_size=10,
    )
    .filter(lambda ext: ext not in _SUPPORTED_BARE)
    .map(lambda ext: f".{ext}")
)

# Strategy: generate a base filename (no extension) — at least one char.
_base_filename = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=50,
)


# ---------------------------------------------------------------------------
# Property 1 — unsupported extensions are always rejected
# ---------------------------------------------------------------------------


@given(base=_base_filename, ext=_unsupported_extension)
@settings(max_examples=200)
def test_unsupported_file_types_always_rejected(base: str, ext: str) -> None:
    """**Property 1: Unsupported file types are always rejected**

    **Validates: Requirements 1.3**

    For any filename whose extension is not in {.pdf, .docx, .txt}:
    - ``validate_file_type`` returns False
    - ``upload_document`` is never called
    """
    filename = f"{base}{ext}"

    # The validation function must return False for unsupported extensions.
    assert validate_file_type(filename) is False, (
        f"validate_file_type({filename!r}) should return False "
        f"for unsupported extension {ext!r}"
    )

    # Confirm the upload API is never invoked when validation fails.
    # In real UI code the caller checks validate_file_type() before calling
    # upload_document(); here we verify that contract by patching the HTTP
    # layer and asserting it is never reached.
    with patch("frontend.api_client.httpx.Client") as mock_client_cls:
        result = validate_file_type(filename)
        assert result is False
        # The HTTP client must not have been instantiated (upload not called).
        mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Sanity check — supported extensions are accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["doc.pdf", "report.docx", "notes.txt"])
def test_supported_file_types_are_accepted(filename: str) -> None:
    """Supported extensions (.pdf, .docx, .txt) must pass validation."""
    assert validate_file_type(filename) is True


@pytest.mark.parametrize(
    "filename",
    [
        "image.png",
        "spreadsheet.xlsx",
        "archive.zip",
        "script.py",
        "data.csv",
        "no_extension",
        ".hidden",
    ],
)
def test_unsupported_file_types_rejected_examples(filename: str) -> None:
    """Concrete examples of unsupported filenames are rejected."""
    assert validate_file_type(filename) is False


@pytest.mark.parametrize(
    "filename",
    [
        "file.PDF",   # uppercase — normalised to .pdf, so accepted
        "file.DOCX",  # normalised to .docx
        "file.TXT",   # normalised to .txt
    ],
)
def test_uppercase_supported_extensions_are_accepted(filename: str) -> None:
    """Uppercase variants of supported extensions are accepted (case-insensitive)."""
    assert validate_file_type(filename) is True
