"""Text cleaning for parsed document content.

Normalizes raw extracted text (especially from PDFs) before chunking and
embedding. The goal is to remove formatting noise that hurts retrieval
quality -- control characters, zero-width junk, broken hyphenation, and
irregular whitespace -- while preserving the natural text that transformer
embedding models are trained on.

Design notes
------------
This cleaner is intentionally conservative. It does NOT lowercase, strip
punctuation, or remove stopwords: modern sentence-embedding models expect
natural, cased text with punctuation, so those classic NLP steps would hurt
retrieval rather than help it. It only removes characters and layout
artifacts that carry no meaning.
"""

from __future__ import annotations

import re
import unicodedata

# Control characters that carry no textual meaning. Tab and newline are kept
# (whitespace handling deals with them); everything else in the C0/C1 ranges
# plus the Unicode replacement char is removed.
_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]"
)

# Zero-width and other invisible formatting characters that PDFs often embed.
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

# A word split across a line break with a hyphen, e.g. "inter-\nnational".
# Rejoins into "international". Only fires on lowercase-hyphen-lowercase so it
# doesn't merge things like "well-\nknown" list items incorrectly is a risk,
# but rejoining is the far more common correct behavior for wrapped prose.
_HYPHEN_LINEBREAK = re.compile(r"([A-Za-z])-\n([a-z])")

# Horizontal whitespace (spaces, tabs, non-breaking spaces) runs.
_HORIZONTAL_WS = re.compile(r"[ \t\u00a0\u2000-\u200a\u205f\u3000]+")

# Three or more consecutive newlines collapse to a paragraph break.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Trailing horizontal whitespace at the end of each line.
_TRAILING_WS = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """Normalize raw extracted text for chunking and embedding.

    The steps, in order:
      1. Unicode NFKC normalization (ligatures, full-width forms, fancy
         quotes/dashes become their plain equivalents).
      2. Normalize line endings to ``\\n``.
      3. Rejoin words hyphenated across line breaks.
      4. Remove control characters, the replacement char, and zero-width
         formatting characters.
      5. Collapse horizontal whitespace runs to a single space.
      6. Strip trailing whitespace per line and cap consecutive blank lines.

    Args:
        text: Raw text as returned by the document parser.

    Returns:
        Cleaned text. Returns an empty string for empty/whitespace-only input.
    """
    if not text or not text.strip():
        return ""

    # 1. Canonical Unicode form. NFKC folds ligatures (ﬁ -> fi), full-width
    #    characters, and many typographic variants into plain ASCII-ish forms.
    text = unicodedata.normalize("NFKC", text)

    # 2. Normalize line endings before any newline-sensitive step.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Rejoin hyphenated line breaks from wrapped prose.
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # 4. Drop invisible/control characters that carry no meaning.
    text = _CONTROL_CHARS.sub("", text)
    text = _ZERO_WIDTH.sub("", text)

    # 5. Collapse horizontal whitespace runs (does not touch newlines).
    text = _HORIZONTAL_WS.sub(" ", text)

    # 6. Tidy vertical whitespace: no trailing spaces, at most one blank line.
    text = _TRAILING_WS.sub("\n", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)

    return text.strip()
