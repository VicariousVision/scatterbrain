"""Validation tests for legal chunking and retrieval settings."""

import pytest
from pydantic import ValidationError

from config import Settings


def test_structured_defaults_are_consistent() -> None:
    fields = Settings.model_fields
    assert (fields["chunk_size"].default, fields["chunk_overlap"].default) == (1000, 200)
    assert (
        fields["legal_chunk_min_chars"].default,
        fields["legal_chunk_target_chars"].default,
        fields["legal_chunk_hard_max_chars"].default,
    ) == (450, 900, 1400)
    assert fields["legal_forced_split_overlap_chars"].default == 120
    assert (
        fields["retrieval_candidate_pool"].default,
        fields["retrieval_top_k"].default,
    ) == (10, 5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"legal_chunk_min_chars": 901},
        {"legal_chunk_target_chars": 1401},
        {"legal_forced_split_overlap_chars": 900},
        {"legal_parent_max_chars": 1000},
        {"retrieval_candidate_pool": 4, "retrieval_top_k": 5},
    ],
)
def test_invalid_related_settings_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)
