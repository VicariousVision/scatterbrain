"""Validation tests for the optional Gemini chat provider settings."""
import pytest
from pydantic import ValidationError

from config import Settings


def test_llm_provider_defaults_to_ollama() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.allow_cloud_llm is False
    assert settings.gemini_model == "gemini-3.5-flash-lite"


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_provider": "gemini"},
        {"llm_provider": "gemini", "gemini_api_key": "k"},
        {"llm_provider": "gemini", "allow_cloud_llm": True},
        {"llm_provider": "gemini", "gemini_api_key": "  ", "allow_cloud_llm": True},
        {"llm_provider": "openai"},
    ],
)
def test_gemini_requires_key_and_opt_in(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_gemini_with_key_and_opt_in_is_valid_and_key_is_secret() -> None:
    settings = Settings(
        _env_file=None, llm_provider="gemini", gemini_api_key="sekret", allow_cloud_llm=True
    )
    assert settings.gemini_api_key.get_secret_value() == "sekret"
    assert "sekret" not in repr(settings)
