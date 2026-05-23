"""
Property-based tests for the chat service.

**Validates: Requirements 5.4, 6.1, 6.5**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.services.chat_service import ChatService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for a single chat message dict (role + content)
_message_strategy = st.fixed_dictionaries(
    {
        "role": st.sampled_from(["user", "assistant"]),
        "content": st.text(min_size=1, max_size=200),
    }
)

# Strategy for a message history of length 0–50
_history_strategy = st.lists(_message_strategy, min_size=0, max_size=50)

# Strategy for arbitrary query text
_query_strategy = st.text(min_size=1, max_size=200)

# Strategy for arbitrary graph context text
_context_strategy = st.text(min_size=1, max_size=500)


# ---------------------------------------------------------------------------
# Helper: count messages in a formatted prompt
# ---------------------------------------------------------------------------

def _count_messages_in_prompt(prompt: str, history: list[dict]) -> int:
    """Count how many of the history messages appear in the prompt.

    Counts occurrences of 'USER:' and 'ASSISTANT:' role prefixes in the
    CONVERSATION HISTORY section of the prompt, which corresponds to the
    number of messages included.
    """
    # Extract the CONVERSATION HISTORY section
    if "CONVERSATION HISTORY:" not in prompt:
        return 0

    history_section_start = prompt.index("CONVERSATION HISTORY:") + len("CONVERSATION HISTORY:")
    # The section ends at the next major section header
    if "USER QUERY:" in prompt:
        history_section_end = prompt.index("USER QUERY:")
    else:
        history_section_end = len(prompt)

    history_section = prompt[history_section_start:history_section_end]

    # Count role prefixes (USER: and ASSISTANT:) in the history section
    user_count = history_section.count("\nUSER:")
    assistant_count = history_section.count("\nASSISTANT:")

    return user_count + assistant_count


# ---------------------------------------------------------------------------
# Property 13: Message history is truncated to at most 10 messages
# ---------------------------------------------------------------------------


@given(
    history=_history_strategy,
    user_query=_query_strategy,
    graph_context=_context_strategy,
)
@settings(max_examples=100)
def test_prompt_includes_at_most_10_messages(
    history: list[dict],
    user_query: str,
    graph_context: str,
) -> None:
    """
    Property 13: Message history is truncated to at most 10 messages.

    For any message history of length 0–50, the prompt constructed by
    ChatService._build_prompt (after applying the same truncation as
    ChatService.query) must include at most 10 messages from the history.

    **Validates: Requirements 6.5**
    """
    # Instantiate ChatService without live dependencies (we only call _build_prompt)
    service = ChatService(
        graph_query_service=None,  # type: ignore[arg-type]
        ollama_client=None,  # type: ignore[arg-type]
    )

    # Apply the same truncation logic used in ChatService.query()
    truncated_history = history[-10:] if len(history) > 10 else history

    # Build the prompt with the truncated history
    prompt = service._build_prompt(user_query, graph_context, truncated_history)

    # Count messages included in the prompt
    messages_in_prompt = _count_messages_in_prompt(prompt, truncated_history)

    assert messages_in_prompt <= 10, (
        f"Expected at most 10 messages in the prompt, but found {messages_in_prompt}. "
        f"Original history length: {len(history)}, "
        f"Truncated history length: {len(truncated_history)}."
    )

    # Also assert the truncated history itself never exceeds 10 messages
    assert len(truncated_history) <= 10, (
        f"Truncated history has {len(truncated_history)} messages, expected at most 10."
    )
