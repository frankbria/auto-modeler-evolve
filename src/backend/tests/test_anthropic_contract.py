"""LLM integration contract tests (issue #15).

The chat suite mocks ``anthropic.Anthropic`` everywhere with permissive
``MagicMock`` objects that verify SSE cards and regex intent, never the LLM
call itself. So a wrong model id, a malformed ``messages`` array, or a removed
SDK method would sail through CI.

These tests close that gap:

- ``test_chat_invokes_llm_with_contract`` drives a real chat request through
  ``send_message`` under the autospec'd ``mock_anthropic`` fixture and asserts
  the *outgoing* call matches the contract: the pinned model id and a
  schema-valid ``messages`` payload (``role`` ∈ {user, assistant}, ``content``
  a string).

- ``test_autospec_enforces_method_and_signature`` proves the autospec is real —
  a bogus method name raises ``AttributeError`` and a call missing a required
  kwarg raises ``TypeError`` — i.e. an SDK upgrade that renames ``stream`` or
  changes its signature fails loudly instead of passing.
"""

import anthropic
import pytest
from anthropic.resources import Messages
from unittest.mock import create_autospec

from api.chat import MAX_TOKENS, MODEL


async def test_chat_invokes_llm_with_contract(client, mock_anthropic):
    """A normal chat turn calls ``messages.stream`` with a contract-valid payload."""
    proj = await client.post("/api/projects", json={"name": "Contract"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    resp = await client.post(f"/api/chat/{project_id}", json={"message": "hello there"})
    assert resp.status_code == 200

    call = mock_anthropic.messages.stream.call_args
    assert call is not None, "send_message never called anthropic messages.stream"

    kwargs = call.kwargs
    # Pinned model id — guards against a silent model swap.
    assert kwargs["model"] == MODEL == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == MAX_TOKENS
    assert isinstance(kwargs["system"], str) and kwargs["system"].strip()

    messages = kwargs["messages"]
    assert isinstance(messages, list) and messages, "messages payload must be non-empty"
    for m in messages:
        assert set(m) == {"role", "content"}, f"unexpected message keys: {set(m)}"
        assert m["role"] in {"user", "assistant"}, f"invalid role: {m['role']!r}"
        assert isinstance(m["content"], str)
    # The turn's user message must be the final entry handed to the model.
    assert messages[-1] == {"role": "user", "content": "hello there"}


def test_autospec_enforces_method_and_signature():
    """The spec rejects unknown methods and bad signatures (wrong calls fail)."""
    spec_client = create_autospec(anthropic.Anthropic, instance=True)
    spec_client.messages = create_autospec(Messages, instance=True)

    # A removed/renamed SDK method must raise instead of silently passing.
    with pytest.raises(AttributeError):
        spec_client.messages.streamz  # noqa: B018 — attribute access is the assertion

    # A call missing required kwargs (e.g. ``max_tokens``/``messages``) must raise.
    with pytest.raises(TypeError):
        spec_client.messages.stream(model="claude-haiku-4-5-20251001")

    # The real call shape is accepted.
    spec_client.messages.stream(
        model=MODEL, max_tokens=MAX_TOKENS, system="sys", messages=[]
    )
    assert spec_client.messages.stream.call_args.kwargs["model"] == MODEL
