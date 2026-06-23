# Issue #15 — Autospec the Anthropic mock + LLM contract test

## Problem
80 test files mock `anthropic.Anthropic` with permissive `MagicMock` (`text_stream = iter([...])`, zero autospec). The LLM integration contract is unverified — a wrong model id, malformed `messages`, or removed SDK method passes CI. Also contradicts CLAUDE.md "no mocking — real services".

## Key findings (verified empirically, SDK 0.84.0)
- Real call: `api/chat.py:21850` `client.messages.stream(model=MODEL, max_tokens=MAX_TOKENS, system=..., messages=api_messages)`; `MODEL="claude-haiku-4-5-20251001"`, `MAX_TOKENS=1024` (chat.py:38-39).
- `api_messages` = `[{"role": ..., "content": ...}]` (chat.py:6071).
- The `stream` call is unconditional in `send_message` → a bare project + plain message reaches it (no dataset/model needed).
- **Naive `create_autospec(anthropic.Anthropic)` does NOT expose `.messages.stream`** (`.messages` is a cached_property). Working spec: `create_autospec(anthropic.Anthropic, instance=True)` + assign `create_autospec(anthropic.resources.Messages, instance=True)` to `.messages` → signature validation works.

## Plan (TDD, scope = contract test + reusable fixture; confirmed by user)
1. **conftest fixture** `mock_anthropic` — reusable autospec'd Anthropic patch.
2. **`tests/test_anthropic_contract.py`** — contract test (model id + messages schema) + guard test (autospec rejects bad method/args).
3. **CLAUDE.md** — document the Anthropic-boundary mocking exception.
4. Run full suite + ruff/black, demo, PR, merge.

## Out of scope (Known Limitation)
Mass-converting all 80 permissive card-test mocks — they verify SSE cards/regex, not the LLM contract.

## Acceptance criteria
- [ ] `create_autospec` used so wrong args/method names fail
- [ ] Contract test asserts model id + schema-valid messages payload
- [ ] Mocking exception documented in CLAUDE.md
