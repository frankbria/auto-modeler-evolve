"""Backend/frontend SSE event contract (#17).

The chat stream emits ~185 structured event types; the frontend dispatches them
by `type`. Before #17 the dispatcher was an untested 239-branch if/else chain,
so an event the backend emitted with no frontend handler (e.g. ``readiness``,
``drift``) was silently dropped and stayed green in CI.

This test pins the contract from the backend side: **every event type emitted by
``api/chat.py`` must have a handler in the frontend's SSE handler map.** It reads
both source files directly (no running server, no JS runtime) so it fails the
moment a new backend emit lacks a frontend handler.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_CHAT = Path(__file__).resolve().parent.parent / "api" / "chat.py"
_FRONTEND_HANDLERS = (
    Path(__file__).resolve().parents[2] / "frontend" / "lib" / "sse-handlers.ts"
)

# Emitted by the stream as plumbing, not cards — the frontend reader consumes
# `token`/`done` directly; these are allowed to exist without a map entry.
_PLUMBING = {"token", "done"}


def _emitted_event_types() -> set[str]:
    text = _BACKEND_CHAT.read_text()
    # Only real SSE frames: `json.dumps({'type': 'foo', ...})`. This deliberately
    # excludes nested dicts that merely carry a `type` field (e.g. alert-list
    # items like {'type': 'no_predictions'}), which are payload data, not events.
    return set(
        re.findall(r"""json\.dumps\(\{['"]type['"]:\s*['"]([a-z_0-9]+)['"]""", text)
    )


def _handled_event_types() -> set[str]:
    text = _FRONTEND_HANDLERS.read_text()
    # Handlers live in the `return { ... }` map as `  foo: (json) => {` / `foo: () => {`.
    return set(re.findall(r"^\s{2}([a-z_0-9]+):\s*\([^)]*\)\s*=>", text, re.MULTILINE))


def test_frontend_handles_every_emitted_event() -> None:
    emitted = _emitted_event_types()
    handled = _handled_event_types()
    assert handled, "Could not parse any handlers from sse-handlers.ts"

    dropped = sorted(emitted - handled - _PLUMBING)
    assert not dropped, (
        "Backend emits SSE event(s) with no frontend handler (silently dropped, "
        f"the #17 bug): {dropped}. Add a handler in frontend/lib/sse-handlers.ts "
        "or stop emitting the event."
    )


def test_previously_dropped_events_are_now_handled() -> None:
    handled = _handled_event_types()
    for t in ("readiness", "drift", "health", "alerts", "history"):
        assert t in handled, f"'{t}' regressed to having no frontend handler"


def test_handler_map_has_no_duplicate_keys() -> None:
    text = _FRONTEND_HANDLERS.read_text()
    keys = re.findall(r"^\s{2}([a-z_0-9]+):\s*\([^)]*\)\s*=>", text, re.MULTILINE)
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"Duplicate handler keys (dead code): {sorted(dupes)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
