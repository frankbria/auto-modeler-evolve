# Issue #18 — [P5.1] Graceful error handling for the primary chat SSE stream

## Problem
`stream_response()` (`src/backend/api/chat.py:21847`) wraps `client.messages.stream(...)`
in `try/finally` with **no `except`**. A missing key / 429 / 529 / network error raises
mid-stream → user sees a truncated stream with no error card, and `finally` persists an
empty assistant message (corrupt history). Violates the "fail gracefully" UX north star.

## Adaptation note
CodeRabbit's plan (2026-06-14) predates #17 (landed 2026-06-23) which moved SSE dispatch
to a **handler map** in `frontend/lib/sse-handlers.ts` and added a reusable
`MonitoringNoteCard` for note-style cards. So instead of CodeRabbit's
page.tsx-branch + new ErrorCard + `ChatMessage.error` field + store action, the lazy/correct
adaptation reuses the existing note card via one handler entry. Backend changes unchanged.

## Backend — `src/backend/api/chat.py`, inside `stream_response()` (~21847)
1. **API-key guard** at top of the generator (before `try`): if neither
   `ANTHROPIC_API_KEY` nor `ANTHROPIC_AUTH_TOKEN` is set, yield
   `{'type':'error','message':'AI service is not configured. Please contact your administrator.'}`
   and `return`. (Pattern from `chat/narration.py:35`.)
2. **`error_occurred = False`** before `try`.
3. **`except` branches** between the `with ... stream` body and `finally`, each yields
   `{'type':'error','message': <friendly>}` and sets `error_occurred = True`:
   - `anthropic.AuthenticationError` → "Authentication failed — please check the API key configuration."
   - `anthropic.RateLimitError` → "The AI service is temporarily busy — please try again in a moment."
   - `anthropic.APIConnectionError` → "Couldn't reach the AI service — please check the connection and try again."
   - `anthropic.APIError` → "The AI service returned an error — please try again."
   - `except Exception  # noqa: BLE001` → "Something went wrong while generating a response — please try again."
4. **Conditional persistence** in `finally`: only append+commit the assistant message when
   `not error_occurred and full_response.strip()`.

## Frontend
5. `lib/types.ts`: add `"error"` to the `MonitoringNote["kind"]` union.
6. `components/chat/monitoring-note-card.tsx`: add `error: "⛔"` to `KIND_ICON`.
7. `lib/sse-handlers.ts`: add `error:` handler → `attachMonitoringNoteToLastMessage(
   {kind:"error", title:"Something went wrong", summary: json.message, tone:"critical"})`.
   (Required anyway: `test_sse_contract.py` fails CI if the backend emits `error` with no handler.)

## Tests
8. `src/backend/tests/test_chat_error_handling.py` (patterns from `test_api_key_chat.py`):
   - `stream()` raises `AuthenticationError` → SSE has a `type:"error"` event.
   - No key env vars → guard yields error event; client never `.stream()`s.
   - On error, the persisted conversation has **no** trailing empty assistant message.
   - `RateLimitError` → its specific friendly message appears.
9. Frontend: extend `__tests__/monitoring-note-card.test.tsx` for the `error` kind icon
   (handler existence is already pinned by `test_sse_contract.py`).

## Out of scope
- No new ErrorCard component / `ChatMessage.error` field / store action (superseded by #17).
