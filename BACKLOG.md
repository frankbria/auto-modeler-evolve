# Evolution Backlog

Living document for coordinating between bot instances and tracking ideation.
**Read this before starting work. Write your focus before implementing.**

## Currently Working On

<!-- Each bot writes here BEFORE starting implementation. Format: -->
<!-- ## [Bot ID / Timestamp] — [Focus Area] -->
<!-- Brief description of what you're doing this session. -->
<!-- Remove your entry when you commit your session wrap-up. -->

## Day 2 (16:08) — Smarter Chat Orchestration (Track B) + Quality Hardening (Track A)

Implementing `chat/prompts.py` (algorithm intros, metric glossary, proactive insight
prompt builder) and `chat/narration.py` (event narrators that auto-inject context into
the conversation after upload and training). Goal: make the chat feel like a proactive
colleague, not just a Q&A bot. Also expanding backend test coverage for the new modules.

## Ideas to Explore

Ideas discovered during sessions. Pick from here or add new ones.

- Full E2E test suite covering upload → explore → train → deploy → predict flow
- Gap analysis: verify every [x] spec item actually works end-to-end
- Integration with XGBoost / LightGBM for better model recommendations
- prompts.py and narration.py modules for richer chat experience
- Self-demo script that exercises the full platform and captures output
- Excel / Google Sheets upload support
- Template projects for common use cases (sales forecast, churn prediction)
- Interactive correlation heatmap visualization
- Multi-dataset join/merge through conversation

## Recently Completed

<!-- Move items here after implementation. Format: -->
<!-- - [Description] — Day N (HH:MM) — [1-line outcome] -->

(none yet — Phase 8 starts next session)
