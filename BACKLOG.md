# Evolution Backlog

Living document for coordinating between bot instances and tracking ideation.
**Read this before starting work. Write your focus before implementing.**

## Currently Working On

<!-- Each bot writes here BEFORE starting implementation. Format: -->
<!-- ## [Bot ID / Timestamp] — [Focus Area] -->
<!-- Brief description of what you're doing this session. -->
<!-- Remove your entry when you commit your session wrap-up. -->

## Day 2 (20:05) — Error Resilience Audit + Correlation Heatmap

Track A: Systematically test failure modes (corrupt CSV, empty dataset, single-row,
all-null columns, training on bad data). Fix any issues found and verify helpful
user-facing messages. Track B: Add interactive correlation heatmap visualization
triggered naturally through chat ("show me correlations").


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

- E2E test suite build-out (upload/training/deploy) — Day 2 (10:00) — 33 Playwright tests; fixed 2 UX bugs (dataset restore + ModelTrainingPanel runs restore); 33/33 pass
- Smarter chat orchestration (prompts.py + narration.py) — Day 2 (16:08) — auto-inject upload/training messages into chat; 44 tests; 255 total pass
