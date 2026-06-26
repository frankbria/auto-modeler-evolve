# Issue #23 — [P7.1] Repo hygiene, docs drift & a11y follow-ups

Self-authored plan (no plan in issue). LOW severity. Mostly deletions + docs.

## A. Remove tracked artifacts
- [x] `git rm --cached` 7 `src/backend/models/__pycache__/*.pyc` (already gitignored)
- [x] Delete `src/backend/tests/debug_imbalance.py`
- [x] Delete `src/frontend/__tests__/check_textdecoder.test.ts`
- [x] Delete `src/frontend/components/deploy/feature-interaction-heatmap-card.tsx` (2-line stub, unreferenced)
- [x] Delete unused create-next-app SVGs in `src/frontend/public/` (file/globe/next/vercel/window — confirmed unreferenced)
- [x] Replace boilerplate `src/frontend/README.md` with a 2-line pointer to root README
- [x] Add `/data` to `src/frontend/.gitignore` (no joblib currently tracked — belt & suspenders)

## B. lucide-react
- [x] Remove `lucide-react` from `package.json` (zero usages); update lockfile
- [x] Add `no-restricted-imports` eslint rule banning lucide-react (Hugeicons-only)

## C. Docs drift
- [x] README: `Next.js 15` → `Next.js 16`
- [x] README: drop hard-coded test counts (3 spots) → link to CI Actions page
- [x] Add `LICENSE` file (MIT) to back the README claim
- [x] README API table: `GET /api/data/{id}/query` → `POST`
- [x] CLAUDE.md scripts tree: add `demo.py` + `run_evolve_cron.sh`

## D. Exception-detail leak (data.py)
- [x] Genericize the 7 `status_code=500, detail="Could not read dataset: {exc}"` sites
      → generic client message + `logger.exception` server-side. (TDD: one test asserting
      500 body carries no raw exception text.)

## E. a11y / error boundary
- [x] Add `app/error.tsx` + `app/global-error.tsx` (cheap, real win)
- [x] Open follow-up issue tracking heavier a11y (form-validation feedback, native
      `confirm()` → modal, heatmap ARIA) — acceptance criterion is "tracked", not "done" → #65

## Out of scope (deliberate, ponytail)
- 400-level parse-error messages left as-is: they are user-actionable CSV/SQL feedback,
  not internal-state leaks.
