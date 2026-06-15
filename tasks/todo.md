# Issue #25 — Frontend authentication + auth-aware API layer

**Issue:** [P1.1-frontend] Frontend authentication + auth-aware API layer (follow-up to #1)
**Plan source:** issue body (adapted to codebase)
**Branch:** `feat/25-frontend-auth`

## Context (from codebase exploration)
- Backend auth is live & backend-native:
  - `POST /api/auth/register` → `{access_token, token_type:"bearer", user:{id,email,name}}`
  - `POST /api/auth/login` → same `TokenResponse`
  - `GET /api/auth/me` → `{id,email,name}` (requires `Authorization: Bearer`)
  - CORS allows all origins + `Authorization` header.
- `lib/api.ts`: one `API_URL` const, ~100 methods each calling raw `fetch()` → add ONE `apiFetch` chokepoint.
- `lib/store.ts`: Zustand, no auth slice. No `middleware.ts`, no auth context. Root layout is plain.
- IDOR: `app/predict/[id]/page.tsx:309` `CompareModelsCard` → `api.deploy.listByProject(projectId)` leaks sibling deployments.

## Key design decisions
1. **Token storage:** `localStorage` via a small `lib/auth-token.ts` helper (get/set/clear). Client-only app → no SSR token need.
2. **Route protection:** client-side guard (recommended over `middleware.ts`, which runs server-side and cannot read a localStorage token; the app has no SSR-protected data). A `<RequireAuth>` wrapper guards `/` and `/project/[id]`; `/login` and `/predict/[id]` stay public.
3. **401 handling:** `apiFetch` detects 401 → clears token + redirects to `/login` (single chokepoint).
4. **IDOR fix:** remove the sibling-enumeration (`listByProject`) from the public predict page. The "Compare Model Versions" feature is owner-only by nature; the public page keeps only the single-deployment public fetch.

## Steps (TDD: RED → GREEN → REFACTOR)

- [x] **Step 1 — Token helper + auth-aware fetch chokepoint**
  - `lib/auth-token.ts`: `getToken/setToken/clearToken` (localStorage, SSR-safe guard).
  - `lib/api.ts`: add `apiFetch(url, options)` injecting `Authorization: Bearer <token>`; on 401 → clear token + redirect `/login`. Route all existing methods through it (keep public `auth.register/login` token-free).
  - `lib/api.ts`: add `api.auth.register/login/me`.
  - `lib/types.ts`: add `User`, `AuthResponse`.
  - Tests: `__tests__/api-auth.test.tsx` — header attached on authed call, absent on login, 401 clears token.

- [x] **Step 2 — Zustand auth slice**
  - `lib/store.ts`: `user`, `token`, `isAuthenticated`, `login()`, `register()`, `logout()`, `loadCurrentUser()`.
  - Tests: `__tests__/auth-store.test.tsx`.

- [x] **Step 3 — Login / register UI**
  - `app/login/page.tsx`: thin email+password form (toggle register), Nova/Hugeicons, calls store actions, redirects on success.
  - Tests: `__tests__/login-page.test.tsx`.

- [x] **Step 4 — Route protection + nav**
  - `components/auth/require-auth.tsx`: client guard; unauthenticated → `/login`.
  - Wrap `app/page.tsx` and `app/project/[id]/page.tsx`.
  - `app/layout.tsx`: show user email + logout when authenticated.
  - Tests: `__tests__/require-auth.test.tsx`.

- [x] **Step 5 — Fix public predict IDOR**
  - `app/predict/[id]/page.tsx`: remove `CompareModelsCard` sibling enumeration (`listByProject`). Public page loads only the single shared deployment.
  - Tests: `__tests__/predict-page-public.test.tsx` — asserts no owner-scoped call.

- [x] **Step 6 — E2E**
  - `e2e/auth.spec.ts`: register/login → protected action; `/predict/[id]` reachable without login.

## Acceptance criteria
- [ ] Unauthenticated users redirected to login for protected routes; `/predict/[id]` stays public.
- [ ] All authenticated API calls carry a Bearer token; 401 triggers re-login.
- [ ] Public predict page calls no owner-scoped endpoint; exposes no sibling-deployment data.
- [ ] Unit tests assert the auth header is attached; E2E covers login → protected action.
