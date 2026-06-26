/**
 * Single source of truth for E2E target URLs.
 *
 * Local dev reality: ports 8000/3000 are often already in flight — sometimes by
 * THIS app's servers (good, reused), sometimes by an unrelated project (bad,
 * silently wrong). Override both when you need to dodge a clash:
 *
 *   E2E_BACKEND_URL=http://localhost:8100 E2E_BASE_URL=http://localhost:3100 npm run test:e2e
 *
 * playwright.config.ts spawns/relocates its own servers to match these, and
 * global-setup.ts fails fast if the backend port is held by a foreign app.
 */
export const BACKEND = process.env.E2E_BACKEND_URL ?? "http://localhost:8000"
export const FRONTEND = process.env.E2E_BASE_URL ?? "http://localhost:3000"
