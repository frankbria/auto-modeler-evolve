import { defineConfig, devices } from "@playwright/test";
import { BACKEND, FRONTEND } from "./e2e/env";

/**
 * Playwright E2E configuration for AutoModeler.
 *
 * Targets default to the local dev servers (backend :8000, frontend :3000) but
 * are env-overridable via E2E_BACKEND_URL / E2E_BASE_URL (see e2e/env.ts) so a
 * run can dodge an in-flight port. The servers below are spawned on the ports
 * derived from those URLs, and e2e/global-setup.ts fails fast if the backend
 * port is held by a non-AutoModeler process.
 *
 * Locally, existing servers are reused; in CI both are started fresh.
 */
const backendPort = new URL(BACKEND).port || "8000";
const frontendPort = new URL(FRONTEND).port || "3000";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // sequential to avoid race conditions on shared SQLite
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: "html",
  globalSetup: "./e2e/global-setup.ts",

  use: {
    baseURL: FRONTEND,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  webServer: [
    {
      command: `cd ../backend && uv run uvicorn main:app --host 0.0.0.0 --port ${backendPort}`,
      url: `${BACKEND}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command: `npm run dev -- --port ${frontendPort}`,
      url: FRONTEND,
      // Point the spawned dev server at the (possibly relocated) backend so the
      // app, the fixtures, and the preflight all agree on one origin.
      env: { NEXT_PUBLIC_API_URL: BACKEND },
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
});
