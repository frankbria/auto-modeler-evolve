import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E configuration for AutoModeler.
 *
 * The tests assume:
 *   - Frontend: http://localhost:3000  (Next.js dev server)
 *   - Backend:  http://localhost:8000  (FastAPI uvicorn)
 *
 * In CI, both servers are started via the webServer config below.
 * Locally, start them manually before running `npx playwright test`.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // sequential to avoid race conditions on shared SQLite
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: "html",

  use: {
    baseURL: "http://localhost:3000",
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
      command:
        "cd ../backend && uv run uvicorn main:app --host 0.0.0.0 --port 8000",
      url: "http://localhost:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command: "npm run dev",
      url: "http://localhost:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
});
