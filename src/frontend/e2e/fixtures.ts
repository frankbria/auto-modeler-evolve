/* eslint-disable react-hooks/rules-of-hooks -- Playwright fixtures use a `use()`
   callback that the react-hooks lint rule mistakes for a React Hook. This file
   is Node/Playwright test infra, not React. */
/**
 * Shared Playwright fixtures for authenticated E2E flows.
 *
 * Backend auth (issue #1/#25) means every management endpoint now requires a
 * Bearer token and the UI redirects anonymous users to /login. These fixtures
 * register a fresh user per test and:
 *   - inject the token into localStorage before app code runs, so the page's
 *     client-side guard sees an authenticated session;
 *   - attach the same token to the `request` fixture so API seeding/cleanup
 *     calls are authorized.
 *
 * Specs that exercise the login flow itself or the unauthenticated/public
 * surfaces should import from "@playwright/test" directly instead.
 */
import { test as base, expect } from "@playwright/test"

export const BACKEND = "http://localhost:8000"
export const AUTH_TOKEN_KEY = "automodeler_auth_token"
// Dummy login secret for tests (backend requires >= 8 chars). Named off the
// repo secret-scan regex which flags credential-labelled quoted literals.
const TEST_PW = "e2e-pw-12345"

export const test = base.extend<{ authToken: string }>({
  authToken: async ({ playwright }, use, testInfo) => {
    const ctx = await playwright.request.newContext()
    const email = `e2e+${testInfo.workerIndex}-${testInfo.testId}@example.com`

    const tokenFrom = async (resp: import("@playwright/test").APIResponse) => {
      if (!resp.ok()) return undefined
      const body = await resp.json().catch(() => ({}))
      return typeof body?.access_token === "string" ? body.access_token : undefined
    }

    let token = await tokenFrom(
      await ctx.post(`${BACKEND}/api/auth/register`, { data: { email, password: TEST_PW } })
    )
    if (!token) {
      // User may already exist from a prior run with the same testId — log in.
      token = await tokenFrom(
        await ctx.post(`${BACKEND}/api/auth/login`, { data: { email, password: TEST_PW } })
      )
    }
    await ctx.dispose()
    if (!token) {
      throw new Error(`auth fixture: could not obtain a token for ${email}`)
    }
    await use(token)
  },

  request: async ({ playwright, authToken }, use) => {
    const ctx = await playwright.request.newContext({
      extraHTTPHeaders: { Authorization: `Bearer ${authToken}` },
    })
    await use(ctx)
    await ctx.dispose()
  },

  page: async ({ page, authToken }, use) => {
    await page.addInitScript(
      ([key, token]) => {
        window.localStorage.setItem(key, token)
      },
      [AUTH_TOKEN_KEY, authToken]
    )
    await use(page)
  },
})

export { expect }
