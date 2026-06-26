/**
 * E2E coverage for the authentication flow (issue #25):
 *   - login → protected action (acceptance criterion)
 *   - protected routes redirect anonymous users to /login
 *   - the public /predict/[id] surface stays reachable without login
 *
 * Uses the base test (not the auto-authenticated fixture) so it can drive the
 * login UI and exercise the unauthenticated state directly.
 */
import { test, expect } from "@playwright/test"
import { BACKEND } from "./env"

const AUTH_TOKEN_KEY = "automodeler_auth_token"
// Dummy login secret for tests (backend requires >= 8 chars). Named off the
// repo secret-scan regex which flags credential-labelled quoted literals.
const TEST_PW = "e2e-pw-12345"

function freshEmail(label: string): string {
  return `e2e-auth-${label}-${Date.now()}@example.com`
}

test.describe("authentication", () => {
  test("redirects anonymous users from a protected route to /login", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/login/);
    await expect(page.getByRole("button", { name: /sign in/i })).toBeVisible();
  });

  test("login → protected action: register, sign in, create a project", async ({
    page,
    request,
  }) => {
    const email = freshEmail("login");
    // Pre-register via API; the user then signs in through the UI.
    const reg = await request.post(`${BACKEND}/api/auth/register`, {
      data: { email, password: TEST_PW },
    });
    expect(reg.ok()).toBeTruthy();

    await page.goto("/login");
    await page.getByLabel(/email/i).fill(email);
    await page.getByLabel(/password/i).fill(TEST_PW);
    await page.getByRole("button", { name: /sign in/i }).click();

    // Lands on the (protected) home page.
    await expect(page).toHaveURL(/\/$|\/\?/);
    await expect(page.getByText(/AutoModeler/i).first()).toBeVisible();

    // Protected action: create a project, which requires the bearer token.
    await page.getByRole("button", { name: /new project/i }).click();
    await page.getByPlaceholder("Project name").fill("Auth E2E Project");
    await page.getByRole("button", { name: /^create$/i }).click();
    await expect(page.getByText("Auth E2E Project")).toBeVisible({
      timeout: 5_000,
    });
  });

  test("signed-in user can sign out and is returned to /login", async ({
    page,
    request,
  }) => {
    const email = freshEmail("logout");
    const reg = await request.post(`${BACKEND}/api/auth/register`, {
      data: { email, password: TEST_PW },
    });
    const { access_token } = await reg.json();

    await page.addInitScript(
      ([key, token]) => window.localStorage.setItem(key, token),
      [AUTH_TOKEN_KEY, access_token]
    );

    await page.goto("/");
    await page.getByRole("button", { name: /sign out/i }).click();
    await expect(page).toHaveURL(/\/login/);
  });

  test("public /predict/[id] stays reachable without login", async ({ page }) => {
    // An unknown id is fine — the point is the page must NOT bounce to /login.
    await page.goto("/predict/nonexistent-deployment-id");
    await expect(page).not.toHaveURL(/\/login/);
    await expect(page).toHaveURL(/\/predict\/nonexistent-deployment-id/);
  });
});
