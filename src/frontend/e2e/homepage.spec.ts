import { test, expect } from "@playwright/test";

const BACKEND = "http://localhost:8000";

// Helper: delete all projects via API before each test so state is clean
async function cleanProjects(request: import("@playwright/test").APIRequestContext) {
  const list = await request.get(`${BACKEND}/api/projects`);
  if (!list.ok()) return;
  const projects: Array<{ id: string }> = await list.json();
  for (const p of projects) {
    await request.delete(`${BACKEND}/api/projects/${p.id}`);
  }
}

test.describe("AutoModeler homepage", () => {
  test.beforeEach(async ({ request }) => {
    await cleanProjects(request);
  });

  test("shows empty-state panel with no projects", async ({ page }) => {
    await page.goto("/");
    // The empty-state card should be visible when there are no projects
    await expect(page.getByText(/AutoModeler/i).first()).toBeVisible();
    // Empty state text — matches the onboarding copy in the frontend
    await expect(page.getByText(/No projects yet/i)).toBeVisible();
  });

  test("can create a new project", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /new project/i }).click();
    // Form appears — fill in a name and submit
    await page.getByPlaceholder("Project name").fill("My E2E Project");
    await page.getByRole("button", { name: /^create$/i }).click();
    // After creation, the project card should appear
    await expect(page.getByText("My E2E Project")).toBeVisible({
      timeout: 5_000,
    });
  });

  test("created project appears in list", async ({ page, request }) => {
    // Create via API to avoid UI timing issues
    const res = await request.post(`${BACKEND}/api/projects`, {
      data: { name: "E2E Test Project" },
    });
    expect(res.ok()).toBeTruthy();

    await page.goto("/");
    await expect(page.getByText("E2E Test Project")).toBeVisible();
  });

  test("can delete a project", async ({ page, request }) => {
    await request.post(`${BACKEND}/api/projects`, {
      data: { name: "Delete Me" },
    });

    await page.goto("/");
    await expect(page.getByText("Delete Me")).toBeVisible();

    // Accept the confirm() dialog before clicking delete
    page.on("dialog", (dialog) => dialog.accept());

    const card = page.locator('[data-testid="project-card"]', {
      hasText: "Delete Me",
    });
    await card.getByRole("button", { name: /delete/i }).click();

    await expect(page.getByText("Delete Me")).not.toBeVisible({ timeout: 3_000 });
  });
});

test.describe("Project workspace", () => {
  let projectId: string;

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request);
    const res = await request.post(`${BACKEND}/api/projects`, {
      data: { name: "Workspace E2E" },
    });
    const body = await res.json();
    projectId = body.id;
  });

  test("workspace loads with chat panel", async ({ page }) => {
    await page.goto(`/project/${projectId}`);
    // Chat input should be present
    await expect(page.getByPlaceholder(/ask|message|chat/i)).toBeVisible({
      timeout: 10_000,
    });
  });

  test("can upload sample CSV via load-sample button", async ({ page }) => {
    await page.goto(`/project/${projectId}`);
    const sampleBtn = page.getByRole("button", { name: /load sample/i });
    await expect(sampleBtn).toBeVisible({ timeout: 5_000 });
    await sampleBtn.click();
    // After loading sample data, the data preview should show row count badge
    await expect(page.getByText("200 rows", { exact: true })).toBeVisible({
      timeout: 10_000,
    });
  });
});
