/**
 * E2E tests for Phase 1/2: Data Upload & Preview
 *
 * Tests the CSV upload flow — both drag-and-drop (simulated via setInputFiles)
 * and the "Load sample data" shortcut. Verifies that the data preview, row count
 * badge, column stats, and insights all appear after upload.
 */

import { test, expect } from "@playwright/test"
import path from "path"

const BACKEND = "http://localhost:8000"
const SAMPLE_CSV = path.join(
  __dirname,
  "../../backend/data/sample/sample_sales.csv"
)

async function cleanProjects(request: import("@playwright/test").APIRequestContext) {
  const list = await request.get(`${BACKEND}/api/projects`)
  if (!list.ok()) return
  const projects: Array<{ id: string }> = await list.json()
  for (const p of projects) {
    await request.delete(`${BACKEND}/api/projects/${p.id}`)
  }
}

async function createProject(
  request: import("@playwright/test").APIRequestContext,
  name = "Upload E2E"
): Promise<string> {
  const res = await request.post(`${BACKEND}/api/projects`, {
    data: { name },
  })
  const body = await res.json()
  return body.id
}

test.describe("Data Upload — sample data shortcut", () => {
  let projectId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    projectId = await createProject(request)
  })

  test("upload panel shows dropzone and sample link", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    // Dropzone copy
    await expect(page.getByText(/Drop your CSV here/i)).toBeVisible({
      timeout: 8_000,
    })
    // Sample data link
    await expect(
      page.getByRole("button", { name: /load sample/i })
    ).toBeVisible()
  })

  test("load sample data shows data preview with 200 rows", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: /load sample/i }).click()
    // Row count badge in the data panel header
    await expect(page.getByText("200 rows", { exact: true })).toBeVisible({
      timeout: 10_000,
    })
  })

  test("load sample data shows column count badge", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: /load sample/i }).click()
    // Should show 5 columns badge
    await expect(page.getByText("5 columns", { exact: true })).toBeVisible({
      timeout: 10_000,
    })
  })

  test("load sample data activates the data preview tab with a table", async ({
    page,
  }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: /load sample/i }).click()
    // Wait for the data table header row to appear (column names visible)
    await expect(page.getByRole("table")).toBeVisible({ timeout: 10_000 })
    // The sample CSV has a "revenue" column
    await expect(page.getByRole("columnheader", { name: /revenue/i })).toBeVisible({
      timeout: 5_000,
    })
  })

  test("load sample data shows column statistics cards", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: /load sample/i }).click()
    // Wait for data to load, then check column stats section
    await expect(page.getByText("Column Statistics")).toBeVisible({
      timeout: 10_000,
    })
  })

  test("load sample data sends a narration message in chat", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: /load sample/i }).click()
    // narrate_upload auto-injects a message about the dataset into the chat panel
    // Scope to the chat area to avoid matching the data panel header
    const chatPanel = page.locator(".flex.flex-col.border-r")
    await expect(
      chatPanel.getByText(/sample.sales/i).first()
    ).toBeVisible({ timeout: 12_000 })
  })
})

test.describe("Data Upload — file input", () => {
  let projectId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    projectId = await createProject(request, "FileUpload E2E")
  })

  test("uploading a CSV via file input shows 200-row badge", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    // Wait for upload panel to be visible
    await expect(page.getByText(/Drop your CSV here/i)).toBeVisible({
      timeout: 8_000,
    })
    // The hidden file <input> inside the dropzone
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles(SAMPLE_CSV)
    // After upload, the data preview should show row count
    await expect(page.getByText("200 rows", { exact: true })).toBeVisible({
      timeout: 12_000,
    })
  })

  test("uploading a CSV shows the filename in the data panel header", async ({
    page,
  }) => {
    await page.goto(`/project/${projectId}`)
    await expect(page.getByText(/Drop your CSV here/i)).toBeVisible({
      timeout: 8_000,
    })
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles(SAMPLE_CSV)
    // The data panel header has an <h2> with the filename
    await expect(page.getByRole("heading", { name: /sample_sales\.csv/i })).toBeVisible({
      timeout: 12_000,
    })
  })
})

test.describe("Data tabs — after upload", () => {
  let projectId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    projectId = await createProject(request, "Tabs E2E")
    // Load sample data via API for speed
    await request.post(`${BACKEND}/api/data/sample`, {
      data: { project_id: projectId },
    })
  })

  test("all six tabs are visible once data is loaded", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    // Dataset state is restored from DB on mount (project.dataset_id → preview fetch)
    for (const label of ["Data", "Features", "Importance", "Models", "Validate", "Deploy"]) {
      await expect(page.getByRole("button", { name: label })).toBeVisible({
        timeout: 10_000,
      })
    }
  })

  test("Features tab shows feature suggestions after upload", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    // Dataset state is restored from DB on mount — tabs are immediately visible
    await page.getByRole("button", { name: "Features" }).click({ timeout: 10_000 })
    // Should see either the section heading or the loading text
    await expect(
      page.getByRole("heading", { name: /Feature Suggestions/i })
    ).toBeVisible({ timeout: 10_000 })
  })
})
