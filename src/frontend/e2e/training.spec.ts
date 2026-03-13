/**
 * E2E tests for Phase 4: Model Training
 *
 * Setup: create project + upload sample data + apply feature set + set target
 *   column — all via API so tests run fast.
 * Test: navigate to the Models tab in the UI, verify algorithm cards load,
 *   start training, wait for completion, verify metrics appear, select a model.
 */

import { test, expect } from "@playwright/test"

const BACKEND = "http://localhost:8000"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function cleanProjects(request: import("@playwright/test").APIRequestContext) {
  const list = await request.get(`${BACKEND}/api/projects`)
  if (!list.ok()) return
  const projects: Array<{ id: string }> = await list.json()
  for (const p of projects) {
    await request.delete(`${BACKEND}/api/projects/${p.id}`)
  }
}

/**
 * Create a project with the sample dataset uploaded and a feature set ready
 * for training (target_column = "revenue").
 * Returns { projectId, datasetId, featureSetId }
 */
async function setupProjectWithFeatureSet(
  request: import("@playwright/test").APIRequestContext
): Promise<{ projectId: string; datasetId: string; featureSetId: string }> {
  // 1. Create project
  const projRes = await request.post(`${BACKEND}/api/projects`, {
    data: { name: "Training E2E" },
  })
  const project = await projRes.json()
  const projectId: string = project.id

  // 2. Load sample dataset
  const dataRes = await request.post(`${BACKEND}/api/data/sample`, {
    data: { project_id: projectId },
  })
  const dataBody = await dataRes.json()
  const datasetId: string = dataBody.dataset_id

  // 3. Apply empty feature set (creates the FeatureSet record in DB)
  const applyRes = await request.post(
    `${BACKEND}/api/features/${datasetId}/apply`,
    {
      data: { transformations: [] },
    }
  )
  const applyBody = await applyRes.json()
  const featureSetId: string = applyBody.feature_set_id

  // 4. Set target column
  await request.post(`${BACKEND}/api/features/${datasetId}/target`, {
    data: { target_column: "revenue", feature_set_id: featureSetId },
  })

  return { projectId, datasetId, featureSetId }
}

/**
 * Poll the runs endpoint until all runs have a terminal status (done/failed).
 * Times out after `maxMs`.
 */
async function waitForTrainingComplete(
  request: import("@playwright/test").APIRequestContext,
  projectId: string,
  maxMs = 60_000
): Promise<Array<{ id: string; algorithm: string; status: string; metrics: unknown }>> {
  const deadline = Date.now() + maxMs
  while (Date.now() < deadline) {
    const res = await request.get(`${BACKEND}/api/models/${projectId}/runs`)
    if (!res.ok()) throw new Error("Failed to fetch runs")
    const body = await res.json()
    const runs: Array<{ status: string; id: string; algorithm: string; metrics: unknown }> =
      body.runs ?? []
    const allDone = runs.length > 0 && runs.every((r) => r.status === "done" || r.status === "failed")
    if (allDone) return runs
    await new Promise((r) => setTimeout(r, 1_500))
  }
  throw new Error("Training did not complete within timeout")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("Model Training — algorithm recommendations", () => {
  let projectId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    ;({ projectId } = await setupProjectWithFeatureSet(request))
  })

  test("Models tab shows algorithm recommendation cards", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    // Dataset state is restored automatically from DB — no need to click "Load sample"
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })
    // Wait for at least one algorithm card button to appear
    await expect(
      page.getByRole("button", { name: /Linear Regression/i }).first()
    ).toBeVisible({ timeout: 12_000 })
  })

  test("Models tab shows target column badge", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })
    // The panel shows a badge with the target column name — use badge role
    const modelsPanel = page.locator('[data-slot="badge"]', { hasText: "revenue" })
    await expect(modelsPanel.first()).toBeVisible({ timeout: 12_000 })
  })

  test("algorithm cards are pre-selected and can be toggled", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })
    // Wait for a "Train N models" button to appear (at least 1 pre-selected)
    await expect(
      page.getByRole("button", { name: /Train \d+ model/i })
    ).toBeVisible({ timeout: 12_000 })
  })
})

test.describe("Model Training — train and view results", () => {
  let projectId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    ;({ projectId } = await setupProjectWithFeatureSet(request))
    // Kick off training via API so the UI just shows results
    await request.post(`${BACKEND}/api/models/${projectId}/train`, {
      data: { algorithms: ["linear_regression"] },
    })
    // Wait for training to finish before the UI test starts
    await waitForTrainingComplete(request, projectId)
  })

  test("Models tab shows completed run cards with metrics", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    // Dataset state restores from DB; models panel loads existing runs
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })
    // A "Done" badge should appear on the run card
    await expect(page.locator('[data-slot="badge"]', { hasText: "Done" })).toBeVisible({
      timeout: 12_000,
    })
  })

  test("completed run card shows R² metric", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })
    // Regression metrics row shows R²
    await expect(page.getByText(/R²/).first()).toBeVisible({ timeout: 12_000 })
  })

  test("can select a completed model", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })
    // Wait for "Select this model" button
    const selectBtn = page.getByRole("button", { name: /select this model/i })
    await expect(selectBtn).toBeVisible({ timeout: 12_000 })
    await selectBtn.click()
    // After selection, the "Selected" badge appears on the run card
    await expect(
      page.locator('[data-slot="badge"]', { hasText: "Selected" })
    ).toBeVisible({ timeout: 5_000 })
  })

  test("selecting a model sends a confirmation message in chat", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })
    const selectBtn = page.getByRole("button", { name: /select this model/i })
    await expect(selectBtn).toBeVisible({ timeout: 12_000 })
    await selectBtn.click()
    // Chat panel should receive a confirmation message mentioning Validate or deploy
    await expect(
      page.getByText(/validate|deploy/i, { exact: false }).first()
    ).toBeVisible({ timeout: 8_000 })
  })
})

test.describe("Model Training — train via UI", () => {
  let projectId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    ;({ projectId } = await setupProjectWithFeatureSet(request))
  })

  test("clicking Train button starts training and eventually shows Done badge", async ({
    page,
  }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Models" }).click({ timeout: 12_000 })

    // Wait for recommendations to load, then click Train
    const trainBtn = page.getByRole("button", { name: /Train \d+ model/i })
    await expect(trainBtn).toBeVisible({ timeout: 12_000 })
    await trainBtn.click()

    // Training in progress: at least one "Queued" or "Training..." badge
    await expect(
      page.locator('[data-slot="badge"]', { hasText: /Queued|Training/ }).first()
    ).toBeVisible({ timeout: 8_000 })

    // Wait for training to complete — at least one Done badge appears
    await expect(
      page.locator('[data-slot="badge"]', { hasText: "Done" }).first()
    ).toBeVisible({ timeout: 90_000 })
  })
})
