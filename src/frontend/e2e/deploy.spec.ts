/**
 * E2E tests for Phase 6: Deployment & Prediction
 *
 * Setup: create project → upload sample data → apply features → set target →
 *   train model → select model — all via API.
 * Tests: UI-level deployment flow (Deploy tab → Deploy Model button), the
 *   deployed state display, and the public /predict/[id] dashboard.
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

async function waitForRunsDone(
  request: import("@playwright/test").APIRequestContext,
  projectId: string,
  maxMs = 60_000
): Promise<string[]> {
  const deadline = Date.now() + maxMs
  while (Date.now() < deadline) {
    const res = await request.get(`${BACKEND}/api/models/${projectId}/runs`)
    if (!res.ok()) throw new Error("Failed to fetch runs")
    const body = await res.json()
    const runs: Array<{ status: string; id: string }> = body.runs ?? []
    if (runs.length > 0 && runs.every((r) => r.status === "done" || r.status === "failed")) {
      return runs.filter((r) => r.status === "done").map((r) => r.id)
    }
    await new Promise((r) => setTimeout(r, 1_500))
  }
  throw new Error("Training did not complete within timeout")
}

/**
 * Full setup: project + sample data + feature set + train + select model.
 * Returns { projectId, selectedRunId }
 */
async function setupDeployReady(
  request: import("@playwright/test").APIRequestContext
): Promise<{ projectId: string; selectedRunId: string }> {
  // 1. Create project
  const projRes = await request.post(`${BACKEND}/api/projects`, {
    data: { name: "Deploy E2E" },
  })
  const project = await projRes.json()
  const projectId: string = project.id

  // 2. Load sample dataset
  const dataRes = await request.post(`${BACKEND}/api/data/sample`, {
    data: { project_id: projectId },
  })
  const dataBody = await dataRes.json()
  const datasetId: string = dataBody.dataset_id

  // 3. Apply empty feature set
  const applyRes = await request.post(
    `${BACKEND}/api/features/${datasetId}/apply`,
    { data: { transformations: [] } }
  )
  const applyBody = await applyRes.json()
  const featureSetId: string = applyBody.feature_set_id

  // 4. Set target column
  await request.post(`${BACKEND}/api/features/${datasetId}/target`, {
    data: { target_column: "revenue", feature_set_id: featureSetId },
  })

  // 5. Train a single fast model
  await request.post(`${BACKEND}/api/models/${projectId}/train`, {
    data: { algorithms: ["linear_regression"] },
  })

  // 6. Wait for training to complete
  const doneIds = await waitForRunsDone(request, projectId)
  if (doneIds.length === 0) throw new Error("No successful model runs after training")
  const runId = doneIds[0]

  // 7. Select the model
  await request.post(`${BACKEND}/api/models/${runId}/select`)

  return { projectId, selectedRunId: runId }
}

// ---------------------------------------------------------------------------
// Tests: Deploy tab UI
// ---------------------------------------------------------------------------

test.describe("Deployment — Deploy tab UI", () => {
  let projectId: string
  let selectedRunId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    ;({ projectId, selectedRunId } = await setupDeployReady(request))
  })

  test("Deploy tab shows Deploy Model button when model is selected", async ({
    page,
  }) => {
    await page.goto(`/project/${projectId}`)
    // Dataset state restores from DB automatically
    await page.getByRole("button", { name: "Deploy" }).click({ timeout: 12_000 })
    await expect(
      page.getByRole("button", { name: /Deploy Model/i })
    ).toBeVisible({ timeout: 8_000 })
  })

  test("clicking Deploy Model deploys and shows green status indicator", async ({
    page,
  }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Deploy" }).click({ timeout: 12_000 })
    const deployBtn = page.getByRole("button", { name: /Deploy Model/i })
    await expect(deployBtn).toBeVisible({ timeout: 8_000 })
    await deployBtn.click()

    // After deploy, the panel shows "Model deployed" status
    await expect(page.getByText(/Model deployed/i)).toBeVisible({ timeout: 15_000 })
  })

  test("deployed state shows dashboard URL and API endpoint", async ({ page }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Deploy" }).click({ timeout: 12_000 })
    await page.getByRole("button", { name: /Deploy Model/i }).click({ timeout: 8_000 })
    await expect(page.getByText(/Model deployed/i)).toBeVisible({ timeout: 15_000 })

    // Dashboard URL section heading
    await expect(page.getByText("Dashboard URL")).toBeVisible()
    // API Endpoint heading — use exact match (chat says "API endpoint" lowercase)
    await expect(page.getByText("API Endpoint", { exact: true }).first()).toBeVisible()
  })

  test("deploying a model posts a chat message with the share link", async ({
    page,
  }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Deploy" }).click({ timeout: 12_000 })
    await page.getByRole("button", { name: /Deploy Model/i }).click({ timeout: 8_000 })
    await expect(page.getByText(/Model deployed/i)).toBeVisible({ timeout: 15_000 })

    // Chat should receive a message mentioning the share link
    await expect(
      page.getByText(/your model is live/i, { exact: false })
    ).toBeVisible({ timeout: 10_000 })
  })

  test("Undeploy button appears after deployment and removes it", async ({
    page,
  }) => {
    await page.goto(`/project/${projectId}`)
    await page.getByRole("button", { name: "Deploy" }).click({ timeout: 12_000 })
    await page.getByRole("button", { name: /Deploy Model/i }).click({ timeout: 8_000 })
    await expect(page.getByText(/Model deployed/i)).toBeVisible({ timeout: 15_000 })

    const undeployBtn = page.getByRole("button", { name: /Undeploy/i })
    await expect(undeployBtn).toBeVisible()
    await undeployBtn.click()

    // After undeploy, the Deploy Model button returns
    await expect(
      page.getByRole("button", { name: /Deploy Model/i })
    ).toBeVisible({ timeout: 10_000 })
  })
})

// ---------------------------------------------------------------------------
// Tests: Prediction dashboard (/predict/[id])
// ---------------------------------------------------------------------------

test.describe("Prediction dashboard — /predict/[id]", () => {
  let deploymentId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    const { selectedRunId } = await setupDeployReady(request)

    // Deploy via API
    const depRes = await request.post(`${BACKEND}/api/deploy/${selectedRunId}`)
    expect(depRes.ok()).toBeTruthy()
    const dep = await depRes.json()
    deploymentId = dep.id
  })

  test("prediction dashboard loads with input form", async ({ page }) => {
    await page.goto(`/predict/${deploymentId}`)
    // Form should have input fields for the feature columns
    await expect(page.getByRole("button", { name: /predict|get prediction/i })).toBeVisible({
      timeout: 10_000,
    })
  })

  test("prediction dashboard shows the model name", async ({ page }) => {
    await page.goto(`/predict/${deploymentId}`)
    // The page shows the algorithm or deployment heading
    await expect(
      page.getByText(/linear regression|prediction/i, { exact: false })
    ).toBeVisible({ timeout: 10_000 })
  })

  test("submitting the form returns a prediction result", async ({ page }) => {
    await page.goto(`/predict/${deploymentId}`)
    // Wait for the form to render with pre-filled defaults
    const predictBtn = page.getByRole("button", { name: /predict|get prediction/i })
    await expect(predictBtn).toBeVisible({ timeout: 10_000 })
    // Submit with defaults
    await predictBtn.click()
    // After prediction, the result card shows "Predicted revenue"
    // This text only appears inside the result card, not in the form
    await expect(
      page.getByText(/Predicted revenue/i)
    ).toBeVisible({ timeout: 10_000 })
  })
})

// ---------------------------------------------------------------------------
// Tests: Batch prediction API (via API request, verifying endpoint exists)
// ---------------------------------------------------------------------------

test.describe("Batch prediction — API endpoint", () => {
  let deploymentId: string

  test.beforeEach(async ({ request }) => {
    await cleanProjects(request)
    const { selectedRunId } = await setupDeployReady(request)
    const depRes = await request.post(`${BACKEND}/api/deploy/${selectedRunId}`)
    const dep = await depRes.json()
    deploymentId = dep.id
  })

  test("batch prediction endpoint accepts a CSV and returns predictions", async ({
    request,
  }) => {
    // Create a minimal CSV matching the sample's feature columns
    const csvContent = [
      "date,product,region,units_sold",
      "2024-01-01,Electronics,North,50",
      "2024-02-01,Clothing,South,30",
    ].join("\n")

    const res = await request.post(
      `${BACKEND}/api/predict/${deploymentId}/batch`,
      {
        multipart: {
          file: {
            name: "batch.csv",
            mimeType: "text/csv",
            buffer: Buffer.from(csvContent),
          },
        },
      }
    )
    // Should return 200 with a CSV file
    expect(res.ok()).toBeTruthy()
    const contentType = res.headers()["content-type"] ?? ""
    expect(contentType).toContain("text/csv")
  })
})
