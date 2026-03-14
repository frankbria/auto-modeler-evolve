/**
 * Unit tests for the API client (lib/api.ts).
 *
 * Uses jest-fetch-mock to intercept fetch calls and verify the correct
 * URLs, methods, headers, and body shapes are sent to the backend.
 * No real HTTP requests are made.
 */

import fetchMock from "jest-fetch-mock"

fetchMock.enableMocks()

// Importing AFTER enabling mocks so fetch is mocked in the module scope
import { api } from "../lib/api"

beforeEach(() => {
  fetchMock.resetMocks()
})

const BASE = "http://localhost:8000"

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

describe("api.projects", () => {
  it("list() calls GET /api/projects", async () => {
    fetchMock.mockResponseOnce(JSON.stringify([{ id: "p1", name: "Test" }]))
    const result = await api.projects.list()
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/projects`)
    expect(result[0].id).toBe("p1")
  })

  it("create() sends POST with name and description", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "p2", name: "New" }))
    await api.projects.create("New", "desc")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "New", description: "desc" }),
    })
  })

  it("create() works without description", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "p3", name: "No Desc" }))
    await api.projects.create("No Desc")
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.description).toBeUndefined()
  })

  it("get() calls GET /api/projects/:id", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "p1" }))
    await api.projects.get("p1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/projects/p1`)
  })

  it("update() sends PATCH with partial fields", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "p1", name: "Renamed" }))
    await api.projects.update("p1", { name: "Renamed" })
    expect(fetchMock.mock.calls[0][1]?.method).toBe("PATCH")
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.name).toBe("Renamed")
  })

  it("duplicate() sends POST to /api/projects/:id/duplicate", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "p2", name: "Copy of p1" }))
    await api.projects.duplicate("p1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/projects/p1/duplicate`, {
      method: "POST",
    })
  })

  it("delete() sends DELETE to /api/projects/:id", async () => {
    fetchMock.mockResponseOnce("", { status: 204 })
    await api.projects.delete("p1")
    expect(fetchMock.mock.calls[0][1]?.method).toBe("DELETE")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/projects/p1`, { method: "DELETE" })
  })
})

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

describe("api.data", () => {
  it("upload() sends multipart form with project_id and file", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ dataset_id: "ds1", row_count: 100 }))
    const file = new File(["col1,col2\n1,2"], "data.csv", { type: "text/csv" })
    await api.data.upload("proj-1", file)
    const call = fetchMock.mock.calls[0]
    expect(call[0]).toBe(`${BASE}/api/data/upload`)
    expect(call[1]?.method).toBe("POST")
    // FormData should contain project_id and file
    const formData = call[1]?.body as FormData
    expect(formData.get("project_id")).toBe("proj-1")
    expect(formData.get("file")).toBeInstanceOf(File)
  })

  it("loadSample() sends POST to /api/data/sample", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ dataset_id: "ds2", row_count: 200 }))
    await api.data.loadSample("proj-1")
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST")
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.project_id).toBe("proj-1")
  })

  it("preview() calls GET /api/data/:id/preview", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ dataset_id: "ds1" }))
    await api.data.preview("ds1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/data/ds1/preview`)
  })

  it("query() sends POST with question", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ answer: "North region" }))
    await api.data.query("ds1", "Which region has highest sales?")
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.question).toBe("Which region has highest sales?")
  })

  it("timeseries() omits query params when not provided", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ chart_spec: null }))
    await api.data.timeseries("ds1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/data/ds1/timeseries`)
  })

  it("timeseries() appends query params when provided", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ chart_spec: null }))
    await api.data.timeseries("ds1", "revenue", 14)
    const url = fetchMock.mock.calls[0][0] as string
    expect(url).toContain("value_column=revenue")
    expect(url).toContain("window=14")
  })

  it("correlations() calls GET /api/data/:id/correlations", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ chart_spec: null }))
    await api.data.correlations("ds1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/data/ds1/correlations`)
  })
})

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

describe("api.chat", () => {
  it("send() sends POST with message body", async () => {
    fetchMock.mockResponseOnce("data: Hello\n\n")
    await api.chat.send("proj-1", "What is my data?")
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST")
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.message).toBe("What is my data?")
  })

  it("history() calls GET /api/chat/:id/history", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ messages: [] }))
    const result = await api.chat.history("proj-1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/chat/proj-1/history`)
    expect(result.messages).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

describe("api.features", () => {
  it("suggestions() calls GET /api/features/:id/suggestions", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ suggestions: [] }))
    await api.features.suggestions("ds1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/features/ds1/suggestions`)
  })

  it("apply() sends POST with transformations list", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ feature_set_id: "fs1" }))
    const transforms = [{ column: "date", transform_type: "date_decompose" }]
    await api.features.apply("ds1", transforms)
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.transformations).toHaveLength(1)
    expect(body.transformations[0].column).toBe("date")
  })

  it("setTarget() sends POST with target_column", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ target_column: "revenue", problem_type: "regression" }))
    await api.features.setTarget("ds1", "revenue")
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.target_column).toBe("revenue")
  })

  it("importance() encodes target column in URL", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ features: [] }))
    await api.features.importance("ds1", "sales revenue")
    const url = fetchMock.mock.calls[0][0] as string
    expect(url).toContain("target_column=sales%20revenue")
  })
})

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

describe("api.models", () => {
  it("train() sends POST with algorithms list", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ status: "started" }))
    await api.models.train("proj-1", ["random_forest", "linear"])
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.algorithms).toEqual(["random_forest", "linear"])
  })

  it("select() sends POST to /api/models/:id/select", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "run-1", is_selected: true }))
    await api.models.select("run-1")
    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/api/models/run-1/select`)
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST")
  })

  it("comparisonRadar() returns null on 204 response", async () => {
    fetchMock.mockResponseOnce("", { status: 204 })
    const result = await api.models.comparisonRadar("proj-1")
    expect(result).toBeNull()
  })

  it("downloadUrl() returns correct URL string", () => {
    expect(api.models.downloadUrl("run-1")).toBe(`${BASE}/api/models/run-1/download`)
  })

  it("reportUrl() returns correct URL string", () => {
    expect(api.models.reportUrl("run-1")).toBe(`${BASE}/api/models/run-1/report`)
  })

  it("trainingStreamUrl() returns SSE URL", () => {
    expect(api.models.trainingStreamUrl("proj-1")).toBe(
      `${BASE}/api/models/proj-1/training-stream`
    )
  })
})

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

describe("api.validation", () => {
  it("metrics() calls GET /api/validate/:id/metrics", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ cross_validation: {} }))
    await api.validation.metrics("run-1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/validate/run-1/metrics`)
  })

  it("explain() calls GET /api/validate/:id/explain", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ feature_importance: [] }))
    await api.validation.explain("run-1")
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/validate/run-1/explain`)
  })

  it("explainRow() calls GET /api/validate/:id/explain/:row", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ contributions: [] }))
    await api.validation.explainRow("run-1", 42)
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/validate/run-1/explain/42`)
  })
})

// ---------------------------------------------------------------------------
// Deploy
// ---------------------------------------------------------------------------

describe("api.deploy", () => {
  it("deploy() sends POST to /api/deploy/:id", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ id: "dep-1", is_active: true }))
    await api.deploy.deploy("run-1")
    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/api/deploy/run-1`)
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST")
  })

  it("list() calls GET /api/deployments", async () => {
    fetchMock.mockResponseOnce(JSON.stringify([]))
    await api.deploy.list()
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/api/deployments`)
  })

  it("undeploy() sends DELETE to /api/deploy/:id", async () => {
    fetchMock.mockResponseOnce("", { status: 204 })
    await api.deploy.undeploy("dep-1")
    expect(fetchMock.mock.calls[0][1]?.method).toBe("DELETE")
  })

  it("predict() sends POST with input data", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ prediction: 42.5 }))
    await api.deploy.predict("dep-1", { region: "North", product: "A", units: 10 })
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string)
    expect(body.region).toBe("North")
    expect(body.units).toBe(10)
  })
})
