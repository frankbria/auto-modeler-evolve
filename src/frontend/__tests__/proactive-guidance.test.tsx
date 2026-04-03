/**
 * Tests for proactive guidance features:
 * 1. Upload response suggestions → shown as chatSuggestions chips
 * 2. next_step SSE event → updates chatSuggestions
 * 3. ModelTrainingPanel onTrainingComplete callback fires with chips from all_done event
 */

import React from "react"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"
import * as apiModule from "@/lib/api"

fetchMock.enableMocks()

// ---------------------------------------------------------------------------
// Shared mocks for ProjectWorkspace
// ---------------------------------------------------------------------------

const mockPush = jest.fn()
jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush, replace: jest.fn(), back: jest.fn(), prefetch: jest.fn() }),
  useParams: () => ({ id: "test-project-id" }),
  usePathname: () => "/project/test-project-id",
  useSearchParams: () => new URLSearchParams(),
}))

jest.mock("react-dropzone", () => ({
  useDropzone: () => ({
    getRootProps: () => ({ "data-testid": "dropzone-root" }),
    getInputProps: () => ({ "data-testid": "file-input", type: "file" }),
    isDragActive: false,
  }),
}))

jest.mock("@/components/models/model-training-panel", () => ({
  ModelTrainingPanel: ({
    projectId,
    onModelSelected,
    onTrainingComplete,
  }: {
    projectId: string
    onModelSelected?: (runId: string, algo: string) => void
    onTrainingComplete?: (chips: string[]) => void
  }) => (
    <div data-testid="model-training-panel" data-project={projectId}>
      <button onClick={() => onModelSelected?.("run-1", "random_forest")}>Select Model</button>
      <button
        data-testid="trigger-training-complete"
        onClick={() => onTrainingComplete?.(["Deploy my model", "Validate results", "Share model"])}
      >
        Trigger Training Complete
      </button>
    </div>
  ),
}))

jest.mock("@/components/validation/validation-panel", () => ({
  ValidationPanel: ({ selectedRunId }: { selectedRunId: string | null }) => (
    <div data-testid="validation-panel" data-run={selectedRunId ?? "none"} />
  ),
}))

jest.mock("@/components/deploy/deployment-panel", () => ({
  DeploymentPanel: ({
    onDeployed,
  }: {
    onDeployed: (dep: { dashboard_url: string; endpoint_path: string }) => void
  }) => (
    <div data-testid="deployment-panel">
      <button
        onClick={() =>
          onDeployed({
            dashboard_url: "http://localhost/predict/dep-1",
            endpoint_path: "/api/predict/dep-1",
          })
        }
      >
        Trigger Deploy
      </button>
    </div>
  ),
}))

jest.mock("@/components/features/feature-suggestions", () => ({
  FeatureSuggestionsPanel: ({
    onApplied,
  }: {
    onApplied: (result: { new_columns: string[]; total_columns: number }) => void
  }) => (
    <div data-testid="feature-suggestions-panel">
      <button onClick={() => onApplied({ new_columns: [], total_columns: 5 })}>Apply</button>
    </div>
  ),
  FeatureImportancePanel: ({ targetColumn }: { targetColumn: string }) => (
    <div data-testid="feature-importance-panel" data-target={targetColumn} />
  ),
  DatasetListPanel: ({
    onMerged,
  }: {
    onMerged: (result: {
      join_key: string; how: string; row_count: number; column_count: number;
      filename: string; conflict_columns: string[]
    }) => void
  }) => (
    <div data-testid="dataset-list-panel">
      <button onClick={() => onMerged({ join_key: "id", how: "inner", row_count: 100, column_count: 5, filename: "merged.csv", conflict_columns: [] })}>Merge</button>
    </div>
  ),
}))

import ProjectWorkspace from "../app/project/[id]/page"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const mockProject = {
  id: "test-project-id",
  name: "Test Project",
  description: "A test",
  created_at: "2024-01-01T00:00:00",
  updated_at: "2024-01-02T00:00:00",
  status: "exploring",
  dataset_id: null as string | null,
}

const mockChatHistoryEmpty = { messages: [] }

const mockSampleUploadWithSuggestions = {
  dataset_id: "sample-ds",
  filename: "sample_sales.csv",
  row_count: 200,
  column_count: 5,
  preview: [],
  column_stats: [],
  insights: [],
  suggestions: [
    "Show me the revenue trend over time",
    "Show me revenue by region",
    "Are there any unusual records in the data?",
  ],
}

const mockSampleUploadNoSuggestions = {
  dataset_id: "sample-ds",
  filename: "sample_sales.csv",
  row_count: 200,
  column_count: 5,
  preview: [],
  column_stats: [],
  insights: [],
}

function resetStore() {
  useAppStore.setState({
    projects: [],
    currentProject: null,
    currentDataset: null,
    dataPreview: [],
    columnStats: [],
    dataInsights: [],
    messages: [],
    isStreaming: false,
  })
}

// ---------------------------------------------------------------------------
// Tests: Upload suggestions → suggestion chips appear
// ---------------------------------------------------------------------------

describe("Proactive upload suggestions", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("shows suggestion chips after sample load returns suggestions", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUploadWithSuggestions))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/load sample sales data/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText(/load sample sales data/i))
    })

    await waitFor(() => {
      expect(screen.getByTestId("suggestion-chips")).toBeInTheDocument()
    })
    const chips = screen.getAllByTestId("suggestion-chip")
    expect(chips.length).toBe(3)
    expect(chips[0]).toHaveTextContent("Show me the revenue trend over time")
  })

  it("shows 'Try asking:' label when suggestions are present", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUploadWithSuggestions))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/load sample sales data/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText(/load sample sales data/i))
    })

    await waitFor(() => {
      expect(screen.getByText(/try asking:/i)).toBeInTheDocument()
    })
  })

  it("does NOT show suggestion chips when upload returns no suggestions", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUploadNoSuggestions))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/load sample sales data/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText(/load sample sales data/i))
    })

    await waitFor(() => {
      expect(screen.getByText(/loaded the sample sales dataset/i)).toBeInTheDocument()
    })
    expect(screen.queryByTestId("suggestion-chips")).not.toBeInTheDocument()
  })

  it("clicking a chip pre-fills the chat input", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUploadWithSuggestions))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/load sample sales data/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText(/load sample sales data/i))
    })

    await waitFor(() => {
      expect(screen.getByTestId("suggestion-chips")).toBeInTheDocument()
    })

    const chip = screen.getAllByTestId("suggestion-chip")[0]
    fireEvent.click(chip)

    const input = screen.getByPlaceholderText(/ask about your data/i)
    expect(input).toHaveValue("Show me the revenue trend over time")
  })
})

// ---------------------------------------------------------------------------
// Tests: next_step SSE event → suggestion chips
// ---------------------------------------------------------------------------

describe("next_step SSE event updates suggestion chips", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("shows chips from next_step SSE event after chat response", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))

    const sseText = [
      'data: {"type":"token","content":"Training done!"}\n\n',
      'data: {"type":"next_step","chips":["Deploy my model","Share model with team","Check model accuracy"]}\n\n',
      'data: {"type":"done"}\n\n',
    ].join("")

    // jest-fetch-mock doesn't create a streaming Response body; spy on api.chat.send
    // and return a hand-rolled mock whose getReader() yields the SSE text then done.
    // jest-fetch-mock doesn't create a streaming Response body; spy on api.chat.send
    // and return a hand-rolled mock whose getReader() yields the SSE text then done.
    // jest-fetch-mock doesn't create a streaming Response body; spy on api.chat.send
    // and return a hand-rolled mock whose getReader() yields the SSE text then done.
    const sseBytes = Buffer.from(sseText)
    let yielded = false
    const mockResponse = {
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: jest.fn().mockImplementation(() => {
            if (!yielded) {
              yielded = true
              return Promise.resolve({ done: false, value: sseBytes })
            }
            return Promise.resolve({ done: true, value: undefined })
          }),
        }),
      },
    } as unknown as Response
    const sendSpy = jest
      .spyOn(apiModule.api.chat, "send")
      .mockResolvedValueOnce(mockResponse)

    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByPlaceholderText(/ask about your data/i)).toBeInTheDocument())

    const input = screen.getByPlaceholderText(/ask about your data/i)
    fireEvent.change(input, { target: { value: "train my model" } })
    fireEvent.click(screen.getByRole("button", { name: /send/i }))

    // Wait for suggestion chips to appear (next_step event processed + isStreaming = false)
    await waitFor(
      () => expect(screen.queryByTestId("suggestion-chips")).toBeInTheDocument(),
      { timeout: 4000 }
    )
    const chips = screen.getAllByTestId("suggestion-chip")
    expect(chips.some((c) => c.textContent?.includes("Deploy my model"))).toBe(true)
    sendSpy.mockRestore()
  })
})

// ---------------------------------------------------------------------------
// Tests: ModelTrainingPanel onTrainingComplete callback
// ---------------------------------------------------------------------------

describe("ModelTrainingPanel onTrainingComplete", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("sets chatSuggestions when onTrainingComplete fires with chips", async () => {
    const mockProjectWithDs = { ...mockProject, dataset_id: "ds-123" }
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDs))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(
      JSON.stringify({
        dataset_id: "ds-123",
        filename: "data.csv",
        row_count: 200,
        column_count: 5,
        preview: [],
        column_stats: [],
        insights: [],
        suggestions: [],
      })
    )
    fetchMock.mockResponseOnce(JSON.stringify({ runs: [] }))

    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByRole("tab", { name: "Models" })).toBeInTheDocument())
    fireEvent.click(screen.getByRole("tab", { name: "Models" }))

    await waitFor(() => expect(screen.getByTestId("model-training-panel")).toBeInTheDocument())

    // Trigger the training complete callback from the stubbed panel
    await act(async () => {
      fireEvent.click(screen.getByTestId("trigger-training-complete"))
    })

    await waitFor(() => {
      expect(screen.getByTestId("suggestion-chips")).toBeInTheDocument()
    })
    const chips = screen.getAllByTestId("suggestion-chip")
    expect(chips.some((c) => c.textContent?.includes("Deploy my model"))).toBe(true)
  })
})
