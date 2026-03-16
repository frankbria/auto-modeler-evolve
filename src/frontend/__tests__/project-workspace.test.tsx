/**
 * Tests for app/project/[id]/page.tsx (ProjectWorkspace).
 *
 * Strategy: mock all child panel components to simple stubs so this file
 * focuses on the workspace orchestration logic — loading, state transitions,
 * tab switching, chat send, SSE streaming, callbacks, mobile view, etc.
 */

import React from "react"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"

// Enable fetch mocking BEFORE any module imports that touch fetch
fetchMock.enableMocks()

// ---------------------------------------------------------------------------
// Mocks — nav, dropzone, child panels
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
    onModelDownload,
    onModelReport,
  }: {
    projectId: string
    onModelSelected?: (runId: string, algo: string) => void
    onModelDownload?: (runId: string) => void
    onModelReport?: (runId: string) => void
  }) => (
    <div data-testid="model-training-panel" data-project={projectId}>
      <button onClick={() => onModelSelected?.("run-1", "random_forest")}>Select Model</button>
      <button onClick={() => onModelDownload?.("run-1")}>Download Model</button>
      <button onClick={() => onModelReport?.("run-1")}>View Report</button>
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
      <button onClick={() => onApplied({ new_columns: ["col_a", "col_b"], total_columns: 7 })}>
        Apply Features
      </button>
    </div>
  ),
  FeatureImportancePanel: ({ targetColumn }: { targetColumn: string }) => (
    <div data-testid="feature-importance-panel" data-target={targetColumn} />
  ),
  DatasetListPanel: ({
    onMerged,
  }: {
    onMerged: (result: {
      join_key: string
      how: string
      row_count: number
      column_count: number
      filename: string
      conflict_columns: string[]
    }) => void
  }) => (
    <div data-testid="dataset-list-panel">
      <button
        onClick={() =>
          onMerged({
            join_key: "id",
            how: "inner",
            row_count: 100,
            column_count: 5,
            filename: "merged.csv",
            conflict_columns: ["score"],
          })
        }
      >
        Trigger Merge
      </button>
    </div>
  ),
}))

// Import the component AFTER all jest.mock() calls
import ProjectWorkspace from "../app/project/[id]/page"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Shared test fixtures
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

const mockProjectWithDataset = { ...mockProject, dataset_id: "ds-123" }

const mockChatHistoryEmpty = { messages: [] }

const mockChatHistoryWithMessages = {
  messages: [
    { role: "assistant", content: "Welcome! Upload data.", timestamp: "2024-01-01T10:00:00" },
    { role: "user", content: "What is this?", timestamp: "2024-01-01T10:01:00" },
    { role: "assistant", content: "This is your dataset.", timestamp: "2024-01-01T10:02:00" },
  ],
}

const mockPreview = {
  dataset_id: "ds-123",
  filename: "sales.csv",
  row_count: 200,
  column_count: 5,
  preview: [{ date: "2024-01", revenue: 1000, region: "North" }],
  column_stats: [
    {
      name: "revenue",
      dtype: "float64",
      null_count: 0,
      null_pct: 0,
      unique_count: 200,
      mean: 1000,
      std: 100,
      min: 500,
      max: 2000,
      outliers: { count: 2, pct: 1.0 },
    },
    {
      name: "region",
      dtype: "object",
      null_count: 0,
      null_pct: 0,
      unique_count: 4,
      mean: null,
      std: null,
      min: null,
      max: null,
      outliers: null,
    },
  ],
  insights: [
    { title: "High Variance", detail: "Revenue varies significantly.", severity: "warning" as const },
    { title: "Good Coverage", detail: "No missing values detected.", severity: "info" as const },
  ],
}

const mockPreviewNoInsights = { ...mockPreview, insights: [] }

const mockRuns = {
  runs: [
    {
      id: "run-1",
      algorithm: "random_forest",
      status: "done",
      metrics: { r2: 0.9, mae: 50 },
      is_selected: true,
      created_at: "2024-01-02T00:00:00",
    },
  ],
}

const mockSampleUpload = {
  dataset_id: "sample-ds",
  filename: "sample_sales.csv",
  row_count: 200,
  column_count: 5,
  preview: [],
  column_stats: [],
  insights: [],
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
// Tests
// ---------------------------------------------------------------------------

describe("ProjectWorkspace — loading and initial state", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("renders project name in breadcrumb after load", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText("Test Project")).toBeInTheDocument()
    })
  })

  it("renders upload panel when project has no dataset", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText(/drop your csv or excel file/i)).toBeInTheDocument()
    })
  })

  it("shows WELCOME_MESSAGE in chat on first visit (no history)", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText(/upload a csv or excel file to get started/i)).toBeInTheDocument()
    })
  })

  it("shows welcome-back message when returning with conversation history", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryWithMessages))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText(/welcome back to \*\*test project\*\*/i)).toBeInTheDocument()
    })
  })

  it("does NOT add welcome-back when history has only assistant messages", async () => {
    const assistantOnly = {
      messages: [
        { role: "assistant", content: "Welcome!", timestamp: "2024-01-01T10:00:00" },
      ],
    }
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(assistantOnly))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      // Should show the stored assistant message, not a welcome-back message
      expect(screen.queryByText(/welcome back to/i)).not.toBeInTheDocument()
    })
  })

  it("shows WELCOME_MESSAGE when project fetch fails", async () => {
    fetchMock.mockRejectOnce(new Error("Network error"))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText(/upload a csv or excel file to get started/i)).toBeInTheDocument()
    })
  })
})

describe("ProjectWorkspace — project with existing dataset", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("loads dataset preview when project has dataset_id", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockPreview))
    fetchMock.mockResponseOnce(JSON.stringify(mockRuns))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText("sales.csv")).toBeInTheDocument()
    })
  })

  it("renders all 6 tab labels when dataset is loaded", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockPreview))
    fetchMock.mockResponseOnce(JSON.stringify({ runs: [] }))
    render(<ProjectWorkspace />)
    // "Data" appears twice (mobile toggle + tab), use getAllByText
    await waitFor(() => expect(screen.getAllByText("Data").length).toBeGreaterThanOrEqual(1))
    for (const tab of ["Features", "Importance", "Models"]) {
      expect(screen.getByText(tab)).toBeInTheDocument()
    }
    expect(screen.getByTestId("tab-validate")).toBeInTheDocument()
    expect(screen.getByTestId("tab-deploy")).toBeInTheDocument()
  })

  it("shows insights in Data tab", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockPreview))
    fetchMock.mockResponseOnce(JSON.stringify({ runs: [] }))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText("High Variance")).toBeInTheDocument()
      expect(screen.getByText("Good Coverage")).toBeInTheDocument()
    })
  })

  it("restores selected model run on load so Validate tab sees run-1", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockPreview))
    fetchMock.mockResponseOnce(JSON.stringify(mockRuns))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByTestId("tab-validate")).toBeInTheDocument())

    fireEvent.click(screen.getByTestId("tab-validate"))
    await waitFor(() => {
      expect(screen.getByTestId("validation-panel").getAttribute("data-run")).toBe("run-1")
    })
  })

  it("shows upload panel when dataset preview fails (file missing)", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockRejectOnce(new Error("File not found")) // preview fails
    fetchMock.mockResponseOnce(JSON.stringify({ runs: [] }))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText(/drop your csv or excel file/i)).toBeInTheDocument()
    })
  })
})

describe("ProjectWorkspace — tab switching", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    resetStore()
    // Standard setup with dataset
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockPreviewNoInsights))
    fetchMock.mockResponseOnce(JSON.stringify({ runs: [] }))
  })

  it("switches to Models tab showing ModelTrainingPanel", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Models")).toBeInTheDocument())
    fireEvent.click(screen.getByText("Models"))
    await waitFor(() => {
      expect(screen.getByTestId("model-training-panel")).toBeInTheDocument()
    })
  })

  it("switches to Validate tab showing ValidationPanel", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByTestId("tab-validate")).toBeInTheDocument())
    fireEvent.click(screen.getByTestId("tab-validate"))
    await waitFor(() => {
      expect(screen.getByTestId("validation-panel")).toBeInTheDocument()
    })
  })

  it("switches to Deploy tab showing DeploymentPanel", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByTestId("tab-deploy")).toBeInTheDocument())
    fireEvent.click(screen.getByTestId("tab-deploy"))
    await waitFor(() => {
      expect(screen.getByTestId("deployment-panel")).toBeInTheDocument()
    })
  })

  it("switches to Features tab and fetches suggestions", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ suggestions: [] }))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Features")).toBeInTheDocument())
    fireEvent.click(screen.getByText("Features"))
    await waitFor(() => {
      expect(screen.getByTestId("feature-suggestions-panel")).toBeInTheDocument()
    })
  })

  it("switches to Importance tab showing target column input", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Importance")).toBeInTheDocument())
    fireEvent.click(screen.getByText("Importance"))
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/target column/i)).toBeInTheDocument()
    })
  })
})

describe("ProjectWorkspace — chat functionality", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("shows chat input with Send button", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/ask about your data/i)).toBeInTheDocument()
    })
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument()
  })

  it("Send button is disabled when input is empty", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByRole("button", { name: /send/i })).toBeDisabled())
  })

  it("Send button enables when text is typed", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByPlaceholderText(/ask about your data/i)).toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText(/ask about your data/i), {
      target: { value: "Hello" },
    })
    expect(screen.getByRole("button", { name: /send/i })).not.toBeDisabled()
  })

  it("sends message and clears input", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    const sseBody = 'data: {"type":"token","content":"Hi there"}\n\ndata: {"type":"done"}\n\n'
    fetchMock.mockResponseOnce(sseBody)

    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByPlaceholderText(/ask about your data/i)).toBeInTheDocument())

    const input = screen.getByPlaceholderText(/ask about your data/i)
    fireEvent.change(input, { target: { value: "What is my data?" } })
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /send/i }))
    })

    expect(input).toHaveValue("")
    await waitFor(() => {
      expect(screen.getByText("What is my data?")).toBeInTheDocument()
    })
  })

  it("sends message on Enter key press", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce('data: {"type":"done"}\n\n')

    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByPlaceholderText(/ask about your data/i)).toBeInTheDocument())

    const input = screen.getByPlaceholderText(/ask about your data/i)
    fireEvent.change(input, { target: { value: "Test query" } })
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter", code: "Enter" })
    })
    expect(input).toHaveValue("")
  })

  it("does not send on Shift+Enter", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByPlaceholderText(/ask about your data/i)).toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText(/ask about your data/i), {
      target: { value: "Test" },
    })
    fireEvent.keyDown(screen.getByPlaceholderText(/ask about your data/i), {
      key: "Enter",
      shiftKey: true,
    })
    // Only 2 calls (project + history), no 3rd for chat
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("handles connection error during streaming — shows error in chat", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockRejectOnce(new Error("Connection refused"))

    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByPlaceholderText(/ask about your data/i)).toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText(/ask about your data/i), {
      target: { value: "Hello" },
    })
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /send/i }))
    })

    await waitFor(() => {
      expect(screen.getByText(/connection error/i)).toBeInTheDocument()
    })
  })
})

describe("ProjectWorkspace — navigation and UI controls", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("navigates to home when '← Projects' is clicked", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/← Projects/)).toBeInTheDocument())

    fireEvent.click(screen.getByText(/← Projects/))
    expect(mockPush).toHaveBeenCalledWith("/")
  })

  it("toggles right panel with Hide/Show button", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Hide panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Hide panel"))
    await waitFor(() => expect(screen.getByText("Show panel")).toBeInTheDocument())
  })

  it("has mobile Chat/Data toggle buttons", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Chat")).toBeInTheDocument())
    // "Data" appears both as mobile toggle and as a tab label when dataset is loaded
    // Here no dataset, so only mobile toggle "Data" should appear
    const dataButtons = screen.queryAllByText("Data")
    expect(dataButtons.length).toBeGreaterThan(0)
  })
})

describe("ProjectWorkspace — upload actions (UploadPanel)", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    resetStore()
  })

  it("loads sample data when sample button clicked", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUpload))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/load sample sales data/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText(/load sample sales data/i))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/data/sample"),
        expect.any(Object)
      )
    })
  })

  it("adds assistant message after sample load", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUpload))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/load sample sales data/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText(/load sample sales data/i))
    })

    await waitFor(() => {
      expect(screen.getByText(/loaded the sample sales dataset/i)).toBeInTheDocument()
    })
  })

  it("shows error when sample load fails", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockRejectOnce(new Error("Server error"))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/load sample sales data/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.click(screen.getByText(/load sample sales data/i))
    })

    await waitFor(() => {
      expect(screen.getByText(/problem loading the sample data/i)).toBeInTheDocument()
    })
  })

  it("URL import toggle reveals input", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/import from google sheets/i)).toBeInTheDocument())

    fireEvent.click(screen.getByText(/import from google sheets/i))
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/docs.google.com/i)).toBeInTheDocument()
    })
  })

  it("URL import Cancel hides input", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/import from google sheets/i)).toBeInTheDocument())

    fireEvent.click(screen.getByText(/import from google sheets/i))
    await waitFor(() => expect(screen.getByText("Cancel")).toBeInTheDocument())
    fireEvent.click(screen.getByText("Cancel"))
    await waitFor(() => {
      expect(screen.queryByPlaceholderText(/docs.google.com/i)).not.toBeInTheDocument()
    })
  })

  it("submits URL import via Import button", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUpload))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/import from google sheets/i)).toBeInTheDocument())

    fireEvent.click(screen.getByText(/import from google sheets/i))
    await waitFor(() => expect(screen.getByPlaceholderText(/docs.google.com/i)).toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText(/docs.google.com/i), {
      target: { value: "https://example.com/data.csv" },
    })
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^import$/i }))
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/data/upload-url"),
        expect.any(Object)
      )
    })
  })

  it("submits URL import via Enter key", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockSampleUpload))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/import from google sheets/i)).toBeInTheDocument())

    fireEvent.click(screen.getByText(/import from google sheets/i))
    await waitFor(() => expect(screen.getByPlaceholderText(/docs.google.com/i)).toBeInTheDocument())

    const urlInput = screen.getByPlaceholderText(/docs.google.com/i)
    fireEvent.change(urlInput, { target: { value: "https://example.com/data.csv" } })
    await act(async () => {
      fireEvent.keyDown(urlInput, { key: "Enter", code: "Enter" })
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/data/upload-url"),
        expect.any(Object)
      )
    })
  })

  it("shows error when URL import fails", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProject))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockRejectOnce(new Error("Bad URL"))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText(/import from google sheets/i)).toBeInTheDocument())

    fireEvent.click(screen.getByText(/import from google sheets/i))
    await waitFor(() => expect(screen.getByPlaceholderText(/docs.google.com/i)).toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText(/docs.google.com/i), {
      target: { value: "https://bad.com/data.csv" },
    })
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^import$/i }))
    })

    await waitFor(() => {
      expect(screen.getByText(/problem importing from that url/i)).toBeInTheDocument()
    })
  })
})

describe("ProjectWorkspace — child panel callbacks", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    resetStore()
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockPreviewNoInsights))
    fetchMock.mockResponseOnce(JSON.stringify({ runs: [] }))
  })

  it("onModelSelected updates selected run and adds chat message", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Models")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Models"))
    await waitFor(() => expect(screen.getByTestId("model-training-panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Select Model"))
    await waitFor(() => {
      expect(screen.getByText(/selected this model/i)).toBeInTheDocument()
    })
    // Validate tab should show the run ID
    fireEvent.click(screen.getByTestId("tab-validate"))
    await waitFor(() => {
      expect(screen.getByTestId("validation-panel").getAttribute("data-run")).toBe("run-1")
    })
  })

  it("onDeployed adds deployment live message to chat", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByTestId("tab-deploy")).toBeInTheDocument())

    fireEvent.click(screen.getByTestId("tab-deploy"))
    await waitFor(() => expect(screen.getByTestId("deployment-panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Trigger Deploy"))
    await waitFor(() => {
      expect(screen.getByText(/your model is live/i)).toBeInTheDocument()
    })
  })

  it("onMerged adds merge summary to chat", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByTestId("dataset-list-panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Trigger Merge"))
    await waitFor(() => {
      expect(screen.getByText(/merged the two datasets on/i)).toBeInTheDocument()
    })
  })

  it("merge with conflict columns mentions renamed columns in message", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByTestId("dataset-list-panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Trigger Merge"))
    await waitFor(() => {
      // "score" was in conflict_columns
      expect(screen.getByText(/score/i)).toBeInTheDocument()
    })
  })

  it("feature apply callback adds message with column count", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ suggestions: [] }))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Features")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Features"))
    await waitFor(() => expect(screen.getByTestId("feature-suggestions-panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Apply Features"))
    await waitFor(() => {
      expect(screen.getByText(/applied your feature transformations/i)).toBeInTheDocument()
    })
  })

  it("Importance Analyze button shows panel after API responds", async () => {
    fetchMock.mockResponseOnce(
      JSON.stringify({ features: [{ name: "revenue", importance: 0.9 }], problem_type: "regression" })
    )
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Importance")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Importance"))
    await waitFor(() => expect(screen.getByPlaceholderText(/target column/i)).toBeInTheDocument())

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(/target column/i), {
        target: { value: "revenue" },
      })
    })
    await act(async () => {
      // GET requests don't include options object — just check by URL
      fireEvent.click(screen.getByRole("button", { name: /analyze/i }))
    })

    await waitFor(() => {
      // After successful API call, FeatureImportancePanel renders with result
      expect(screen.getByTestId("feature-importance-panel")).toBeInTheDocument()
    })
  })

  it("Importance Enter key also triggers importance load", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ features: [{ name: "revenue", importance: 0.5 }], problem_type: "regression" }))
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Importance")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Importance"))
    await waitFor(() => expect(screen.getByPlaceholderText(/target column/i)).toBeInTheDocument())

    const input = screen.getByPlaceholderText(/target column/i)
    await act(async () => {
      fireEvent.change(input, { target: { value: "revenue" } })
    })
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter", code: "Enter" })
    })

    await waitFor(() => {
      expect(screen.getByTestId("feature-importance-panel")).toBeInTheDocument()
    })
  })

  it("onModelDownload calls window.open", async () => {
    const openSpy = jest.spyOn(window, "open").mockImplementation(() => null)
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Models")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Models"))
    await waitFor(() => expect(screen.getByTestId("model-training-panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Download Model"))
    expect(openSpy).toHaveBeenCalled()
    openSpy.mockRestore()
  })

  it("onModelReport calls window.open", async () => {
    const openSpy = jest.spyOn(window, "open").mockImplementation(() => null)
    render(<ProjectWorkspace />)
    await waitFor(() => expect(screen.getByText("Models")).toBeInTheDocument())

    fireEvent.click(screen.getByText("Models"))
    await waitFor(() => expect(screen.getByTestId("model-training-panel")).toBeInTheDocument())

    fireEvent.click(screen.getByText("View Report"))
    expect(openSpy).toHaveBeenCalled()
    openSpy.mockRestore()
  })
})

describe("DataPreviewPanel — inline component rendering", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    resetStore()
    fetchMock.mockResponseOnce(JSON.stringify(mockProjectWithDataset))
    fetchMock.mockResponseOnce(JSON.stringify(mockChatHistoryEmpty))
    fetchMock.mockResponseOnce(JSON.stringify(mockPreview))
    fetchMock.mockResponseOnce(JSON.stringify({ runs: [] }))
  })

  it("shows row and column count badges", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText("200 rows")).toBeInTheDocument()
      expect(screen.getByText("5 columns")).toBeInTheDocument()
    })
  })

  it("shows filename in panel header", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText("sales.csv")).toBeInTheDocument()
    })
  })

  it("shows dtype badges for columns", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText("float64")).toBeInTheDocument()
      expect(screen.getByText("object")).toBeInTheDocument()
    })
  })

  it("shows outlier warning when outliers present", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText(/2 outlier/i)).toBeInTheDocument()
    })
  })

  it("renders data table column headers from preview", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => {
      // Preview has columns: date, revenue, region
      // "region" and "revenue" also appear in column stats cards, so use getAllByText
      expect(screen.getByText("date")).toBeInTheDocument()
      expect(screen.getAllByText("region").length).toBeGreaterThan(0)
    })
  })

  it("renders data preview row count label", async () => {
    render(<ProjectWorkspace />)
    await waitFor(() => {
      expect(screen.getByText(/data preview/i)).toBeInTheDocument()
    })
  })
})
