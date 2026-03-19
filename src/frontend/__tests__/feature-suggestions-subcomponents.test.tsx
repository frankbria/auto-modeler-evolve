/**
 * Tests for the sub-components exported from feature-suggestions.tsx:
 *   - PipelinePanel
 *   - DatasetListPanel
 *   - FeatureImportancePanel
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import {
  PipelinePanel,
  DatasetListPanel,
  FeatureImportancePanel,
} from "../components/features/feature-suggestions"
import { api } from "../lib/api"
import type {
  FeatureImportanceEntry,
  DatasetListItem,
  JoinKeySuggestion,
  MergeResponse,
} from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    features: {
      apply: jest.fn(),
      suggestions: jest.fn(),
      setTarget: jest.fn(),
      importance: jest.fn(),
      getSteps: jest.fn(),
      addStep: jest.fn(),
      removeStep: jest.fn(),
    },
    data: {
      listByProject: jest.fn(),
      joinKeys: jest.fn(),
      merge: jest.fn(),
    },
  },
}))

const mockGetSteps = api.features.getSteps as jest.MockedFunction<typeof api.features.getSteps>
const mockRemoveStep = api.features.removeStep as jest.MockedFunction<typeof api.features.removeStep>
const mockListByProject = api.data.listByProject as jest.MockedFunction<typeof api.data.listByProject>
const mockJoinKeys = api.data.joinKeys as jest.MockedFunction<typeof api.data.joinKeys>
const mockMerge = api.data.merge as jest.MockedFunction<typeof api.data.merge>

beforeEach(() => {
  jest.clearAllMocks()
})

// ---------------------------------------------------------------------------
// PipelinePanel
// ---------------------------------------------------------------------------

describe("PipelinePanel — loading state", () => {
  it("shows loading text while fetching steps", async () => {
    // Never resolves so we stay in loading
    mockGetSteps.mockReturnValue(new Promise(() => {}))
    render(<PipelinePanel featureSetId="fs-1" />)
    expect(screen.getByText(/loading pipeline/i)).toBeInTheDocument()
  })
})

describe("PipelinePanel — empty state", () => {
  it("shows 'no transformations' message when steps list is empty", async () => {
    mockGetSteps.mockResolvedValue({ steps: [] })
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() =>
      expect(screen.getByText(/no transformations applied yet/i)).toBeInTheDocument()
    )
  })

  it("still renders gracefully when getSteps rejects", async () => {
    mockGetSteps.mockRejectedValue(new Error("network error"))
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() =>
      expect(screen.getByText(/no transformations applied yet/i)).toBeInTheDocument()
    )
  })
})

describe("PipelinePanel — step list", () => {
  const steps = [
    { index: 0, column: "order_date", transform_type: "date_decompose" },
    { index: 1, column: "price", transform_type: "log_transform" },
  ]

  beforeEach(() => {
    mockGetSteps.mockResolvedValue({ steps })
  })

  it("renders one row per step", async () => {
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() => expect(screen.getByText("order_date")).toBeInTheDocument())
    expect(screen.getByText("price")).toBeInTheDocument()
  })

  it("shows pipeline count text", async () => {
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() =>
      expect(screen.getByText(/2 transformations in pipeline/i)).toBeInTheDocument()
    )
  })

  it("shows singular text for 1 step", async () => {
    mockGetSteps.mockResolvedValue({ steps: [steps[0]] })
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() =>
      expect(screen.getByText(/1 transformation in pipeline/i)).toBeInTheDocument()
    )
  })

  it("shows transform type label for each step", async () => {
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() => expect(screen.getByText("Date Parts")).toBeInTheDocument())
    expect(screen.getByText("Log Transform")).toBeInTheDocument()
  })

  it("shows Undo button for each step", async () => {
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() => {
      const undos = screen.getAllByText("Undo")
      expect(undos).toHaveLength(2)
    })
  })
})

describe("PipelinePanel — undo step", () => {
  const steps = [
    { index: 0, column: "order_date", transform_type: "date_decompose" },
    { index: 1, column: "price", transform_type: "log_transform" },
  ]

  it("calls removeStep with correct featureSetId and index", async () => {
    mockGetSteps.mockResolvedValue({ steps })
    mockRemoveStep.mockResolvedValue({ steps: [steps[0]], new_columns: ["order_date_year"] })
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() => screen.getAllByText("Undo"))
    const undos = screen.getAllByText("Undo")
    fireEvent.click(undos[1]) // Undo step index 1
    await waitFor(() => expect(mockRemoveStep).toHaveBeenCalledWith("fs-1", 1))
  })

  it("updates the step list after undo", async () => {
    mockGetSteps.mockResolvedValue({ steps })
    mockRemoveStep.mockResolvedValue({ steps: [steps[0]], new_columns: ["order_date_year"] })
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() => screen.getAllByText("Undo"))
    fireEvent.click(screen.getAllByText("Undo")[1])
    await waitFor(() =>
      expect(screen.getByText(/1 transformation in pipeline/i)).toBeInTheDocument()
    )
  })

  it("calls onStepRemoved callback with new_columns", async () => {
    mockGetSteps.mockResolvedValue({ steps })
    const newCols = ["order_date_year", "order_date_month"]
    mockRemoveStep.mockResolvedValue({ steps: [], new_columns: newCols })
    const onStepRemoved = jest.fn()
    render(<PipelinePanel featureSetId="fs-1" onStepRemoved={onStepRemoved} />)
    await waitFor(() => screen.getAllByText("Undo"))
    fireEvent.click(screen.getAllByText("Undo")[0])
    await waitFor(() => expect(onStepRemoved).toHaveBeenCalledWith(newCols))
  })

  it("disables all Undo buttons while a removal is in flight", async () => {
    mockGetSteps.mockResolvedValue({ steps })
    // Keep removal in-flight
    mockRemoveStep.mockReturnValue(new Promise(() => {}))
    render(<PipelinePanel featureSetId="fs-1" />)
    await waitFor(() => screen.getAllByText("Undo"))
    fireEvent.click(screen.getAllByText("Undo")[0])
    await waitFor(() => {
      const buttons = screen.getAllByTitle("Undo this step")
      buttons.forEach((btn) => expect(btn).toBeDisabled())
    })
  })
})

// ---------------------------------------------------------------------------
// DatasetListPanel
// ---------------------------------------------------------------------------

const makeDataset = (overrides: Partial<DatasetListItem> = {}): DatasetListItem => ({
  dataset_id: "ds-1",
  filename: "sales.csv",
  row_count: 200,
  column_count: 8,
  uploaded_at: "2024-01-15T10:00:00Z",
  size_bytes: 4096,
  ...overrides,
})

describe("DatasetListPanel — loading state", () => {
  it("shows loading text while fetching datasets", () => {
    mockListByProject.mockReturnValue(new Promise(() => {}))
    render(<DatasetListPanel projectId="proj-1" />)
    expect(screen.getByText(/loading datasets/i)).toBeInTheDocument()
  })
})

describe("DatasetListPanel — dataset list", () => {
  it("shows '0 datasets' message when empty", async () => {
    mockListByProject.mockResolvedValue([])
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/0 datasets in this project/i)).toBeInTheDocument()
    )
  })

  it("renders each dataset's filename", async () => {
    mockListByProject.mockResolvedValue([
      makeDataset({ filename: "sales.csv" }),
      makeDataset({ dataset_id: "ds-2", filename: "customers.csv" }),
    ])
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText("sales.csv")).toBeInTheDocument())
    expect(screen.getByText("customers.csv")).toBeInTheDocument()
  })

  it("shows row and column counts", async () => {
    mockListByProject.mockResolvedValue([makeDataset({ row_count: 500, column_count: 12 })])
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/500 rows · 12 columns/i)).toBeInTheDocument()
    )
  })

  it("shows singular 'dataset' for 1 dataset", async () => {
    mockListByProject.mockResolvedValue([makeDataset()])
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/1 dataset in this project/i)).toBeInTheDocument()
    )
  })

  it("renders without error when listByProject rejects", async () => {
    mockListByProject.mockRejectedValue(new Error("network error"))
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/0 datasets in this project/i)).toBeInTheDocument()
    )
  })
})

describe("DatasetListPanel — merge button visibility", () => {
  it("does not show 'Merge two datasets' button when fewer than 2 datasets", async () => {
    mockListByProject.mockResolvedValue([makeDataset()])
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText("sales.csv"))
    expect(screen.queryByText(/merge two datasets/i)).not.toBeInTheDocument()
  })

  it("shows 'Merge two datasets' button when 2+ datasets present", async () => {
    mockListByProject.mockResolvedValue([
      makeDataset({ filename: "sales.csv" }),
      makeDataset({ dataset_id: "ds-2", filename: "customers.csv" }),
    ])
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/merge two datasets/i)).toBeInTheDocument()
    )
  })
})

describe("DatasetListPanel — merge UI open/close", () => {
  beforeEach(() => {
    mockListByProject.mockResolvedValue([
      makeDataset({ filename: "sales.csv" }),
      makeDataset({ dataset_id: "ds-2", filename: "customers.csv" }),
    ])
  })

  it("opens merge panel on button click", async () => {
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))
    expect(screen.getByText(/combine two datasets on a shared column/i)).toBeInTheDocument()
  })

  it("shows 'Cancel merge' text after opening", async () => {
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))
    expect(screen.getByText(/cancel merge/i)).toBeInTheDocument()
  })

  it("closes merge panel on cancel click", async () => {
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/cancel merge/i))
    expect(screen.queryByText(/combine two datasets on a shared column/i)).not.toBeInTheDocument()
  })
})

describe("DatasetListPanel — join key suggestions", () => {
  const twoDatasets = [
    makeDataset({ dataset_id: "ds-1", filename: "sales.csv" }),
    makeDataset({ dataset_id: "ds-2", filename: "customers.csv" }),
  ]
  const joinKeySuggestions: JoinKeySuggestion[] = [
    { name: "customer_id", dtype_left: "int64", dtype_right: "int64", unique_left: 200, unique_right: 200, uniqueness_left: 1.0, uniqueness_right: 1.0, recommended: true },
    { name: "region", dtype_left: "object", dtype_right: "object", unique_left: 5, unique_right: 5, uniqueness_left: 0.025, uniqueness_right: 0.025, recommended: false },
  ]

  beforeEach(() => {
    mockListByProject.mockResolvedValue(twoDatasets)
    mockJoinKeys.mockResolvedValue({ join_key_suggestions: joinKeySuggestions })
  })

  it("fetches join keys when both datasets are selected", async () => {
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    // First select = left dataset, second = right dataset
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })

    await waitFor(() => expect(mockJoinKeys).toHaveBeenCalledWith("ds-1", "ds-2"))
  })

  it("shows join key options after loading", async () => {
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })

    await waitFor(() => expect(screen.getByText(/customer_id ★/)).toBeInTheDocument())
  })

  it("shows 'no common columns' warning when join keys empty", async () => {
    mockJoinKeys.mockResolvedValue({ join_key_suggestions: [] })
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })

    await waitFor(() =>
      expect(screen.getByText(/no common columns found/i)).toBeInTheDocument()
    )
  })
})

describe("DatasetListPanel — merge action", () => {
  const twoDatasets = [
    makeDataset({ dataset_id: "ds-1", filename: "sales.csv" }),
    makeDataset({ dataset_id: "ds-2", filename: "customers.csv" }),
  ]
  const joinKeySuggestions: JoinKeySuggestion[] = [
    { name: "customer_id", dtype_left: "int64", dtype_right: "int64", unique_left: 200, unique_right: 200, uniqueness_left: 1.0, uniqueness_right: 1.0, recommended: true },
  ]
  const mergeResponse: MergeResponse = {
    dataset_id: "ds-merged",
    filename: "merged_sales_customers.csv",
    row_count: 185,
    column_count: 14,
    join_key: "customer_id",
    how: "inner",
    conflict_columns: [],
    preview: [],
    column_stats: [],
  }

  beforeEach(() => {
    mockListByProject.mockResolvedValue(twoDatasets)
    mockJoinKeys.mockResolvedValue({ join_key_suggestions: joinKeySuggestions })
    mockMerge.mockResolvedValue(mergeResponse)
  })

  it("calls merge API with correct params", async () => {
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })
    await waitFor(() => expect(mockJoinKeys).toHaveBeenCalled())

    // Re-fetch selects after merge UI is fully rendered
    fireEvent.click(screen.getByRole("button", { name: /merge datasets/i }))
    await waitFor(() =>
      expect(mockMerge).toHaveBeenCalledWith("proj-1", {
        dataset_id_1: "ds-1",
        dataset_id_2: "ds-2",
        join_key: "customer_id",
        how: "inner",
      })
    )
  })

  it("shows merge success message with row count", async () => {
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })
    await waitFor(() => expect(mockJoinKeys).toHaveBeenCalled())

    fireEvent.click(screen.getByRole("button", { name: /merge datasets/i }))
    await waitFor(() =>
      expect(screen.getByText(/185 rows · 14 columns/i)).toBeInTheDocument()
    )
  })

  it("calls onMerged callback with result", async () => {
    const onMerged = jest.fn()
    render(<DatasetListPanel projectId="proj-1" onMerged={onMerged} />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })
    await waitFor(() => expect(mockJoinKeys).toHaveBeenCalled())

    fireEvent.click(screen.getByRole("button", { name: /merge datasets/i }))
    await waitFor(() => expect(onMerged).toHaveBeenCalledWith(mergeResponse))
  })

  it("shows merge error when API fails", async () => {
    mockMerge.mockRejectedValue(new Error("merge failed"))
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })
    await waitFor(() => expect(mockJoinKeys).toHaveBeenCalled())

    fireEvent.click(screen.getByRole("button", { name: /merge datasets/i }))
    await waitFor(() =>
      expect(screen.getByText(/merge failed/i)).toBeInTheDocument()
    )
  })

  it("shows conflict columns count when columns were renamed", async () => {
    mockMerge.mockResolvedValue({
      ...mergeResponse,
      conflict_columns: ["region", "date"],
    })
    render(<DatasetListPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText(/merge two datasets/i))
    fireEvent.click(screen.getByText(/merge two datasets/i))

    const selects = screen.getAllByRole("combobox")
    fireEvent.change(selects[0], { target: { value: "ds-1" } })
    fireEvent.change(selects[1], { target: { value: "ds-2" } })
    await waitFor(() => expect(mockJoinKeys).toHaveBeenCalled())

    fireEvent.click(screen.getByRole("button", { name: /merge datasets/i }))
    await waitFor(() =>
      expect(screen.getByText(/2 columns renamed with suffixes/i)).toBeInTheDocument()
    )
  })
})

// ---------------------------------------------------------------------------
// FeatureImportancePanel
// ---------------------------------------------------------------------------

const makeFeature = (overrides: Partial<FeatureImportanceEntry> = {}): FeatureImportanceEntry => ({
  column: "revenue",
  importance_pct: 45.0,
  rank: 1,
  description: "Revenue is the top driver",
  ...overrides,
})

describe("FeatureImportancePanel — rendering", () => {
  it("renders target column name", () => {
    render(
      <FeatureImportancePanel
        features={[makeFeature()]}
        targetColumn="churn"
        problemType="classification"
      />
    )
    expect(screen.getByText("churn")).toBeInTheDocument()
  })

  it("shows 'Classification' for classification problem type", () => {
    render(
      <FeatureImportancePanel
        features={[makeFeature()]}
        targetColumn="churn"
        problemType="classification"
      />
    )
    expect(screen.getByText(/classification/i)).toBeInTheDocument()
  })

  it("shows 'Regression' for regression problem type", () => {
    render(
      <FeatureImportancePanel
        features={[makeFeature()]}
        targetColumn="revenue"
        problemType="regression"
      />
    )
    expect(screen.getByText(/regression/i)).toBeInTheDocument()
  })

  it("renders each feature column name", () => {
    render(
      <FeatureImportancePanel
        features={[
          makeFeature({ column: "region", importance_pct: 45.0, rank: 1 }),
          makeFeature({ column: "season", importance_pct: 30.0, rank: 2 }),
          makeFeature({ column: "category", importance_pct: 25.0, rank: 3 }),
        ]}
        targetColumn="revenue"
        problemType="regression"
      />
    )
    expect(screen.getByText("region")).toBeInTheDocument()
    expect(screen.getByText("season")).toBeInTheDocument()
    expect(screen.getByText("category")).toBeInTheDocument()
  })

  it("renders importance percentage values", () => {
    render(
      <FeatureImportancePanel
        features={[
          makeFeature({ column: "region", importance_pct: 45.3, rank: 1 }),
          makeFeature({ column: "season", importance_pct: 30.1, rank: 2 }),
        ]}
        targetColumn="revenue"
        problemType="regression"
      />
    )
    expect(screen.getByText("45.3%")).toBeInTheDocument()
    expect(screen.getByText("30.1%")).toBeInTheDocument()
  })

  it("renders empty panel gracefully when features is empty", () => {
    render(
      <FeatureImportancePanel
        features={[]}
        targetColumn="revenue"
        problemType="regression"
      />
    )
    // Should render the intro text without crashing
    expect(screen.getByText(/predicting/i)).toBeInTheDocument()
  })

  it("scales bar width: top feature always at 100%", () => {
    const { container } = render(
      <FeatureImportancePanel
        features={[
          makeFeature({ column: "top_feature", importance_pct: 50.0, rank: 1 }),
          makeFeature({ column: "second_feature", importance_pct: 25.0, rank: 2 }),
        ]}
        targetColumn="revenue"
        problemType="regression"
      />
    )
    const bars = container.querySelectorAll(".bg-primary")
    expect(bars[0]).toHaveStyle("width: 100%")
    expect(bars[1]).toHaveStyle("width: 50%")
  })
})
