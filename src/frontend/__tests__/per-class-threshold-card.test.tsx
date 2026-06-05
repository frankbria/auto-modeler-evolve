import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { PerClassThresholdCard } from "@/components/models/per-class-threshold-card"
import type { PerClassThresholdResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const mockSweep = Array.from({ length: 19 }, (_, i) => ({
  threshold: parseFloat(((i + 1) * 0.05).toFixed(2)),
  precision: 0.5 + i * 0.01,
  recall: 0.9 - i * 0.02,
  f1: 0.6 + i * 0.005,
  positive_rate: 0.4 - i * 0.01,
}))

const mockResult: PerClassThresholdResult = {
  model_run_id: "run-1",
  algorithm: "random_forest_classifier",
  target_col: "priority",
  n_classes: 3,
  n_samples: 300,
  n_actionable: 2,
  summary: "2 of 3 classes benefit from threshold tuning: 'high', 'critical'.",
  classes: [
    {
      class_name: "low",
      class_index: 0,
      n_positive: 120,
      prevalence: 0.4,
      sweep: mockSweep,
      optimal_threshold: 0.5,
      optimal_f1: 0.72,
      optimal_precision: 0.7,
      optimal_recall: 0.75,
      default_f1: 0.72,
      f1_gain: 0.0,
      direction: "keep",
      recommendation: "Default 50% is already optimal for 'low'.",
      is_actionable: false,
    },
    {
      class_name: "high",
      class_index: 1,
      n_positive: 90,
      prevalence: 0.3,
      sweep: mockSweep,
      optimal_threshold: 0.35,
      optimal_f1: 0.8,
      optimal_precision: 0.75,
      optimal_recall: 0.85,
      default_f1: 0.72,
      f1_gain: 0.08,
      direction: "lower",
      recommendation: "Lower to 35%: catch more 'high' cases, +8pp F1 gain.",
      is_actionable: true,
    },
    {
      class_name: "critical",
      class_index: 2,
      n_positive: 90,
      prevalence: 0.3,
      sweep: mockSweep,
      optimal_threshold: 0.65,
      optimal_f1: 0.78,
      optimal_precision: 0.82,
      optimal_recall: 0.74,
      default_f1: 0.72,
      f1_gain: 0.06,
      direction: "raise",
      recommendation: "Raise to 65%: fewer false 'critical' predictions, +6pp F1 gain.",
      is_actionable: true,
    },
  ],
}

jest.mock("recharts", () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ReactMock = require("react") as typeof React
  const mock = (name: string) =>
    function MockChart(props: Record<string, unknown>) {
      return ReactMock.createElement("div", { "data-testid": `recharts-${name}` }, props.children as React.ReactNode)
    }
  return {
    ResponsiveContainer: mock("responsive-container"),
    LineChart: mock("line-chart"),
    Line: () => null,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
    ReferenceLine: () => null,
  }
})

describe("PerClassThresholdCard", () => {
  it("renders the card header", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByText("Per-Class Threshold Tuning")).toBeInTheDocument()
  })

  it("shows n-classes badge", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByTestId("n-classes-badge")).toHaveTextContent("3 classes")
  })

  it("shows actionable badge when n_actionable > 0", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByTestId("actionable-badge")).toHaveTextContent("2 actionable")
  })

  it("does not show actionable badge when n_actionable is 0", () => {
    const noActionResult = { ...mockResult, n_actionable: 0 }
    render(<PerClassThresholdCard result={noActionResult} />)
    expect(screen.queryByTestId("actionable-badge")).not.toBeInTheDocument()
  })

  it("shows target column", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByText("Target: priority")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByTestId("summary")).toHaveTextContent("2 of 3 classes")
  })

  it("renders one row per class", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const rows = screen.getAllByTestId("per-class-row")
    expect(rows).toHaveLength(3)
  })

  it("shows class names", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByText("low")).toBeInTheDocument()
    expect(screen.getByText("high")).toBeInTheDocument()
    expect(screen.getByText("critical")).toBeInTheDocument()
  })

  it("shows direction badges", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const badges = screen.getAllByTestId("direction-badge")
    const texts = badges.map((b) => b.textContent)
    expect(texts).toEqual(expect.arrayContaining([
      expect.stringContaining("Default OK"),
      expect.stringContaining("Lower"),
      expect.stringContaining("Raise"),
    ]))
  })

  it("shows keep badge for non-actionable class", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const badges = screen.getAllByTestId("direction-badge")
    const keepBadge = badges.find((b) => b.textContent?.includes("Default OK"))
    expect(keepBadge).toBeTruthy()
  })

  it("shows f1 gain badge for actionable classes", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const gainBadges = screen.getAllByTestId("f1-gain-badge")
    expect(gainBadges).toHaveLength(2)
    expect(gainBadges[0]).toHaveTextContent("+8pp F1")
  })

  it("shows recommendations", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const recs = screen.getAllByTestId("class-recommendation")
    const texts = recs.map((r) => r.textContent ?? "")
    expect(texts.some((t) => t.includes("Lower to 35%"))).toBe(true)
    expect(texts.some((t) => t.includes("Raise to 65%"))).toBe(true)
  })

  it("shows how-to-use footer", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByText(/How to use:/)).toBeInTheDocument()
  })

  it("has accessible region label", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    expect(screen.getByRole("region", { name: /per-class threshold analysis/i })).toBeInTheDocument()
  })

  it("has sr-only figcaption", () => {
    const { container } = render(<PerClassThresholdCard result={mockResult} />)
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).toBeTruthy()
    expect(caption?.textContent).toContain("priority")
  })

  it("shows expand button for actionable classes", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const expandButtons = screen.getAllByText(/Show sweep/)
    expect(expandButtons).toHaveLength(2)
  })

  it("does not show expand button for non-actionable classes", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    // 'low' class is not actionable — no expand button for it
    const rows = screen.getAllByTestId("per-class-row")
    const lowRow = rows[0] // first class is 'low'
    expect(lowRow.querySelector("button")).toBeNull()
  })

  it("expands to show chart when Show sweep clicked", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const expandButtons = screen.getAllByText(/Show sweep/)
    fireEvent.click(expandButtons[0])
    expect(screen.getByTestId("class-sweep-chart")).toBeInTheDocument()
  })

  it("collapses chart when Hide chart clicked", () => {
    render(<PerClassThresholdCard result={mockResult} />)
    const expandButtons = screen.getAllByText(/Show sweep/)
    fireEvent.click(expandButtons[0])
    expect(screen.getByText("Hide chart")).toBeInTheDocument()
    fireEvent.click(screen.getByText("Hide chart"))
    expect(screen.queryByTestId("class-sweep-chart")).not.toBeInTheDocument()
  })

  it("uses violet border when n_actionable > 0", () => {
    const { container } = render(<PerClassThresholdCard result={mockResult} />)
    const figure = container.querySelector("figure")
    expect(figure?.className).toContain("border-violet-300")
  })

  it("uses gray border when n_actionable is 0", () => {
    const noActionResult = { ...mockResult, n_actionable: 0 }
    const { container } = render(<PerClassThresholdCard result={noActionResult} />)
    const figure = container.querySelector("figure")
    expect(figure?.className).toContain("border-gray-200")
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachPerClassThresholdToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "hi", timestamp: new Date().toISOString() },
        { role: "assistant", content: "hello", timestamp: new Date().toISOString() },
      ],
    })
  })

  it("attaches per_class_threshold to last assistant message", () => {
    useAppStore.getState().attachPerClassThresholdToLastMessage(mockResult)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1] as Record<string, unknown>
    expect(last.per_class_threshold).toEqual(mockResult)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "hi", timestamp: new Date().toISOString() },
      ],
    })
    useAppStore.getState().attachPerClassThresholdToLastMessage(mockResult)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1] as Record<string, unknown>
    expect(last.per_class_threshold).toBeUndefined()
  })

  it("preserves other message fields", () => {
    useAppStore.getState().attachPerClassThresholdToLastMessage(mockResult)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.content).toBe("hello")
    expect(last.role).toBe("assistant")
  })
})
