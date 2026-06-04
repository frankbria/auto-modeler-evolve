/**
 * Tests for DeploymentThroughputCard and attachThroughputAssessmentToLastMessage.
 *
 * Covers:
 *  1.  Renders container with data-testid "deployment-throughput-card"
 *  2.  no_data verdict shows no-data-message text
 *  3.  no_data verdict does not show latency stats
 *  4.  fast verdict shows emerald border class
 *  5.  moderate verdict shows amber border class
 *  6.  slow verdict shows orange border class
 *  7.  very_slow verdict shows rose border class
 *  8.  instant verdict shows emerald border
 *  9.  verdict badge is rendered with correct label
 * 10.  target_n badge shows formatted prediction count
 * 11.  p50 latency stat box is rendered
 * 12.  p95 latency stat box is rendered
 * 13.  p99 latency stat box is rendered
 * 14.  max RPS stat box is rendered
 * 15.  duration estimate row shows serial_duration
 * 16.  sample count text shows n_samples
 * 17.  summary paragraph is rendered
 * 18.  sr-only figcaption present for accessibility
 * 19.  Store: attachThroughputAssessmentToLastMessage attaches to last assistant message
 * 20.  Store: does not attach to user message
 * 21.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DeploymentThroughputCard } from "@/components/deploy/deployment-throughput-card"
import type { DeploymentThroughputResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeResult(overrides: Partial<DeploymentThroughputResult> = {}): DeploymentThroughputResult {
  return {
    deployment_id: "dep-1",
    verdict: "fast",
    n_samples: 50,
    target_n: 1000,
    p50_ms: 45.0,
    p95_ms: 80.0,
    p99_ms: 120.0,
    mean_ms: 52.0,
    max_rps: 12.5,
    serial_seconds: 80.0,
    serial_duration: "1 minute 20s",
    summary: "Based on 50 measured predictions, p95 latency is 80 ms. Throughput: ~12.5 RPS.",
    ...overrides,
  }
}

function makeNoDataResult(): DeploymentThroughputResult {
  return {
    deployment_id: "dep-2",
    verdict: "no_data",
    n_samples: 2,
    target_n: 1000,
    p50_ms: null,
    p95_ms: null,
    p99_ms: null,
    mean_ms: null,
    max_rps: null,
    serial_seconds: null,
    serial_duration: null,
    summary: "Not enough latency data yet — make at least 5 predictions to see throughput estimates.",
  }
}

// 1. Container renders
test("renders container with data-testid", () => {
  render(<DeploymentThroughputCard result={makeResult()} />)
  expect(screen.getByTestId("deployment-throughput-card")).toBeInTheDocument()
})

// 2. no_data shows no-data message
test("no_data verdict shows no-data-message", () => {
  render(<DeploymentThroughputCard result={makeNoDataResult()} />)
  expect(screen.getByTestId("no-data-message")).toBeInTheDocument()
})

// 3. no_data does not show latency stats
test("no_data verdict does not show latency stats", () => {
  render(<DeploymentThroughputCard result={makeNoDataResult()} />)
  expect(screen.queryByTestId("latency-stats")).not.toBeInTheDocument()
})

// 4. fast verdict shows emerald border
test("fast verdict shows emerald border class", () => {
  const { container } = render(<DeploymentThroughputCard result={makeResult({ verdict: "fast" })} />)
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-emerald")
})

// 5. moderate verdict shows amber border
test("moderate verdict shows amber border class", () => {
  const { container } = render(<DeploymentThroughputCard result={makeResult({ verdict: "moderate" })} />)
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-amber")
})

// 6. slow verdict shows orange border
test("slow verdict shows orange border class", () => {
  const { container } = render(<DeploymentThroughputCard result={makeResult({ verdict: "slow" })} />)
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-orange")
})

// 7. very_slow verdict shows rose border
test("very_slow verdict shows rose border class", () => {
  const { container } = render(<DeploymentThroughputCard result={makeResult({ verdict: "very_slow" })} />)
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-rose")
})

// 8. instant verdict shows emerald border
test("instant verdict shows emerald border class", () => {
  const { container } = render(<DeploymentThroughputCard result={makeResult({ verdict: "instant" })} />)
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-emerald")
})

// 9. verdict badge rendered
test("verdict badge is rendered", () => {
  render(<DeploymentThroughputCard result={makeResult({ verdict: "fast" })} />)
  expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Fast")
})

// 10. target_n badge
test("target_n badge shows formatted prediction count", () => {
  render(<DeploymentThroughputCard result={makeResult({ target_n: 5000 })} />)
  expect(screen.getByTestId("target-n-badge")).toHaveTextContent("5,000")
})

// 11-14. Latency stats
test("p50 latency stat box is rendered", () => {
  render(<DeploymentThroughputCard result={makeResult()} />)
  const stats = screen.getByTestId("latency-stats")
  expect(stats).toHaveTextContent("45")
})

test("p95 latency stat box is rendered", () => {
  render(<DeploymentThroughputCard result={makeResult()} />)
  const stats = screen.getByTestId("latency-stats")
  expect(stats).toHaveTextContent("80")
})

test("p99 latency stat box is rendered", () => {
  render(<DeploymentThroughputCard result={makeResult()} />)
  const stats = screen.getByTestId("latency-stats")
  expect(stats).toHaveTextContent("120")
})

test("max RPS stat box is rendered", () => {
  render(<DeploymentThroughputCard result={makeResult({ max_rps: 12.5 })} />)
  const stats = screen.getByTestId("latency-stats")
  expect(stats).toHaveTextContent("12.5")
})

// 15. Duration estimate
test("duration estimate row shows serial_duration", () => {
  render(<DeploymentThroughputCard result={makeResult({ serial_duration: "1 minute 20s" })} />)
  expect(screen.getByTestId("duration-estimate")).toHaveTextContent("1 minute 20s")
})

// 16. Sample count
test("sample count text shows n_samples", () => {
  render(<DeploymentThroughputCard result={makeResult({ n_samples: 42 })} />)
  expect(screen.getByTestId("sample-count")).toHaveTextContent("42")
})

// 17. Summary paragraph
test("summary paragraph is rendered", () => {
  const summary = "Based on 50 measured predictions, p95 latency is 80 ms."
  render(<DeploymentThroughputCard result={makeResult({ summary })} />)
  expect(screen.getByTestId("summary")).toHaveTextContent("Based on 50")
})

// 18. Accessibility figcaption
test("sr-only figcaption is present", () => {
  const { container } = render(<DeploymentThroughputCard result={makeResult()} />)
  const figcaption = container.querySelector("figcaption")
  expect(figcaption).toBeInTheDocument()
  expect(figcaption).toHaveClass("sr-only")
})

// 19-21. Store actions
describe("attachThroughputAssessmentToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  test("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "test", id: "u1" },
        { role: "assistant", content: "reply", id: "a1" },
      ],
    })
    const result = makeResult()
    useAppStore.getState().attachThroughputAssessmentToLastMessage(result)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].throughput_assessment).toEqual(result)
  })

  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test", id: "u1" }],
    })
    const result = makeResult()
    useAppStore.getState().attachThroughputAssessmentToLastMessage(result)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].throughput_assessment).toBeUndefined()
  })

  test("does not crash with empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachThroughputAssessmentToLastMessage(makeResult())
    ).not.toThrow()
  })
})
