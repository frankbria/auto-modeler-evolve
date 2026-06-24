/**
 * Contract tests for the SSE handler map (#17).
 *
 * The chat dispatcher was a 239-branch if/else chain (54 duplicate dead
 * branches) that no test exercised: a typo'd type, a dropped backend event, or
 * a duplicated branch all stayed green. Now that dispatch is a handler MAP,
 * these tests pin the contract:
 *   - every handled type is unique and is a function (dedup proof);
 *   - no handler throws on a minimal frame (graceful on missing payload);
 *   - the five previously-dropped events now produce a card / tab switch.
 *
 * A backend-side test (`tests/test_sse_contract.py`) asserts every event the
 * backend EMITS has a handler here — that is the silent-drop guard.
 */
import { createSSEHandlers, SSE_EVENT_TYPES } from "../lib/sse-handlers"
import type { SSEHandlerDeps } from "../lib/sse-handlers"

// A deps object whose every property is a stable jest.fn(), created lazily.
function makeDeps() {
  const fns: Record<string, jest.Mock> = {}
  const deps = new Proxy(
    {},
    { get: (_t, p: string) => (fns[p] ??= jest.fn()) }
  ) as SSEHandlerDeps
  return { deps, fns }
}

describe("SSE handler map contract", () => {
  it("exposes a non-empty, duplicate-free set of event types", () => {
    expect(SSE_EVENT_TYPES.length).toBeGreaterThan(150)
    expect(new Set(SSE_EVENT_TYPES).size).toBe(SSE_EVENT_TYPES.length)
  })

  it("maps every event type to a function", () => {
    const { deps } = makeDeps()
    const handlers = createSSEHandlers(deps)
    for (const t of SSE_EVENT_TYPES) {
      expect(typeof handlers[t]).toBe("function")
    }
  })

  it("does not throw for any event type given a minimal frame", () => {
    const { deps } = makeDeps()
    const handlers = createSSEHandlers(deps)
    for (const t of SSE_EVENT_TYPES) {
      expect(() => handlers[t]({ type: t })).not.toThrow()
    }
  })

  it("dedupes branches that were duplicated in the old chain", () => {
    // These were among the 54 unreachable duplicate branches; each must appear once.
    for (const t of ["interaction", "segment_drift", "uptime_summary", "promotion_readiness"]) {
      expect(SSE_EVENT_TYPES.filter((x) => x === t)).toHaveLength(1)
    }
  })
})

describe("previously-dropped events (#17)", () => {
  for (const t of ["readiness", "drift", "health", "alerts", "history"]) {
    it(`handles '${t}' (was silently dropped)`, () => {
      expect(SSE_EVENT_TYPES).toContain(t)
    })
  }

  it("readiness -> monitoring note with score/checks", () => {
    const { deps, fns } = makeDeps()
    createSSEHandlers(deps).readiness({
      type: "readiness",
      readiness: { score: 90, verdict: "Ready to deploy", checks: [{ passed: true, label: "Trained" }] },
    })
    expect(fns.attachMonitoringNoteToLastMessage).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "readiness", tone: "good" })
    )
    const note = fns.attachMonitoringNoteToLastMessage.mock.calls[0][0]
    expect(note.summary).toContain("90")
    expect(note.items).toContain("✓ Trained")
  })

  it("drift -> monitoring note flagged critical on detected drift", () => {
    const { deps, fns } = makeDeps()
    createSSEHandlers(deps).drift({
      type: "drift",
      drift: { status: "drift_detected", drift_score: 0.42, explanation: "Inputs shifted" },
    })
    expect(fns.attachMonitoringNoteToLastMessage).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "drift", tone: "critical", summary: "Inputs shifted" })
    )
  })

  it("alerts -> monitoring note summarising counts", () => {
    const { deps, fns } = makeDeps()
    createSSEHandlers(deps).alerts({
      type: "alerts",
      alerts: { alert_count: 2, critical_count: 1, warning_count: 1, alerts: [{ message: "A" }, { message: "B" }] },
    })
    const note = fns.attachMonitoringNoteToLastMessage.mock.calls[0][0]
    expect(note.tone).toBe("critical")
    expect(note.items).toEqual(["A", "B"])
  })

  it("health -> monitoring note with health metrics", () => {
    const { deps, fns } = makeDeps()
    createSSEHandlers(deps).health({
      type: "health",
      health: { health_score: 88, status: "healthy", model_age_days: 3, algorithm: "rf" },
    })
    expect(fns.attachMonitoringNoteToLastMessage).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "health", tone: "good" })
    )
  })

  it("history -> switches to the Models tab (where Version History renders)", () => {
    const { deps, fns } = makeDeps()
    createSSEHandlers(deps).history({ type: "history", history: { project_id: "p1" } })
    expect(fns.setActiveTab).toHaveBeenCalledWith("models")
  })
})

describe("core handlers still dispatch correctly", () => {
  it("token appends content; done stops streaming; chart attaches", () => {
    const { deps, fns } = makeDeps()
    const h = createSSEHandlers(deps)
    h.token({ type: "token", content: "hi" })
    h.done({ type: "done" })
    h.chart({ type: "chart", chart: { kind: "bar" } })
    expect(fns.appendToLastMessage).toHaveBeenCalledWith("hi")
    expect(fns.setStreaming).toHaveBeenCalledWith(false)
    expect(fns.attachChartToLastMessage).toHaveBeenCalledWith({ kind: "bar" })
  })
})
