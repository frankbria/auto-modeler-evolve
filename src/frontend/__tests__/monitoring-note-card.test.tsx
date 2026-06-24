/**
 * Tests for MonitoringNoteCard — the generic card shared by the readiness,
 * drift, health, and alerts SSE events (#17).
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { MonitoringNoteCard } from "@/components/chat/monitoring-note-card"
import type { MonitoringNote } from "@/lib/types"

const note: MonitoringNote = {
  kind: "alerts",
  title: "Deployment alerts",
  summary: "2 alert(s): 1 critical, 1 warning.",
  items: ["Model X is degrading", "Quota almost exhausted"],
  tone: "critical",
}

describe("MonitoringNoteCard", () => {
  it("renders title, summary and items", () => {
    render(<MonitoringNoteCard note={note} />)
    expect(screen.getByTestId("monitoring-note-title")).toHaveTextContent("Deployment alerts")
    expect(screen.getByTestId("monitoring-note-summary")).toHaveTextContent("2 alert(s)")
    const items = screen.getByTestId("monitoring-note-items")
    expect(items).toHaveTextContent("Model X is degrading")
    expect(items).toHaveTextContent("Quota almost exhausted")
  })

  it("exposes the kind for styling/testing", () => {
    render(<MonitoringNoteCard note={note} />)
    expect(screen.getByTestId("monitoring-note-card")).toHaveAttribute("data-kind", "alerts")
  })

  it("omits the items list when there are none", () => {
    render(<MonitoringNoteCard note={{ kind: "health", title: "Model health", summary: "All good." }} />)
    expect(screen.queryByTestId("monitoring-note-items")).toBeNull()
  })

  it("renders the error kind (graceful chat-stream failure, #18)", () => {
    render(
      <MonitoringNoteCard
        note={{
          kind: "error",
          title: "Something went wrong",
          summary: "The AI service is temporarily busy — please try again in a moment.",
          tone: "critical",
        }}
      />
    )
    const card = screen.getByTestId("monitoring-note-card")
    expect(card).toHaveAttribute("data-kind", "error")
    expect(card).toHaveTextContent("⛔")
    expect(screen.getByTestId("monitoring-note-summary")).toHaveTextContent("temporarily busy")
  })
})
