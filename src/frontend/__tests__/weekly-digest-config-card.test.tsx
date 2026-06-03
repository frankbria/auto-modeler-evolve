import React from "react"
import { render, screen } from "@testing-library/react"
import WeeklyDigestConfigCard from "@/components/deploy/weekly-digest-config-card"
import { WeeklyDigestConfigResult } from "@/lib/types"

const enabledData: WeeklyDigestConfigResult = {
  deployment_id: "dep-1",
  action: "enabled",
  enabled: true,
  day_of_week: 0,
  day_name: "Monday",
  send_hour: 9,
  last_sent_at: null,
  summary: "Weekly digest enabled.",
}

const disabledData: WeeklyDigestConfigResult = {
  deployment_id: "dep-1",
  action: "disabled",
  enabled: false,
  day_of_week: 0,
  day_name: "Monday",
  send_hour: 9,
  last_sent_at: null,
  summary: "Weekly digest disabled.",
}

const statusData: WeeklyDigestConfigResult = {
  deployment_id: "dep-1",
  action: "status",
  enabled: true,
  day_of_week: 2,
  day_name: "Wednesday",
  send_hour: 8,
  last_sent_at: "2025-06-04T08:05:00",
  summary: "Weekly digest is enabled.",
}

describe("WeeklyDigestConfigCard", () => {
  it("renders the card root element", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByTestId("weekly-digest-config-card")).toBeInTheDocument()
  })

  it("shows enabled badge when enabled", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByTestId("digest-enabled-badge")).toBeInTheDocument()
  })

  it("shows disabled badge when disabled", () => {
    render(<WeeklyDigestConfigCard data={disabledData} />)
    expect(screen.getByTestId("digest-disabled-badge")).toBeInTheDocument()
  })

  it("shows schedule grid when enabled", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByTestId("digest-schedule-grid")).toBeInTheDocument()
  })

  it("does not show schedule grid when disabled", () => {
    render(<WeeklyDigestConfigCard data={disabledData} />)
    expect(screen.queryByTestId("digest-schedule-grid")).not.toBeInTheDocument()
  })

  it("shows correct day name", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByTestId("digest-day")).toHaveTextContent("Monday")
  })

  it("shows correct send time", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByTestId("digest-time")).toHaveTextContent("09:00 UTC")
  })

  it("shows wednesday and 8am from status data", () => {
    render(<WeeklyDigestConfigCard data={statusData} />)
    expect(screen.getByTestId("digest-day")).toHaveTextContent("Wednesday")
    expect(screen.getByTestId("digest-time")).toHaveTextContent("08:00 UTC")
  })

  it("shows last_sent_at when present", () => {
    render(<WeeklyDigestConfigCard data={statusData} />)
    expect(screen.getByTestId("digest-last-sent")).toBeInTheDocument()
  })

  it("does not show last_sent_at when null", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.queryByTestId("digest-last-sent")).not.toBeInTheDocument()
  })

  it("shows webhook note explaining how it works", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByTestId("digest-webhook-note")).toBeInTheDocument()
    expect(screen.getByTestId("digest-webhook-note")).toHaveTextContent("weekly_digest")
  })

  it("shows summary text", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByTestId("digest-summary")).toHaveTextContent("Weekly digest enabled.")
  })

  it("renders sr-only figcaption for accessibility", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    const figcaption = document.querySelector("figcaption")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption).toHaveClass("sr-only")
  })

  it("shows heading for enabled state", () => {
    render(<WeeklyDigestConfigCard data={enabledData} />)
    expect(screen.getByText("Weekly Digest Enabled")).toBeInTheDocument()
  })

  it("shows heading for disabled state", () => {
    render(<WeeklyDigestConfigCard data={disabledData} />)
    expect(screen.getByText("Weekly Digest Disabled")).toBeInTheDocument()
  })

  it("shows status heading for status action", () => {
    render(<WeeklyDigestConfigCard data={statusData} />)
    expect(screen.getByText("Weekly Digest Status")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action test
// ---------------------------------------------------------------------------

describe("attachWeeklyDigestConfigToLastMessage store action", () => {
  it("attaches weekly_digest_config to the last assistant message", () => {
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()

    store.setMessages([
      { id: "1", role: "user", content: "enable weekly digest" },
      { id: "2", role: "assistant", content: "Done!" },
    ])

    store.attachWeeklyDigestConfigToLastMessage(enabledData)

    const msgs = useAppStore.getState().messages
    expect(msgs[1].weekly_digest_config).toEqual(enabledData)
  })

  it("does not attach to user messages", () => {
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()

    store.setMessages([
      { id: "1", role: "user", content: "enable weekly digest" },
    ])

    store.attachWeeklyDigestConfigToLastMessage(enabledData)

    const msgs = useAppStore.getState().messages
    expect(msgs[0].weekly_digest_config).toBeUndefined()
  })
})
