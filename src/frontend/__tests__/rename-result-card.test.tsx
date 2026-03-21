/**
 * Tests for RenameResultCard component and the attachRenameResultToLastMessage store action.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { RenameResultCard } from "@/components/data/rename-result-card"
import type { RenameResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const renameResult: RenameResult = {
  dataset_id: "ds-1",
  old_name: "revenue_usd",
  new_name: "Revenue",
  column_count: 5,
}

// --- Component tests ----------------------------------------------------

describe("RenameResultCard", () => {
  it("renders with testid", () => {
    render(<RenameResultCard result={renameResult} />)
    expect(screen.getByTestId("rename-result-card")).toBeInTheDocument()
  })

  it("shows the old column name with strikethrough", () => {
    render(<RenameResultCard result={renameResult} />)
    expect(screen.getByText("revenue_usd")).toBeInTheDocument()
  })

  it("shows the new column name", () => {
    render(<RenameResultCard result={renameResult} />)
    expect(screen.getByText("Revenue")).toBeInTheDocument()
  })

  it("shows column count", () => {
    render(<RenameResultCard result={renameResult} />)
    expect(screen.getByText(/5 columns/i)).toBeInTheDocument()
  })

  it("shows 'Column Renamed' heading", () => {
    render(<RenameResultCard result={renameResult} />)
    expect(screen.getByText(/Column Renamed/i)).toBeInTheDocument()
  })

  it("shows arrow separator between old and new name", () => {
    render(<RenameResultCard result={renameResult} />)
    expect(screen.getByText("→")).toBeInTheDocument()
  })
})

// --- Store action tests -------------------------------------------------

describe("attachRenameResultToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches rename_result to the last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "rename revenue_usd to Revenue", timestamp: "t1" },
        { role: "assistant", content: "Done! I've renamed the column.", timestamp: "t2" },
      ],
    })
    const { attachRenameResultToLastMessage } = useAppStore.getState()
    attachRenameResultToLastMessage(renameResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].rename_result).toEqual(renameResult)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "rename revenue to Sales", timestamp: "t1" },
      ],
    })
    const { attachRenameResultToLastMessage } = useAppStore.getState()
    attachRenameResultToLastMessage(renameResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].rename_result).toBeUndefined()
  })

  it("is a no-op on empty message list", () => {
    useAppStore.setState({ messages: [] })
    const { attachRenameResultToLastMessage } = useAppStore.getState()
    expect(() => attachRenameResultToLastMessage(renameResult)).not.toThrow()
  })
})
