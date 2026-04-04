import { render, screen } from "@testing-library/react"
import { ConversationExportCard } from "@/components/chat/conversation-export-card"
import type { ConversationExportInfo } from "@/lib/types"

const baseInfo: ConversationExportInfo = {
  project_id: "proj-1",
  download_url: "/api/chat/proj-1/export",
  message_count: 5,
  dataset_name: "sales_data.csv",
}

const noDatasetInfo: ConversationExportInfo = {
  project_id: "proj-1",
  download_url: "/api/chat/proj-1/export",
  message_count: 0,
  dataset_name: null,
}

describe("ConversationExportCard — rendering", () => {
  it("renders the card heading", () => {
    render(<ConversationExportCard info={baseInfo} />)
    expect(screen.getByText("Analysis Report Ready")).toBeInTheDocument()
  })

  it("shows message count badge", () => {
    render(<ConversationExportCard info={baseInfo} />)
    expect(screen.getByText(/5 AI responses/i)).toBeInTheDocument()
  })

  it("shows singular 'response' for count of 1", () => {
    render(<ConversationExportCard info={{ ...baseInfo, message_count: 1 }} />)
    expect(screen.getByText(/1 AI response$/i)).toBeInTheDocument()
  })

  it("shows dataset name badge when provided", () => {
    render(<ConversationExportCard info={baseInfo} />)
    expect(screen.getByText("sales_data.csv")).toBeInTheDocument()
  })

  it("does not show dataset badge when null", () => {
    render(<ConversationExportCard info={noDatasetInfo} />)
    expect(screen.queryByText(/\.csv/i)).not.toBeInTheDocument()
  })

  it("renders a download link", () => {
    render(<ConversationExportCard info={baseInfo} />)
    const link = screen.getByRole("link", { name: /download/i })
    expect(link).toBeInTheDocument()
  })

  it("download link contains the export URL", () => {
    render(<ConversationExportCard info={baseInfo} />)
    const link = screen.getByRole("link", { name: /download/i })
    expect(link.getAttribute("href")).toContain("/api/chat/proj-1/export")
  })

  it("has download attribute on the link", () => {
    render(<ConversationExportCard info={baseInfo} />)
    const link = screen.getByRole("link", { name: /download/i })
    expect(link).toHaveAttribute("download")
  })

  it("has accessible figure role with aria-label", () => {
    render(<ConversationExportCard info={baseInfo} />)
    expect(
      screen.getByRole("figure", { name: /conversation export/i })
    ).toBeInTheDocument()
  })

  it("shows 0 message count without crashing", () => {
    render(<ConversationExportCard info={noDatasetInfo} />)
    expect(screen.getByText(/0 AI responses/i)).toBeInTheDocument()
  })
})
