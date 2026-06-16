import { render, screen, fireEvent } from "@testing-library/react"
import { ConversationExportCard } from "@/components/chat/conversation-export-card"
import type { ConversationExportInfo } from "@/lib/types"

jest.mock("@/lib/api", () => {
  const actual = jest.requireActual("@/lib/api")
  return { ...actual, downloadFile: jest.fn().mockResolvedValue(undefined) }
})
import { downloadFile } from "@/lib/api"

beforeEach(() => jest.clearAllMocks())

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

  it("renders a download button", () => {
    render(<ConversationExportCard info={baseInfo} />)
    const btn = screen.getByRole("button", { name: /download/i })
    expect(btn).toBeInTheDocument()
  })

  it("downloads via the authenticated helper with the export path (no token in URL)", () => {
    render(<ConversationExportCard info={baseInfo} />)
    fireEvent.click(screen.getByRole("button", { name: /download/i }))
    expect(downloadFile).toHaveBeenCalledTimes(1)
    const [url] = (downloadFile as jest.Mock).mock.calls[0]
    expect(url).toBe("/api/chat/proj-1/export")
    expect(url).not.toMatch(/token/i)
  })

  it("does not render a raw download link/href", () => {
    render(<ConversationExportCard info={baseInfo} />)
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
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
