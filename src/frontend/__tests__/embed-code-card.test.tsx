import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { EmbedCodeCard } from "@/components/deploy/embed-code-card"
import type { EmbedCodeResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const base: EmbedCodeResult = {
  deployment_id: "dep-abc",
  dashboard_url: "/predict/dep-abc",
  title: "Revenue Predictor",
  width: "100%",
  height: "700",
  summary: "Here is the embed code for 'Revenue Predictor'.",
}

describe("EmbedCodeCard", () => {
  it("renders the card container with accessible label", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByRole("region", { name: /embed code card/i })).toBeInTheDocument()
  })

  it("shows the 🖼️ icon", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByText("🖼️")).toBeInTheDocument()
  })

  it("shows 'Embed Prediction Dashboard' heading", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByTestId("embed-code-heading")).toHaveTextContent(
      "Embed Prediction Dashboard"
    )
  })

  it("displays the dashboard title", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByTestId("embed-dashboard-title")).toHaveTextContent("Revenue Predictor")
  })

  it("renders the 'Ready to embed' badge", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByText("Ready to embed")).toBeInTheDocument()
  })

  it("shows the code block with iframe HTML", () => {
    render(<EmbedCodeCard result={base} />)
    const codeBlock = screen.getByTestId("embed-code-block")
    expect(codeBlock.textContent).toContain("<iframe")
    expect(codeBlock.textContent).toContain("/predict/dep-abc")
    expect(codeBlock.textContent).toContain("Revenue Predictor")
  })

  it("shows size preset buttons", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByTestId("size-presets")).toBeInTheDocument()
    expect(screen.getByTestId("preset-full")).toHaveTextContent("Full Width")
    expect(screen.getByTestId("preset-fixed")).toHaveTextContent("Fixed")
    expect(screen.getByTestId("preset-compact")).toHaveTextContent("Compact")
  })

  it("full width preset is active by default", () => {
    render(<EmbedCodeCard result={base} />)
    const fullBtn = screen.getByTestId("preset-full")
    expect(fullBtn).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByTestId("preset-fixed")).toHaveAttribute("aria-pressed", "false")
  })

  it("changing preset updates aria-pressed state", () => {
    render(<EmbedCodeCard result={base} />)
    const fixedBtn = screen.getByTestId("preset-fixed")
    fireEvent.click(fixedBtn)
    expect(fixedBtn).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByTestId("preset-full")).toHaveAttribute("aria-pressed", "false")
  })

  it("changing preset updates the iframe dimensions in code block", () => {
    render(<EmbedCodeCard result={base} />)
    fireEvent.click(screen.getByTestId("preset-compact"))
    const codeBlock = screen.getByTestId("embed-code-block")
    expect(codeBlock.textContent).toContain("600px")
    expect(codeBlock.textContent).toContain("500px")
  })

  it("shows the copy button with accessible label", () => {
    render(<EmbedCodeCard result={base} />)
    expect(
      screen.getByRole("button", { name: /copy embed code to clipboard/i })
    ).toBeInTheDocument()
  })

  it("shows the summary text when provided", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByTestId("embed-summary")).toHaveTextContent("Revenue Predictor")
  })

  it("omits summary section when not provided", () => {
    render(<EmbedCodeCard result={{ ...base, summary: undefined }} />)
    expect(screen.queryByTestId("embed-summary")).not.toBeInTheDocument()
  })

  it("shows portal embed instructions", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByText(/SharePoint/i)).toBeInTheDocument()
    expect(screen.getByText(/Notion/i)).toBeInTheDocument()
  })

  it("renders the footer explainer text", () => {
    render(<EmbedCodeCard result={base} />)
    expect(screen.getByText(/works fully inside the iframe/i)).toBeInTheDocument()
  })
})

describe("EmbedCodeCard Zustand store", () => {
  it("attachEmbedCodeToLastMessage attaches to the last assistant message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { role: "user", content: "give me embed code" },
      { role: "assistant", content: "Here is your embed code." },
    ])
    store.attachEmbedCodeToLastMessage(base)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].embed_code).toEqual(base)
  })

  it("does not attach when last message is user", () => {
    const store = useAppStore.getState()
    store.setMessages([{ role: "user", content: "give me embed code" }])
    store.attachEmbedCodeToLastMessage(base)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].embed_code).toBeUndefined()
  })
})
