/**
 * Tests for ApiKeyChatCard component — API Key Management via Chat (Day 62).
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ApiKeyChatCard } from "@/components/chat/api-key-chat-card"
import type { ApiKeyResultInfo } from "@/lib/types"

const GENERATED_RESULT: ApiKeyResultInfo = {
  action: "generated",
  deployment_id: "dep-1",
  is_protected: true,
  api_key: "test_key_abc123xyz",
  summary: "API key generated. Your prediction endpoint is now protected.",
}

const REGENERATED_RESULT: ApiKeyResultInfo = {
  action: "regenerated",
  deployment_id: "dep-1",
  is_protected: true,
  api_key: "new_key_def456uvw",
  summary: "API key regenerated. Your prediction endpoint is now protected.",
}

const DISABLED_RESULT: ApiKeyResultInfo = {
  action: "disabled",
  deployment_id: "dep-1",
  is_protected: false,
  summary: "API key protection removed. Your prediction endpoint is now publicly accessible.",
}

const STATUS_PROTECTED: ApiKeyResultInfo = {
  action: "status",
  deployment_id: "dep-1",
  is_protected: true,
  summary: "Prediction endpoint is currently protected.",
}

const STATUS_OPEN: ApiKeyResultInfo = {
  action: "status",
  deployment_id: "dep-1",
  is_protected: false,
  summary: "Prediction endpoint is currently open (no key required).",
}

// ---------------------------------------------------------------------------
// Generated state
// ---------------------------------------------------------------------------

describe("ApiKeyChatCard — generated", () => {
  it("renders the generated heading", () => {
    render(<ApiKeyChatCard result={GENERATED_RESULT} />)
    expect(screen.getByText(/API Key Generated/i)).toBeInTheDocument()
  })

  it("shows Protected badge", () => {
    render(<ApiKeyChatCard result={GENERATED_RESULT} />)
    expect(screen.getByText("Protected")).toBeInTheDocument()
  })

  it("displays the API key value", () => {
    render(<ApiKeyChatCard result={GENERATED_RESULT} />)
    expect(screen.getByTestId("api-key-value")).toHaveTextContent("test_key_abc123xyz")
  })

  it("shows shown-once warning", () => {
    render(<ApiKeyChatCard result={GENERATED_RESULT} />)
    expect(screen.getByText(/shown once/i)).toBeInTheDocument()
  })

  it("shows copy button", () => {
    render(<ApiKeyChatCard result={GENERATED_RESULT} />)
    expect(screen.getByRole("button", { name: /copy api key/i })).toBeInTheDocument()
  })

  it("has accessible figcaption", () => {
    render(<ApiKeyChatCard result={GENERATED_RESULT} />)
    expect(screen.getByRole("region", { name: /API key generated/i })).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Regenerated state
// ---------------------------------------------------------------------------

describe("ApiKeyChatCard — regenerated", () => {
  it("renders the regenerated heading", () => {
    render(<ApiKeyChatCard result={REGENERATED_RESULT} />)
    expect(screen.getByText(/API Key Regenerated/i)).toBeInTheDocument()
  })

  it("shows the new key value", () => {
    render(<ApiKeyChatCard result={REGENERATED_RESULT} />)
    expect(screen.getByTestId("api-key-value")).toHaveTextContent("new_key_def456uvw")
  })
})

// ---------------------------------------------------------------------------
// Disabled state
// ---------------------------------------------------------------------------

describe("ApiKeyChatCard — disabled", () => {
  it("renders the removal heading", () => {
    render(<ApiKeyChatCard result={DISABLED_RESULT} />)
    expect(screen.getByText(/API Key Protection Removed/i)).toBeInTheDocument()
  })

  it("shows Open Access badge", () => {
    render(<ApiKeyChatCard result={DISABLED_RESULT} />)
    expect(screen.getByText("Open Access")).toBeInTheDocument()
  })

  it("does not show an api key value", () => {
    render(<ApiKeyChatCard result={DISABLED_RESULT} />)
    expect(screen.queryByTestId("api-key-value")).not.toBeInTheDocument()
  })

  it("shows re-enable prompt", () => {
    render(<ApiKeyChatCard result={DISABLED_RESULT} />)
    expect(screen.getByText(/generate an API key/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Status — protected
// ---------------------------------------------------------------------------

describe("ApiKeyChatCard — status protected", () => {
  it("renders Protected badge", () => {
    render(<ApiKeyChatCard result={STATUS_PROTECTED} />)
    expect(screen.getByText("Protected")).toBeInTheDocument()
  })

  it("renders status summary", () => {
    render(<ApiKeyChatCard result={STATUS_PROTECTED} />)
    expect(screen.getByText(/Prediction endpoint is currently protected/i)).toBeInTheDocument()
  })

  it("suggests regenerate or remove options", () => {
    render(<ApiKeyChatCard result={STATUS_PROTECTED} />)
    expect(screen.getByText(/regenerate my API key/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Status — open
// ---------------------------------------------------------------------------

describe("ApiKeyChatCard — status open", () => {
  it("renders Open Access badge", () => {
    render(<ApiKeyChatCard result={STATUS_OPEN} />)
    expect(screen.getByText("Open Access")).toBeInTheDocument()
  })

  it("renders status summary", () => {
    render(<ApiKeyChatCard result={STATUS_OPEN} />)
    expect(screen.getByText(/open \(no key required\)/i)).toBeInTheDocument()
  })

  it("suggests generating a key", () => {
    render(<ApiKeyChatCard result={STATUS_OPEN} />)
    expect(screen.getByText(/generate an API key/i)).toBeInTheDocument()
  })
})
