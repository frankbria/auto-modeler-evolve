/**
 * Tests for ErrorDisplay (#17) — the reusable API-error surface with retry.
 */
import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { ErrorDisplay } from "@/components/ui/error-display"

describe("ErrorDisplay", () => {
  it("renders the message and optional details", () => {
    render(<ErrorDisplay message="Couldn't load" details="HTTP 500" />)
    expect(screen.getByTestId("error-display-message")).toHaveTextContent("Couldn't load")
    expect(screen.getByTestId("error-display-details")).toHaveTextContent("HTTP 500")
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("calls onRetry when the retry button is clicked", () => {
    const onRetry = jest.fn()
    render(<ErrorDisplay message="Failed" onRetry={onRetry} />)
    fireEvent.click(screen.getByTestId("error-display-retry"))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it("omits the retry button when no onRetry is given", () => {
    render(<ErrorDisplay message="Failed" />)
    expect(screen.queryByTestId("error-display-retry")).toBeNull()
  })
})
