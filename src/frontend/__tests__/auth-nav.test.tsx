/**
 * Unit tests for the auth-aware nav (components/auth/auth-nav.tsx).
 *
 * Shows the signed-in user's email and a sign-out control; renders nothing for
 * anonymous visitors so public pages (e.g. /login, /predict/[id]) stay clean.
 */

import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { AuthNav } from "../components/auth/auth-nav"
import { useAppStore } from "../lib/store"
import { api } from "../lib/api"

const pushMock = jest.fn()

jest.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, replace: jest.fn(), refresh: jest.fn() }),
}))

jest.mock("../lib/api", () => ({
  api: { auth: { login: jest.fn(), register: jest.fn(), me: jest.fn() } },
}))

const USER = { id: "u1", email: "ada@b.com", name: "Ada" }

beforeEach(() => {
  jest.clearAllMocks()
  window.localStorage.clear()
  useAppStore.getState().logout()
})

describe("AuthNav", () => {
  it("renders nothing when not authenticated", () => {
    const { container } = render(<AuthNav />)
    expect(container).toBeEmptyDOMElement()
  })

  it("shows the user's email when authenticated", async () => {
    useAppStore.setState({ user: USER, token: "tok-1", isAuthenticated: true })
    render(<AuthNav />)
    expect(await screen.findByText("ada@b.com")).toBeInTheDocument()
  })

  it("signs out and redirects to /login", async () => {
    useAppStore.setState({ user: USER, token: "tok-1", isAuthenticated: true })
    render(<AuthNav />)
    fireEvent.click(await screen.findByRole("button", { name: /sign out/i }))
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/login"))
    expect(useAppStore.getState().isAuthenticated).toBe(false)
  })
})
