/**
 * Unit tests for the client-side route guard (components/auth/require-auth.tsx).
 *
 * The guard renders its children only for authenticated users. Unauthenticated
 * visitors are first given a chance to hydrate from a stored token, and if that
 * fails they are redirected to /login with a redirect back to the current path.
 */

import { render, screen, waitFor } from "@testing-library/react"
import { RequireAuth } from "../components/auth/require-auth"
import { useAppStore } from "../lib/store"
import { api } from "../lib/api"
import { setToken } from "../lib/auth-token"

const replaceMock = jest.fn()

jest.mock("next/navigation", () => ({
  useRouter: () => ({ replace: replaceMock, push: jest.fn(), refresh: jest.fn() }),
  usePathname: () => "/project/abc",
}))

jest.mock("../lib/api", () => ({
  api: { auth: { login: jest.fn(), register: jest.fn(), me: jest.fn() } },
}))

const mockMe = api.auth.me as jest.MockedFunction<typeof api.auth.me>
const USER = { id: "u1", email: "a@b.com", name: "Ada" }

beforeEach(() => {
  jest.clearAllMocks()
  window.localStorage.clear()
  useAppStore.getState().logout()
})

describe("RequireAuth", () => {
  it("renders children when already authenticated", async () => {
    useAppStore.setState({ user: USER, token: "tok-1", isAuthenticated: true })
    render(
      <RequireAuth>
        <div>secret content</div>
      </RequireAuth>
    )
    expect(await screen.findByText("secret content")).toBeInTheDocument()
    expect(replaceMock).not.toHaveBeenCalled()
  })

  it("redirects to /login (with redirect back) when unauthenticated and no token", async () => {
    render(
      <RequireAuth>
        <div>secret content</div>
      </RequireAuth>
    )
    await waitFor(() =>
      expect(replaceMock).toHaveBeenCalledWith(
        "/login?redirect=%2Fproject%2Fabc"
      )
    )
    expect(screen.queryByText("secret content")).not.toBeInTheDocument()
  })

  it("hydrates from a stored token and renders children", async () => {
    setToken("tok-1")
    mockMe.mockResolvedValueOnce(USER)
    render(
      <RequireAuth>
        <div>secret content</div>
      </RequireAuth>
    )
    expect(await screen.findByText("secret content")).toBeInTheDocument()
    expect(replaceMock).not.toHaveBeenCalled()
  })

  it("renders children (no redirect) on a transient /me failure when a token is present", async () => {
    // Token survives (apiFetch only clears on 401); a network/5xx blip must not
    // bounce a logged-in user to /login.
    setToken("tok-1")
    mockMe.mockRejectedValueOnce(new Error("network down"))
    render(
      <RequireAuth>
        <div>secret content</div>
      </RequireAuth>
    )
    expect(await screen.findByText("secret content")).toBeInTheDocument()
    expect(replaceMock).not.toHaveBeenCalled()
  })
})
