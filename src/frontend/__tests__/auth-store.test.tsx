/**
 * Unit tests for the auth slice of the Zustand store (lib/store.ts).
 *
 * Covers login/register (persist token + user), logout (clear everything),
 * and hydrating the session from a stored token on app load.
 */

import { act } from "@testing-library/react"
import { useAppStore } from "../lib/store"
import { api } from "../lib/api"
import * as tokenStore from "../lib/auth-token"

jest.mock("../lib/api", () => ({
  api: {
    auth: {
      login: jest.fn(),
      register: jest.fn(),
      me: jest.fn(),
    },
  },
}))

const mockLogin = api.auth.login as jest.MockedFunction<typeof api.auth.login>
const mockRegister = api.auth.register as jest.MockedFunction<typeof api.auth.register>
const mockMe = api.auth.me as jest.MockedFunction<typeof api.auth.me>

const USER = { id: "u1", email: "a@b.com", name: "Ada" }
const AUTH = { access_token: "tok-1", token_type: "bearer", user: USER }

beforeEach(() => {
  jest.clearAllMocks()
  window.localStorage.clear()
  act(() => {
    useAppStore.getState().logout()
  })
})

describe("auth store slice", () => {
  it("starts unauthenticated", () => {
    const s = useAppStore.getState()
    expect(s.user).toBeNull()
    expect(s.token).toBeNull()
    expect(s.isAuthenticated).toBe(false)
  })

  it("login persists the token and sets the user", async () => {
    mockLogin.mockResolvedValueOnce(AUTH)
    await act(async () => {
      await useAppStore.getState().login("a@b.com", "s3cret-pw")
    })
    const s = useAppStore.getState()
    expect(mockLogin).toHaveBeenCalledWith("a@b.com", "s3cret-pw")
    expect(s.user).toEqual(USER)
    expect(s.token).toBe("tok-1")
    expect(s.isAuthenticated).toBe(true)
    expect(tokenStore.getToken()).toBe("tok-1")
  })

  it("register persists the token and sets the user", async () => {
    mockRegister.mockResolvedValueOnce(AUTH)
    await act(async () => {
      await useAppStore.getState().register("a@b.com", "s3cret-pw", "Ada")
    })
    const s = useAppStore.getState()
    expect(mockRegister).toHaveBeenCalledWith("a@b.com", "s3cret-pw", "Ada")
    expect(s.isAuthenticated).toBe(true)
    expect(tokenStore.getToken()).toBe("tok-1")
  })

  it("logout clears user, token and storage", async () => {
    mockLogin.mockResolvedValueOnce(AUTH)
    await act(async () => {
      await useAppStore.getState().login("a@b.com", "s3cret-pw")
    })
    act(() => {
      useAppStore.getState().logout()
    })
    const s = useAppStore.getState()
    expect(s.user).toBeNull()
    expect(s.token).toBeNull()
    expect(s.isAuthenticated).toBe(false)
    expect(tokenStore.getToken()).toBeNull()
  })

  it("loadCurrentUser hydrates the session from a stored token", async () => {
    tokenStore.setToken("tok-1")
    mockMe.mockResolvedValueOnce(USER)
    await act(async () => {
      await useAppStore.getState().loadCurrentUser()
    })
    const s = useAppStore.getState()
    expect(mockMe).toHaveBeenCalled()
    expect(s.user).toEqual(USER)
    expect(s.isAuthenticated).toBe(true)
  })

  it("loadCurrentUser is a no-op when there is no stored token", async () => {
    await act(async () => {
      await useAppStore.getState().loadCurrentUser()
    })
    expect(mockMe).not.toHaveBeenCalled()
    expect(useAppStore.getState().isAuthenticated).toBe(false)
  })

  it("loadCurrentUser keeps the session on a transient failure (token survives)", async () => {
    // Simulate an established session, then a network/5xx failure where the
    // token is NOT cleared (apiFetch only clears on 401).
    tokenStore.setToken("tok-1")
    useAppStore.setState({ user: USER, token: "tok-1", isAuthenticated: true })
    mockMe.mockRejectedValueOnce(new Error("network down"))
    await act(async () => {
      await useAppStore.getState().loadCurrentUser()
    })
    // No spurious logout.
    expect(useAppStore.getState().isAuthenticated).toBe(true)
    expect(tokenStore.getToken()).toBe("tok-1")
  })

  it("loadCurrentUser logs out when the token was cleared (401)", async () => {
    tokenStore.setToken("tok-1")
    useAppStore.setState({ user: USER, token: "tok-1", isAuthenticated: true })
    // A 401 path: apiFetch would have cleared the token before me() rejects.
    mockMe.mockImplementationOnce(async () => {
      tokenStore.clearToken()
      throw new Error("Not authenticated")
    })
    await act(async () => {
      await useAppStore.getState().loadCurrentUser()
    })
    expect(useAppStore.getState().isAuthenticated).toBe(false)
    expect(useAppStore.getState().user).toBeNull()
  })
})
