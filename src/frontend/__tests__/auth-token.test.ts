/**
 * Unit tests for the auth token helper (lib/auth-token.ts).
 *
 * The token is persisted in localStorage so it survives reloads. The helper
 * is the single source of truth for reading/writing the bearer token.
 */

import { getToken, setToken, clearToken, AUTH_TOKEN_KEY } from "../lib/auth-token"

beforeEach(() => {
  window.localStorage.clear()
})

describe("auth-token helper", () => {
  it("returns null when no token is stored", () => {
    expect(getToken()).toBeNull()
  })

  it("persists a token to localStorage and reads it back", () => {
    setToken("abc.def.ghi")
    expect(window.localStorage.getItem(AUTH_TOKEN_KEY)).toBe("abc.def.ghi")
    expect(getToken()).toBe("abc.def.ghi")
  })

  it("clearToken removes the stored token", () => {
    setToken("abc.def.ghi")
    clearToken()
    expect(getToken()).toBeNull()
    expect(window.localStorage.getItem(AUTH_TOKEN_KEY)).toBeNull()
  })
})
