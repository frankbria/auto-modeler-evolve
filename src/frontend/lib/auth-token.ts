/**
 * Single source of truth for the bearer token.
 *
 * The backend is backend-native auth (JWT). The frontend stores the token in
 * localStorage and forwards it as `Authorization: Bearer <token>` on every
 * management call (see `apiFetch` in lib/api.ts). All access goes through these
 * helpers so storage can change in one place. SSR-safe: returns null on the
 * server where `window` is undefined.
 */

export const AUTH_TOKEN_KEY = "automodeler_auth_token"

export function getToken(): string | null {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage.getItem(AUTH_TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string): void {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(AUTH_TOKEN_KEY, token)
  } catch {
    // Storage unavailable (private mode / quota) — auth simply won't persist.
  }
}

export function clearToken(): void {
  if (typeof window === "undefined") return
  try {
    window.localStorage.removeItem(AUTH_TOKEN_KEY)
  } catch {
    // Nothing to do — token was never persisted.
  }
}
