"use client"

import { useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import { useAppStore } from "@/lib/store"
import { getToken } from "@/lib/auth-token"

/**
 * Client-side route guard for owner-scoped pages (home, project workspace).
 *
 * Backend auth is the real enforcement boundary (every management endpoint
 * returns 401 without a valid token); this guard is the UX layer that keeps
 * unauthenticated users from seeing a broken, request-failing page. On mount it
 * gives a stored token a chance to hydrate the session, then either renders the
 * protected content or redirects to /login.
 *
 * We allow rendering whenever a token is present — not only when hydration
 * succeeded — so a transient `/api/auth/me` failure (network / 5xx) doesn't
 * bounce a logged-in user to /login. A genuinely invalid token is cleared by
 * `apiFetch` on the first 401 (here or from the page's own calls), which flips
 * `getToken()` to null and triggers the redirect.
 */
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const pathname = usePathname()
  const loadCurrentUser = useAppStore((s) => s.loadCurrentUser)
  const isAuthenticated = useAppStore((s) => s.isAuthenticated)
  // null = still deciding, true = render, false = redirecting
  const [allowed, setAllowed] = useState<boolean | null>(
    isAuthenticated ? true : null
  )

  useEffect(() => {
    let active = true
    async function check() {
      if (useAppStore.getState().isAuthenticated) {
        setAllowed(true)
        return
      }
      await loadCurrentUser()
      if (!active) return
      if (getToken()) {
        // Authenticated, or a token survived a transient probe failure.
        setAllowed(true)
      } else {
        setAllowed(false)
        router.replace(`/login?redirect=${encodeURIComponent(pathname)}`)
      }
    }
    check()
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (allowed === null) {
    return (
      <div
        className="flex min-h-[calc(100vh-3rem)] items-center justify-center text-sm text-muted-foreground"
        role="status"
      >
        Loading…
      </div>
    )
  }

  if (!allowed) return null

  return <>{children}</>
}
