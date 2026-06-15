"use client"

import { useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import { useAppStore } from "@/lib/store"

/**
 * Client-side route guard for owner-scoped pages (home, project workspace).
 *
 * Backend auth is the real enforcement boundary (every management endpoint
 * returns 401 without a valid token); this guard is the UX layer that keeps
 * unauthenticated users from seeing a broken, request-failing page. On mount it
 * gives a stored token a chance to hydrate the session, then either renders the
 * protected content or redirects to /login with a redirect back to this path.
 */
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const pathname = usePathname()
  const loadCurrentUser = useAppStore((s) => s.loadCurrentUser)
  const isAuthenticated = useAppStore((s) => s.isAuthenticated)
  const [checking, setChecking] = useState(!isAuthenticated)

  useEffect(() => {
    let active = true
    async function check() {
      if (useAppStore.getState().isAuthenticated) {
        setChecking(false)
        return
      }
      await loadCurrentUser()
      if (!active) return
      if (useAppStore.getState().isAuthenticated) {
        setChecking(false)
      } else {
        router.replace(`/login?redirect=${encodeURIComponent(pathname)}`)
      }
    }
    check()
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!isAuthenticated && checking) {
    return (
      <div
        className="flex min-h-[calc(100vh-3rem)] items-center justify-center text-sm text-muted-foreground"
        role="status"
      >
        Loading…
      </div>
    )
  }

  if (!isAuthenticated) return null

  return <>{children}</>
}
