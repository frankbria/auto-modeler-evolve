"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAppStore } from "@/lib/store"

/**
 * Right-hand side of the global nav: shows the signed-in user's email and a
 * sign-out control. Hydrates the session from a stored token on mount so the
 * nav reflects auth state on every page. Renders nothing for anonymous
 * visitors, keeping public pages (/login, /predict/[id]) uncluttered.
 */
export function AuthNav() {
  const router = useRouter()
  const user = useAppStore((s) => s.user)
  const isAuthenticated = useAppStore((s) => s.isAuthenticated)
  const logout = useAppStore((s) => s.logout)
  const loadCurrentUser = useAppStore((s) => s.loadCurrentUser)

  useEffect(() => {
    if (!useAppStore.getState().isAuthenticated) {
      void loadCurrentUser()
    }
  }, [loadCurrentUser])

  if (!isAuthenticated || !user) return null

  function handleLogout() {
    logout()
    router.push("/login")
  }

  return (
    <div className="ml-auto flex items-center gap-3">
      <span className="text-sm text-muted-foreground" data-testid="auth-user-email">
        {user.email}
      </span>
      <button
        type="button"
        onClick={handleLogout}
        className="text-sm font-medium text-muted-foreground underline-offset-4 transition-colors hover:text-foreground hover:underline"
      >
        Sign out
      </button>
    </div>
  )
}
