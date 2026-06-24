"use client"

import { Button } from "@/components/ui/button"

interface ErrorDisplayProps {
  /** Plain-English error message (no stack traces — see UX north star). */
  message: string
  /** Optional secondary detail (e.g. the backend's `detail` field). */
  details?: string
  /** When provided, renders a "Try again" button. */
  onRetry?: () => void
  /** Full-page centered layout vs. inline within a content area. */
  variant?: "inline" | "full-page"
}

/**
 * Surfaces an API/load failure to the user with a retry action (#17), replacing
 * silent `.catch()` swallowing. Business-analyst-friendly: a clear message and a
 * next step, never a stack trace.
 */
export function ErrorDisplay({
  message,
  details,
  onRetry,
  variant = "inline",
}: ErrorDisplayProps) {
  return (
    <div
      role="alert"
      data-testid="error-display"
      className={
        variant === "full-page"
          ? "flex min-h-[40vh] flex-col items-center justify-center gap-3 text-center"
          : "flex flex-col items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-6"
      }
    >
      <div className="flex items-center gap-2">
        <span aria-hidden="true" className="text-xl leading-none">
          ⚠️
        </span>
        <p className="text-sm font-medium text-foreground" data-testid="error-display-message">
          {message}
        </p>
      </div>
      {details && (
        <p className="text-xs text-muted-foreground" data-testid="error-display-details">
          {details}
        </p>
      )}
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry} data-testid="error-display-retry">
          Try again
        </Button>
      )}
    </div>
  )
}
