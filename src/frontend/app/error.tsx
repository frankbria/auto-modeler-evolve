"use client"

import { ErrorDisplay } from "@/components/ui/error-display"

/**
 * Route-segment error boundary (#23). Catches render/runtime errors below the
 * root layout and shows a friendly recover-able message instead of a blank page
 * or a stack trace — per the UX north star ("fail gracefully").
 */
export default function Error({ reset }: { reset: () => void }) {
  return (
    <div className="p-6">
      <ErrorDisplay
        variant="full-page"
        message="Something went wrong."
        details="The page hit an unexpected error. You can try again, or head back home."
        onRetry={reset}
      />
    </div>
  )
}
