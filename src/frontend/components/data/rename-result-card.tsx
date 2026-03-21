"use client"

import type { RenameResult } from "@/lib/types"

interface RenameResultCardProps {
  result: RenameResult
}

export function RenameResultCard({ result }: RenameResultCardProps) {
  return (
    <div
      className="mt-2 rounded-lg border bg-card p-3"
      data-testid="rename-result-card"
    >
      <p className="mb-1.5 text-xs font-semibold text-muted-foreground">
        Column Renamed
      </p>
      <div className="flex items-center gap-2 text-sm">
        <span className="rounded bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground line-through">
          {result.old_name}
        </span>
        <span className="text-muted-foreground">→</span>
        <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-xs font-semibold text-primary">
          {result.new_name}
        </span>
      </div>
      <p className="mt-1.5 text-xs text-muted-foreground">
        Dataset updated · {result.column_count} columns
      </p>
    </div>
  )
}
