"use client"

import { Badge } from "@/components/ui/badge"
import type { DataExportResult } from "@/lib/types"

interface DataExportCardProps {
  result: DataExportResult
}

export function DataExportCard({ result }: DataExportCardProps) {
  return (
    <div className="rounded-lg border-2 border-indigo-200 bg-card p-4 mt-2">
      <div className="flex items-center gap-2 mb-1">
        <span aria-hidden="true" className="text-indigo-500">⬇</span>
        <span className="font-semibold text-sm">Dataset Export Ready</span>
        {result.filtered && (
          <Badge className="bg-amber-100 text-amber-800 border-amber-300 text-xs">
            Filtered
          </Badge>
        )}
      </div>
      <p className="text-xs text-muted-foreground mb-3">
        {result.filename} &mdash;{" "}
        {result.row_count.toLocaleString()} row{result.row_count !== 1 ? "s" : ""}
        {result.filtered ? " (active filter applied)" : ""}
      </p>
      <a
        href={result.download_url}
        download={result.filename}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Download CSV
      </a>
    </div>
  )
}
