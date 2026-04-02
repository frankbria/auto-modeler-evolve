"use client"

import { Badge } from "@/components/ui/badge"
import type { SplitStrategyResult } from "@/lib/types"

interface SplitStrategyCardProps {
  result: SplitStrategyResult
}

export function SplitStrategyCard({ result }: SplitStrategyCardProps) {
  const isChronological = result.split_strategy === "chronological"

  return (
    <div
      className={`rounded-lg border p-4 mt-2 ${
        isChronological
          ? "border-sky-300 bg-sky-50"
          : "border-slate-300 bg-slate-50"
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        <span aria-hidden="true" className="text-lg">
          {isChronological ? "🗓️" : "🔀"}
        </span>
        <span className="font-semibold text-sm">Split Strategy Updated</span>
        <Badge
          className={
            isChronological
              ? "bg-sky-100 text-sky-800 border-sky-200"
              : "bg-slate-100 text-slate-700 border-slate-200"
          }
          variant="outline"
        >
          {isChronological ? "Time-based" : "Random"}
        </Badge>
      </div>

      {isChronological && result.date_col && (
        <p className="text-xs text-muted-foreground mb-1">
          Sorting by <code className="bg-muted px-1 rounded">{result.date_col}</code>
        </p>
      )}

      <p className="text-sm text-foreground/80">{result.explanation}</p>

      {isChronological && (
        <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-8 h-2 rounded bg-sky-400" />
            <span>80% train (older)</span>
          </div>
          <span aria-hidden="true">→</span>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded bg-amber-400" />
            <span>20% test (recent)</span>
          </div>
        </div>
      )}
    </div>
  )
}
