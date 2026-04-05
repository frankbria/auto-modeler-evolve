"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { DatasetComparisonResult, NumericDrift, CategoricalDrift } from "@/lib/types"

// ---------------------------------------------------------------------------
// Severity badge
// ---------------------------------------------------------------------------

function SeverityBadge({ severity }: { severity: "low" | "medium" | "high" }) {
  if (severity === "high")
    return <Badge className="bg-red-100 text-red-800 border-red-200 text-xs">High</Badge>
  if (severity === "medium")
    return <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 text-xs">Medium</Badge>
  return <Badge className="bg-gray-100 text-gray-700 border-gray-200 text-xs">Low</Badge>
}

// ---------------------------------------------------------------------------
// Drift score badge
// ---------------------------------------------------------------------------

function DriftScoreBadge({ score }: { score: number }) {
  if (score >= 50)
    return (
      <Badge className="bg-red-100 text-red-800 border-red-200 text-xs">
        Drift score: {score}/100
      </Badge>
    )
  if (score >= 20)
    return (
      <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 text-xs">
        Drift score: {score}/100
      </Badge>
    )
  return (
    <Badge className="bg-green-100 text-green-800 border-green-200 text-xs">
      Drift score: {score}/100
    </Badge>
  )
}

// ---------------------------------------------------------------------------
// Numeric drift row
// ---------------------------------------------------------------------------

function NumericDriftRow({ drift }: { drift: NumericDrift }) {
  const direction = drift.pct_change > 0 ? "↑" : "↓"
  const absChange = Math.abs(drift.pct_change)

  return (
    <tr className="border-b border-border last:border-0">
      <td className="px-2 py-1 font-mono text-xs truncate max-w-[100px]" title={drift.col}>
        {drift.col.length > 14 ? drift.col.slice(0, 14) + "…" : drift.col}
      </td>
      <td className="px-2 py-1 text-xs tabular-nums text-muted-foreground">
        {drift.old_mean.toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </td>
      <td className="px-2 py-1 text-xs tabular-nums text-muted-foreground">
        {drift.new_mean.toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </td>
      <td className="px-2 py-1 text-xs">
        <span className={drift.pct_change >= 0 ? "text-blue-700" : "text-orange-700"}>
          {direction} {absChange.toFixed(1)}%
        </span>
      </td>
      <td className="px-2 py-1">
        <SeverityBadge severity={drift.severity} />
      </td>
    </tr>
  )
}

// ---------------------------------------------------------------------------
// Categorical drift row
// ---------------------------------------------------------------------------

function CategoricalDriftRow({ drift }: { drift: CategoricalDrift }) {
  return (
    <div className="rounded border border-border bg-muted/20 px-3 py-2 space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono font-medium">
          {drift.col.length > 20 ? drift.col.slice(0, 20) + "…" : drift.col}
        </span>
        <SeverityBadge severity={drift.severity} />
      </div>
      {drift.new_categories.length > 0 && (
        <p className="text-xs text-green-700">
          <span className="font-medium">New:</span>{" "}
          {drift.new_categories.slice(0, 5).join(", ")}
          {drift.new_categories.length > 5 && ` +${drift.new_categories.length - 5} more`}
        </p>
      )}
      {drift.dropped_categories.length > 0 && (
        <p className="text-xs text-red-700">
          <span className="font-medium">Dropped:</span>{" "}
          {drift.dropped_categories.slice(0, 5).join(", ")}
          {drift.dropped_categories.length > 5 && ` +${drift.dropped_categories.length - 5} more`}
        </p>
      )}
      {drift.top_shift_pct > 0 && (
        <p className="text-xs text-muted-foreground">
          Largest frequency shift: {drift.top_shift_pct.toFixed(1)} percentage points
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface DatasetComparisonCardProps {
  result: DatasetComparisonResult
}

export function DatasetComparisonCard({ result }: DatasetComparisonCardProps) {
  const rowDelta = result.row_count_change_pct
  const rowDeltaStr =
    rowDelta === 0
      ? "same row count"
      : rowDelta > 0
        ? `+${rowDelta}% more rows`
        : `${rowDelta}% fewer rows`

  const totalIssues =
    result.numeric_drifts.length +
    result.categorical_drifts.length +
    result.new_columns.length +
    result.dropped_columns.length

  return (
    <figure aria-label="Dataset comparison report">
      <Card className="border-orange-300">
        <CardHeader className="pb-2">
          <div className="flex flex-wrap items-center gap-2">
            <CardTitle className="text-sm">
              <span aria-hidden="true">📊</span> Data Comparison
            </CardTitle>
            <DriftScoreBadge score={result.drift_score} />
            {totalIssues > 0 && (
              <Badge className="bg-orange-100 text-orange-800 border-orange-200 text-xs">
                {totalIssues} change{totalIssues !== 1 ? "s" : ""}
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* Summary */}
          <p className="text-xs text-muted-foreground">{result.summary}</p>

          {/* Dataset names + row counts */}
          <div className="rounded border border-border bg-muted/20 px-3 py-2 space-y-1">
            <p className="text-xs">
              <span className="font-medium">Baseline:</span>{" "}
              <span className="font-mono">{result.baseline_name}</span>{" "}
              <span className="text-muted-foreground">
                ({result.row_count_old.toLocaleString()} rows,{" "}
                {result.col_count_old} columns)
              </span>
            </p>
            <p className="text-xs">
              <span className="font-medium">New:</span>{" "}
              <span className="font-mono">{result.new_name}</span>{" "}
              <span className="text-muted-foreground">
                ({result.row_count_new.toLocaleString()} rows,{" "}
                {result.col_count_new} columns)
              </span>
            </p>
            <p className="text-xs text-muted-foreground">{rowDeltaStr}</p>
          </div>

          {/* Schema changes */}
          {(result.new_columns.length > 0 || result.dropped_columns.length > 0) && (
            <div className="space-y-1">
              <p className="text-xs font-medium">Schema changes</p>
              {result.new_columns.length > 0 && (
                <p className="text-xs text-green-700">
                  <span className="font-medium">New columns:</span>{" "}
                  {result.new_columns.join(", ")}
                </p>
              )}
              {result.dropped_columns.length > 0 && (
                <p className="text-xs text-red-700">
                  <span className="font-medium">Dropped columns:</span>{" "}
                  {result.dropped_columns.join(", ")}
                </p>
              )}
            </div>
          )}

          {/* Numeric distribution shifts */}
          {result.numeric_drifts.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-medium">
                Numeric distribution shifts ({result.numeric_drifts.length})
              </p>
              <div className="overflow-x-auto rounded border border-border">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border bg-muted/40">
                      <th className="px-2 py-1 text-left font-medium text-muted-foreground">Column</th>
                      <th className="px-2 py-1 text-left font-medium text-muted-foreground">Old avg</th>
                      <th className="px-2 py-1 text-left font-medium text-muted-foreground">New avg</th>
                      <th className="px-2 py-1 text-left font-medium text-muted-foreground">Change</th>
                      <th className="px-2 py-1 text-left font-medium text-muted-foreground">Severity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.numeric_drifts.map((d) => (
                      <NumericDriftRow key={d.col} drift={d} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Categorical distribution changes */}
          {result.categorical_drifts.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium">
                Categorical changes ({result.categorical_drifts.length})
              </p>
              {result.categorical_drifts.map((d) => (
                <CategoricalDriftRow key={d.col} drift={d} />
              ))}
            </div>
          )}

          {/* No changes */}
          {totalIssues === 0 && (
            <p className="text-xs text-green-700">
              No significant distribution changes detected — datasets look compatible.
            </p>
          )}
        </CardContent>
      </Card>
    </figure>
  )
}
