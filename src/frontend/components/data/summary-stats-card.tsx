"use client"

import { Badge } from "@/components/ui/badge"
import type { SummaryStatsResult, NumericColumnStats, CategoricalColumnStats } from "@/lib/types"

interface SummaryStatsCardProps {
  result: SummaryStatsResult
}

function fmtNum(val: number | null): string {
  if (val === null || val === undefined) return "—"
  if (Math.abs(val) >= 1_000_000) return (val / 1_000_000).toFixed(2) + "M"
  if (Math.abs(val) >= 1_000) return (val / 1_000).toFixed(1) + "k"
  return Number.isInteger(val) ? String(val) : val.toFixed(3)
}

function NumericRow({ stat }: { stat: NumericColumnStats }) {
  return (
    <tr className="border-t border-border/50 hover:bg-muted/30">
      <td
        className="py-1.5 pr-2 text-xs font-mono truncate max-w-[100px]"
        title={stat.column}
      >
        {stat.column.replace(/_/g, " ")}
      </td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{stat.count}</td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{fmtNum(stat.mean)}</td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{fmtNum(stat.std)}</td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{fmtNum(stat.min)}</td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{fmtNum(stat.median)}</td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{fmtNum(stat.max)}</td>
      <td className="py-1.5 text-xs text-center">
        {stat.null_count > 0 ? (
          <span className="text-rose-600">{stat.null_count}</span>
        ) : (
          <span className="text-emerald-600">0</span>
        )}
      </td>
    </tr>
  )
}

function CategoricalRow({ stat }: { stat: CategoricalColumnStats }) {
  return (
    <tr className="border-t border-border/50 hover:bg-muted/30">
      <td
        className="py-1.5 pr-2 text-xs font-mono truncate max-w-[100px]"
        title={stat.column}
      >
        {stat.column.replace(/_/g, " ")}
      </td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{stat.count}</td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{stat.unique}</td>
      <td className="py-1.5 pr-2 text-xs text-center truncate max-w-[80px]" title={stat.top ?? ""}>
        {stat.top ? (
          <span className="font-medium">&ldquo;{stat.top}&rdquo;</span>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </td>
      <td className="py-1.5 pr-2 text-xs text-center tabular-nums">{stat.freq}</td>
      <td className="py-1.5 text-xs text-center">
        {stat.null_count > 0 ? (
          <span className="text-rose-600">{stat.null_count}</span>
        ) : (
          <span className="text-emerald-600">0</span>
        )}
      </td>
    </tr>
  )
}

export function SummaryStatsCard({ result }: SummaryStatsCardProps) {
  const hasNumeric = result.numeric_stats.length > 0
  const hasCategorical = result.categorical_stats.length > 0

  return (
    <div
      className="rounded-lg border-2 border-slate-300 bg-card p-4 mt-2"
      role="region"
      aria-label="Dataset summary statistics"
    >
      <div className="flex items-center gap-2 mb-1">
        <span aria-hidden="true" className="text-slate-500">≡</span>
        <span className="font-semibold text-sm">Summary Statistics</span>
        <Badge variant="secondary" className="text-xs">
          {result.total_rows.toLocaleString()} rows
        </Badge>
        <Badge variant="outline" className="text-xs">
          {result.total_cols} columns
        </Badge>
      </div>
      <p className="text-xs text-muted-foreground mb-3">{result.summary}</p>

      {hasNumeric && (
        <div className="mb-4">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
            Numeric Columns
          </p>
          <div className="overflow-x-auto">
            <table
              className="w-full text-sm"
              aria-label="Numeric column statistics"
            >
              <thead>
                <tr className="text-left">
                  {["Column", "Count", "Mean", "Std", "Min", "Median", "Max", "Nulls"].map(
                    (h) => (
                      <th
                        key={h}
                        className="pb-1 text-xs text-muted-foreground font-medium pr-2 text-center first:text-left"
                      >
                        {h}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {result.numeric_stats.map((stat) => (
                  <NumericRow key={stat.column} stat={stat} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {hasCategorical && (
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
            Categorical Columns
          </p>
          <div className="overflow-x-auto">
            <table
              className="w-full text-sm"
              aria-label="Categorical column statistics"
            >
              <thead>
                <tr className="text-left">
                  {["Column", "Count", "Unique", "Most Common", "Freq", "Nulls"].map(
                    (h) => (
                      <th
                        key={h}
                        className="pb-1 text-xs text-muted-foreground font-medium pr-2 text-center first:text-left"
                      >
                        {h}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {result.categorical_stats.map((stat) => (
                  <CategoricalRow key={stat.column} stat={stat} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!hasNumeric && !hasCategorical && (
        <p className="text-xs text-muted-foreground text-center py-2">
          No column statistics available.
        </p>
      )}
    </div>
  )
}
