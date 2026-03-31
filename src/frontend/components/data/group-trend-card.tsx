"use client"

import { Badge } from "@/components/ui/badge"
import type { GroupTrendResult, GroupTrendRow } from "@/lib/types"

interface GroupTrendCardProps {
  result: GroupTrendResult
}

function DirectionIcon({ direction }: { direction: GroupTrendRow["direction"] }) {
  if (direction === "up")
    return (
      <span aria-hidden="true" className="text-emerald-600 font-bold">
        ▲
      </span>
    )
  if (direction === "down")
    return (
      <span aria-hidden="true" className="text-rose-600 font-bold">
        ▼
      </span>
    )
  return (
    <span aria-hidden="true" className="text-muted-foreground">
      →
    </span>
  )
}

function PctBadge({ pct, direction }: { pct: number; direction: GroupTrendRow["direction"] }) {
  const sign = pct >= 0 ? "+" : ""
  const colorClass =
    direction === "up"
      ? "bg-emerald-100 text-emerald-800 border-emerald-200"
      : direction === "down"
        ? "bg-rose-100 text-rose-800 border-rose-200"
        : "bg-muted text-muted-foreground border-border"
  return (
    <span className={`text-xs font-mono border rounded px-1.5 py-0.5 ${colorClass}`}>
      {sign}
      {pct.toFixed(1)}%
    </span>
  )
}

function formatVal(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return v % 1 === 0 ? String(v) : v.toFixed(2)
}

export function GroupTrendCard({ result }: GroupTrendCardProps) {
  return (
    <div
      className="rounded-lg border-2 border-orange-300 bg-card p-4 mt-2"
      role="region"
      aria-label={`Group trends: ${result.value_col} by ${result.group_col}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-base font-bold text-orange-500">
          📈
        </span>
        <span className="font-semibold text-sm">
          Trend:{" "}
          <span className="font-mono">{result.value_col.replace(/_/g, " ")}</span>
          {" by "}
          <span className="font-mono">{result.group_col.replace(/_/g, " ")}</span>
        </span>
        {result.rising > 0 && (
          <Badge className="bg-emerald-100 text-emerald-800 border border-emerald-200 text-xs">
            ▲ {result.rising} rising
          </Badge>
        )}
        {result.falling > 0 && (
          <Badge className="bg-rose-100 text-rose-800 border border-rose-200 text-xs">
            ▼ {result.falling} falling
          </Badge>
        )}
        {result.flat > 0 && (
          <Badge variant="secondary" className="text-xs">
            → {result.flat} flat
          </Badge>
        )}
      </div>

      {/* Group table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="text-xs text-muted-foreground border-b border-border">
              <th className="text-left py-1 pr-3 font-medium">#</th>
              <th className="text-left py-1 pr-3 font-medium">
                {result.group_col.replace(/_/g, " ")}
              </th>
              <th className="text-right py-1 pr-3 font-medium">First</th>
              <th className="text-right py-1 pr-3 font-medium">Last</th>
              <th className="text-right py-1 pr-3 font-medium">Change</th>
              <th className="text-center py-1 font-medium">Trend</th>
            </tr>
          </thead>
          <tbody>
            {result.groups.map((row) => (
              <tr
                key={row.group}
                className={`border-b border-border/40 last:border-0 ${
                  row.direction === "up"
                    ? "hover:bg-emerald-50/50"
                    : row.direction === "down"
                      ? "hover:bg-rose-50/50"
                      : "hover:bg-muted/30"
                }`}
              >
                <td className="py-1.5 pr-3 text-xs text-muted-foreground tabular-nums">
                  {row.rank}
                </td>
                <td className="py-1.5 pr-3 font-medium text-foreground truncate max-w-[120px]">
                  {row.group}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums text-muted-foreground text-xs">
                  {formatVal(row.first_value)}
                </td>
                <td className="py-1.5 pr-3 text-right tabular-nums font-medium">
                  {formatVal(row.last_value)}
                </td>
                <td className="py-1.5 pr-3 text-right">
                  <PctBadge pct={row.pct_change} direction={row.direction} />
                </td>
                <td className="py-1.5 text-center">
                  <DirectionIcon direction={row.direction} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary footer */}
      <p className="text-xs text-muted-foreground border-t border-border/50 pt-2 mt-2">
        {result.summary}
      </p>
    </div>
  )
}
