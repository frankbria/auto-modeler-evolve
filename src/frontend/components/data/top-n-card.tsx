"use client"

import type { TopNResult, TopNRow } from "@/lib/types"

interface TopNCardProps {
  result: TopNResult
}

function formatCellValue(val: number | string | null): string {
  if (val === null || val === undefined) return "—"
  if (typeof val === "string") return val
  if (typeof val === "number") {
    if (Math.abs(val) >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`
    if (Math.abs(val) >= 1_000) return `${(val / 1_000).toFixed(1)}k`
    if (Number.isInteger(val)) return val.toLocaleString()
    return val.toFixed(2)
  }
  return String(val)
}

function RankMedal({ rank }: { rank: number }) {
  if (rank === 1) return <span aria-label="1st place">🥇</span>
  if (rank === 2) return <span aria-label="2nd place">🥈</span>
  if (rank === 3) return <span aria-label="3rd place">🥉</span>
  return <span className="text-muted-foreground font-mono text-xs">{rank}</span>
}

function DataRow({
  row,
  displayCols,
  sortCol,
  isTop3,
}: {
  row: TopNRow
  displayCols: string[]
  sortCol: string
  isTop3: boolean
}) {
  return (
    <tr className={isTop3 ? "bg-amber-50/60 dark:bg-amber-900/10" : "even:bg-muted/30"}>
      <td className="px-3 py-2 text-center w-8">
        <RankMedal rank={row._rank} />
      </td>
      {displayCols.map((col) => (
        <td
          key={col}
          className={`px-3 py-2 text-sm truncate max-w-[160px] ${
            col === sortCol ? "font-semibold text-foreground" : "text-muted-foreground"
          }`}
          title={String(row[col] ?? "")}
        >
          {formatCellValue(row[col] as number | string | null)}
        </td>
      ))}
    </tr>
  )
}

export function TopNCard({ result }: TopNCardProps) {
  const { sort_col, direction, n_returned, total_rows, display_cols, rows, summary } = result
  const colLabel = sort_col.replace(/_/g, " ")
  const directionLabel = direction === "top" ? "Highest" : "Lowest"
  const borderColor = direction === "top" ? "border-emerald-500" : "border-rose-500"
  const headerBg = direction === "top" ? "bg-emerald-50 dark:bg-emerald-900/20" : "bg-rose-50 dark:bg-rose-900/20"
  const headerText = direction === "top" ? "text-emerald-800 dark:text-emerald-200" : "text-rose-800 dark:text-rose-200"

  return (
    <div className={`rounded-lg border-2 ${borderColor} overflow-hidden my-2`}>
      {/* Header */}
      <div className={`px-4 py-2 ${headerBg} flex items-center justify-between`}>
        <span className={`font-semibold text-sm ${headerText}`}>
          {directionLabel} {n_returned} by {colLabel}
        </span>
        <span className="text-xs text-muted-foreground">
          {n_returned} of {total_rows} rows
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="px-3 py-2 text-center text-xs text-muted-foreground font-medium w-8">#</th>
              {display_cols.map((col) => (
                <th
                  key={col}
                  className={`px-3 py-2 text-left text-xs font-medium truncate max-w-[160px] ${
                    col === sort_col
                      ? "text-foreground underline decoration-dotted"
                      : "text-muted-foreground"
                  }`}
                >
                  {col.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <DataRow
                key={row._rank}
                row={row}
                displayCols={display_cols}
                sortCol={sort_col}
                isTop3={row._rank <= 3}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary footer */}
      <div className="px-4 py-2 border-t border-border bg-muted/20">
        <p className="text-xs text-muted-foreground">{summary}</p>
      </div>
    </div>
  )
}
