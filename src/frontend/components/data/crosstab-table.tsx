"use client"

import type { CrosstabResult } from "@/lib/types"

interface CrosstabTableProps {
  result: CrosstabResult
}

function formatCell(value: number | null, agg: string): string {
  if (value === null || value === undefined) return "—"
  if (agg === "count") return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(value) >= 1_000_000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 })
  if (Number.isInteger(value)) return value.toLocaleString()
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

export function CrosstabTable({ result }: CrosstabTableProps) {
  const { row_col, col_col, value_col, agg_func, col_headers, rows, col_totals, grand_total, summary } = result

  const metricLabel = value_col
    ? `${agg_func.charAt(0).toUpperCase() + agg_func.slice(1)} of ${value_col}`
    : "Count"

  return (
    <div className="mt-2 rounded-lg border bg-card p-3 text-sm" data-testid="crosstab-table">
      <p className="mb-1 text-xs font-semibold text-muted-foreground">
        {metricLabel}: {row_col} × {col_col}
      </p>
      <p className="mb-2 text-xs text-muted-foreground">{summary}</p>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              {/* Corner header */}
              <th className="border border-border bg-muted px-2 py-1 text-left font-semibold text-muted-foreground">
                {row_col} ↓ / {col_col} →
              </th>
              {col_headers.map((header) => (
                <th
                  key={header}
                  className="border border-border bg-muted px-2 py-1 text-right font-semibold"
                  title={header}
                >
                  {header.length > 12 ? header.slice(0, 11) + "…" : header}
                </th>
              ))}
              <th className="border border-border bg-primary/10 px-2 py-1 text-right font-semibold text-primary">
                Total
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr
                key={row.row_label}
                className={rowIdx % 2 === 0 ? "bg-background" : "bg-muted/30"}
              >
                <td className="border border-border px-2 py-1 font-medium" title={row.row_label}>
                  {row.row_label.length > 20 ? row.row_label.slice(0, 19) + "…" : row.row_label}
                </td>
                {row.cells.map((cell, cellIdx) => (
                  <td
                    key={cellIdx}
                    className="border border-border px-2 py-1 text-right tabular-nums"
                  >
                    {formatCell(cell, agg_func)}
                  </td>
                ))}
                <td className="border border-border bg-primary/5 px-2 py-1 text-right font-semibold tabular-nums">
                  {formatCell(row.row_total, agg_func)}
                </td>
              </tr>
            ))}
            {/* Totals row */}
            <tr className="bg-primary/10 font-semibold">
              <td className="border border-border px-2 py-1 text-primary">Total</td>
              {col_totals.map((total, idx) => (
                <td
                  key={idx}
                  className="border border-border px-2 py-1 text-right tabular-nums text-primary"
                >
                  {formatCell(total, agg_func)}
                </td>
              ))}
              <td className="border border-border px-2 py-1 text-right tabular-nums text-primary">
                {formatCell(grand_total, agg_func)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
