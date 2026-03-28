"use client"

import type { RecordTableResult, RecordTableRow } from "@/lib/types"

interface RecordTableCardProps {
  result: RecordTableResult
}

function formatCellValue(val: number | string | null): string {
  if (val === null || val === undefined) return "—"
  if (typeof val === "string") {
    return val.length > 30 ? val.slice(0, 28) + "…" : val
  }
  if (typeof val === "number") {
    if (!isFinite(val)) return "—"
    if (Number.isInteger(val)) return val.toLocaleString()
    return val.toFixed(3)
  }
  return String(val)
}

function DataRow({ row, columns }: { row: RecordTableRow; columns: string[] }) {
  return (
    <tr className="even:bg-muted/30 hover:bg-muted/50 transition-colors">
      {columns.map((col) => (
        <td
          key={col}
          className="px-3 py-1.5 text-sm text-muted-foreground truncate max-w-[140px]"
          title={String(row[col] ?? "")}
        >
          {formatCellValue(row[col] as number | string | null)}
        </td>
      ))}
    </tr>
  )
}

export function RecordTableCard({ result }: RecordTableCardProps) {
  const { columns, rows, total_rows, filtered_rows, shown_rows, filtered, condition_summary, summary } = result

  return (
    <div className="rounded-lg border border-sky-200 dark:border-sky-800 bg-card overflow-hidden mt-2">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-sky-50/50 dark:bg-sky-900/10">
        <span className="text-base font-semibold text-foreground">Data Preview</span>
        <span className="ml-auto text-xs font-medium px-2 py-0.5 rounded-full bg-sky-100 dark:bg-sky-800 text-sky-700 dark:text-sky-200">
          {columns.length} columns
        </span>
        {filtered && (
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-800 text-amber-700 dark:text-amber-200">
            filtered
          </span>
        )}
      </div>

      {/* Filter condition badge */}
      {filtered && condition_summary && (
        <div className="px-4 py-2 border-b border-border bg-muted/30 text-xs text-muted-foreground">
          Showing rows where:{" "}
          <span className="font-mono text-foreground">{condition_summary}</span>
        </div>
      )}

      {/* Table */}
      {rows.length === 0 ? (
        <div className="px-4 py-6 text-center text-muted-foreground text-sm">
          No rows match the specified condition.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-border bg-muted/40">
                {columns.map((col) => (
                  <th
                    key={col}
                    className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide truncate max-w-[140px]"
                    title={col}
                  >
                    {col.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <DataRow key={i} row={row} columns={columns} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-2 border-t border-border bg-muted/20 text-xs text-muted-foreground">
        {filtered
          ? `${filtered_rows.toLocaleString()} matching rows · showing ${shown_rows} of ${total_rows.toLocaleString()} total`
          : `Showing ${shown_rows} of ${total_rows.toLocaleString()} rows`}
      </div>
    </div>
  )
}
