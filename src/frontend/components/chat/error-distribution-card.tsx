"use client"

import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { Badge } from "@/components/ui/badge"
import type {
  ErrorDistributionClassRow,
  ErrorDistributionResult,
} from "@/lib/types"

interface Props {
  result: ErrorDistributionResult
}

function BiasLabel({ label }: { label?: string }) {
  if (!label || label === "unbiased") {
    return (
      <Badge className="bg-emerald-100 text-emerald-800 border border-emerald-200">
        Unbiased
      </Badge>
    )
  }
  if (label === "over-predicts") {
    return (
      <Badge className="bg-rose-100 text-rose-800 border border-rose-200">
        Tends to over-predict
      </Badge>
    )
  }
  return (
    <Badge className="bg-amber-100 text-amber-800 border border-amber-200">
      Tends to under-predict
    </Badge>
  )
}

function ClassErrorRow({ row, maxRate }: { row: ErrorDistributionClassRow; maxRate: number }) {
  const barPct = maxRate > 0 ? (row.error_rate / maxRate) * 100 : 0
  const color =
    row.error_pct >= 30
      ? "bg-rose-500"
      : row.error_pct >= 15
        ? "bg-amber-500"
        : "bg-emerald-500"

  return (
    <tr className="border-b border-border/50 last:border-0">
      <td className="py-1.5 pr-3 font-mono text-sm text-foreground">{row.class}</td>
      <td className="py-1.5 pr-3 text-sm text-muted-foreground text-right">{row.total}</td>
      <td className="py-1.5 pr-3 text-sm text-right">
        <span
          className={
            row.error_pct >= 30
              ? "text-rose-700 font-medium"
              : row.error_pct >= 15
                ? "text-amber-700"
                : "text-emerald-700"
          }
        >
          {row.error_pct.toFixed(1)}%
        </span>
      </td>
      <td className="py-1.5 w-32">
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${color} transition-all`}
            style={{ width: `${barPct}%` }}
            aria-label={`${row.error_pct.toFixed(1)}% error rate`}
          />
        </div>
      </td>
    </tr>
  )
}

export function ErrorDistributionCard({ result }: Props) {
  const { problem_type, bins, class_breakdown, stats, summary, algorithm, target_col } = result

  const algName = algorithm
    ? algorithm
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : "Model"

  const maxClassRate =
    class_breakdown && class_breakdown.length > 0
      ? Math.max(...class_breakdown.map((r) => r.error_rate))
      : 0

  // Color histogram bars by position relative to zero for regression
  const midBin = bins.length > 0 ? bins[Math.floor(bins.length / 2)] : null
  const histData = bins.map((b) => ({
    name: b.label,
    count: b.count,
    pct: b.pct,
    // Negative residual bins are rose, positive amber, near-zero emerald
    color: midBin
      ? b.hi <= 0
        ? "#f87171" // rose-400
        : b.lo >= 0
          ? "#fb923c" // orange-400
          : "#34d399" // emerald-400
      : "#60a5fa", // blue-400 fallback
  }))

  return (
    <figure
      className="rounded-lg border border-indigo-300 bg-indigo-50 p-4 text-sm my-2"
      aria-label={`Prediction error distribution for ${target_col ?? "target"}`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-base" aria-hidden="true">
          📊
        </span>
        <span className="font-semibold text-indigo-900">Error Distribution</span>
        <Badge className="bg-indigo-100 text-indigo-800 border border-indigo-200">
          {algName}
        </Badge>
        {target_col && (
          <Badge className="bg-slate-100 text-slate-700 border border-slate-200">
            Target: {target_col}
          </Badge>
        )}
        <Badge className="bg-slate-100 text-slate-700 border border-slate-200 capitalize">
          {problem_type}
        </Badge>
      </div>

      {/* Regression: residual histogram */}
      {problem_type === "regression" && bins.length > 0 && (
        <>
          <div className="flex flex-wrap gap-4 mb-3">
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Mean residual</p>
              <p className="font-semibold text-foreground">{stats.mean?.toFixed(3) ?? "—"}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Std</p>
              <p className="font-semibold text-foreground">{stats.std?.toFixed(3) ?? "—"}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">MAE</p>
              <p className="font-semibold text-foreground">{stats.mae?.toFixed(3) ?? "—"}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Within ±1 std</p>
              <p className="font-semibold text-foreground">
                {stats.within_1std_pct?.toFixed(1) ?? "—"}%
              </p>
            </div>
            <div className="flex items-center">
              <BiasLabel label={stats.bias_label} />
            </div>
          </div>

          <div className="h-40 w-full mb-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histData} margin={{ top: 4, right: 4, bottom: 20, left: 4 }}>
                <XAxis
                  dataKey="name"
                  tick={false}
                  label={{
                    value: "Residual (actual − predicted)",
                    position: "insideBottom",
                    offset: -8,
                    fontSize: 11,
                    fill: "#6b7280",
                  }}
                />
                <YAxis
                  tick={{ fontSize: 10 }}
                  label={{
                    value: "Count",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 11,
                    fill: "#6b7280",
                  }}
                />
                <Tooltip
                  formatter={(value) => [value, "rows"]}
                  labelFormatter={(label) => `Residual: ${label}`}
                />
                <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                  {histData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-muted-foreground text-center mb-2">
            Negative = under-predicted &nbsp;·&nbsp; Positive = over-predicted
          </p>
        </>
      )}

      {/* Classification: per-class error rates */}
      {problem_type === "classification" && class_breakdown && class_breakdown.length > 0 && (
        <div className="mb-3">
          <div className="flex flex-wrap gap-4 mb-2">
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Overall accuracy</p>
              <p className="font-semibold text-foreground">
                {stats.overall_accuracy !== undefined
                  ? `${(stats.overall_accuracy * 100).toFixed(1)}%`
                  : "—"}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Total wrong</p>
              <p className="font-semibold text-foreground">{stats.total_wrong ?? "—"}</p>
            </div>
            <div className="text-center">
              <p className="text-xs text-muted-foreground">Classes</p>
              <p className="font-semibold text-foreground">{stats.n_classes ?? "—"}</p>
            </div>
          </div>

          <table className="w-full text-left" role="table">
            <thead>
              <tr className="border-b border-border">
                <th className="py-1 pr-3 text-xs font-medium text-muted-foreground">Class</th>
                <th className="py-1 pr-3 text-xs font-medium text-muted-foreground text-right">
                  Total
                </th>
                <th className="py-1 pr-3 text-xs font-medium text-muted-foreground text-right">
                  Error rate
                </th>
                <th className="py-1 text-xs font-medium text-muted-foreground">Errors</th>
              </tr>
            </thead>
            <tbody>
              {class_breakdown.map((row) => (
                <ClassErrorRow key={row.class} row={row} maxRate={maxClassRate} />
              ))}
            </tbody>
          </table>
          <p className="text-xs text-muted-foreground mt-1">Sorted highest error rate first</p>
        </div>
      )}

      {/* Summary */}
      <figcaption className="text-xs text-muted-foreground italic border-t border-indigo-200 pt-2 mt-2">
        {summary}
      </figcaption>
    </figure>
  )
}
