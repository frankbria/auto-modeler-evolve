"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type {
  DeploymentPredictionComparisonResult,
  DeploymentPredictionDistStats,
  DeploymentPredictionClassShift,
} from "@/lib/types"

interface Props {
  result: DeploymentPredictionComparisonResult
}

function fmt(v: number): string {
  if (Math.abs(v) >= 10000) return v.toFixed(0)
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(2)
  return v.toFixed(4)
}

function StatGrid({ stats, label }: { stats: DeploymentPredictionDistStats; label: string }) {
  return (
    <div className="space-y-1">
      <p className="text-xs font-semibold text-foreground">{label}</p>
      <div className="grid grid-cols-3 gap-1 text-xs">
        {(
          [
            ["Mean", stats.mean],
            ["Median", stats.median],
            ["Std", stats.std],
            ["Min", stats.min],
            ["P25", stats.p25],
            ["P75", stats.p75],
            ["Max", stats.max],
          ] as [string, number][]
        ).map(([k, v]) => (
          <div key={k} className="flex flex-col bg-muted rounded p-1">
            <span className="text-muted-foreground leading-tight">{k}</span>
            <span className="font-mono font-medium text-foreground leading-tight">{fmt(v)}</span>
          </div>
        ))}
        <div className="flex flex-col bg-muted rounded p-1">
          <span className="text-muted-foreground leading-tight">N</span>
          <span className="font-mono font-medium text-foreground leading-tight">{stats.n}</span>
        </div>
      </div>
    </div>
  )
}

function ClassShiftRow({ shift }: { shift: DeploymentPredictionClassShift }) {
  const dir = shift.shift_pct > 0 ? "↑" : shift.shift_pct < 0 ? "↓" : "→"
  const colorClass =
    Math.abs(shift.shift_pct) >= 10
      ? shift.shift_pct > 0
        ? "text-emerald-600"
        : "text-rose-600"
      : "text-muted-foreground"
  return (
    <tr className="border-b last:border-0">
      <td className="py-1.5 pr-3 text-xs font-medium text-foreground">{shift.class_label}</td>
      <td className="py-1.5 pr-3 text-xs font-mono text-muted-foreground">{shift.baseline_pct}%</td>
      <td className="py-1.5 pr-3 text-xs font-mono text-foreground">{shift.current_pct}%</td>
      <td className={`py-1.5 text-xs font-medium whitespace-nowrap ${colorClass}`}>
        <span aria-hidden="true">{dir}</span>{" "}
        {shift.shift_pct > 0 ? "+" : ""}
        {shift.shift_pct.toFixed(1)}pp
      </td>
    </tr>
  )
}

export function DeploymentPredictionDistributionCard({ result }: Props) {
  const isNoData = result.verdict === "no_data"

  const borderClass = isNoData
    ? "border-slate-300"
    : result.verdict === "current_higher"
    ? "border-emerald-500/40"
    : result.verdict === "current_lower"
    ? "border-rose-500/40"
    : result.verdict === "distribution_shifted"
    ? "border-amber-500/40"
    : "border-sky-500/40"

  const verdictBadgeClass = isNoData
    ? "bg-slate-100 text-slate-700 border-slate-200"
    : result.verdict === "current_higher"
    ? "bg-emerald-100 text-emerald-800 border-emerald-200"
    : result.verdict === "current_lower"
    ? "bg-rose-100 text-rose-800 border-rose-200"
    : result.verdict === "distribution_shifted"
    ? "bg-amber-100 text-amber-800 border-amber-200"
    : "bg-sky-100 text-sky-800 border-sky-200"

  return (
    <Card
      data-testid="deploy-pred-dist-compare-card"
      role="region"
      aria-label="Deployment prediction distribution comparison"
      className={borderClass}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <span aria-hidden="true">📊</span> Deployment Prediction Comparison
          </CardTitle>
          <div className="flex gap-1.5 flex-wrap">
            {!isNoData && (
              <Badge className={`${verdictBadgeClass} text-xs`}>{result.verdict_label}</Badge>
            )}
            <Badge className="bg-slate-100 text-slate-700 border-slate-200 text-xs">
              {result.problem_type === "regression" ? "Regression" : "Classification"}
            </Badge>
            {!isNoData && (
              <Badge className="bg-slate-100 text-slate-700 border-slate-200 text-xs">
                {result.n_baseline} baseline / {result.n_current} current
              </Badge>
            )}
          </div>
        </div>

        {result.current_algorithm && result.baseline_algorithm && (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Baseline:</span>{" "}
            {result.baseline_algorithm}
            {" → "}
            <span className="font-medium text-foreground">Current:</span>{" "}
            {result.current_algorithm}
            {result.target_col && (
              <>
                {" "}
                <span className="text-muted-foreground">(target: {result.target_col})</span>
              </>
            )}
          </p>
        )}
      </CardHeader>

      <CardContent className="space-y-3">
        {isNoData ? (
          <p className="text-xs text-muted-foreground">{result.summary}</p>
        ) : result.problem_type === "regression" &&
          result.baseline_stats &&
          result.current_stats ? (
          <>
            {/* Mean shift highlight */}
            {result.mean_shift_pct !== undefined && (
              <div className={`rounded-lg p-2 ${verdictBadgeClass.replace("text-", "bg-").replace("100 text-", "50 text-")}`}>
                <p className="text-xs font-medium">
                  Mean shift:{" "}
                  <span className="font-mono">
                    {result.mean_shift_pct > 0 ? "+" : ""}
                    {result.mean_shift_pct.toFixed(1)}%
                  </span>
                  {" "}({fmt(result.baseline_stats.mean)} → {fmt(result.current_stats.mean)})
                </p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <StatGrid stats={result.baseline_stats} label="Previous deployment" />
              <StatGrid stats={result.current_stats} label="Current deployment" />
            </div>
          </>
        ) : result.problem_type === "classification" && result.class_shifts ? (
          <div className="overflow-x-auto">
            <table className="w-full" aria-label="Class distribution comparison">
              <thead>
                <tr className="border-b">
                  <th className="py-1.5 pr-3 text-left text-xs font-medium text-muted-foreground">
                    Class
                  </th>
                  <th className="py-1.5 pr-3 text-left text-xs font-medium text-muted-foreground">
                    Previous
                  </th>
                  <th className="py-1.5 pr-3 text-left text-xs font-medium text-muted-foreground">
                    Current
                  </th>
                  <th className="py-1.5 text-left text-xs font-medium text-muted-foreground">
                    Shift
                  </th>
                </tr>
              </thead>
              <tbody>
                {result.class_shifts.map((s) => (
                  <ClassShiftRow key={s.class_label} shift={s} />
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        <p className="text-xs text-muted-foreground border-t pt-2">{result.summary}</p>
        <figcaption className="sr-only">
          Deployment prediction distribution comparison:{" "}
          {result.verdict_label || result.summary}
        </figcaption>
      </CardContent>
    </Card>
  )
}
