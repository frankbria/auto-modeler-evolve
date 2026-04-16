"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { DeploymentVersionComparisonResult, VersionMetricDiff } from "@/lib/types"

interface DeploymentVersionComparisonCardProps {
  result: DeploymentVersionComparisonResult
}

const METRIC_LABELS: Record<string, string> = {
  r2: "R²",
  accuracy: "Accuracy",
  mae: "MAE",
  rmse: "RMSE",
  f1: "F1",
  precision: "Precision",
  recall: "Recall",
}

function formatMetricValue(metric: string, value: number): string {
  if (metric === "r2" || metric === "accuracy" || metric === "f1" || metric === "precision" || metric === "recall") {
    return (value * 100).toFixed(1) + "%"
  }
  if (Math.abs(value) >= 1000) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(2)
  return value.toFixed(4)
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "unknown date"
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
  } catch {
    return iso
  }
}

function MetricRow({ diff }: { diff: VersionMetricDiff }) {
  const label = METRIC_LABELS[diff.metric] ?? diff.metric.toUpperCase()
  const prevFmt = formatMetricValue(diff.metric, diff.previous)
  const curFmt = formatMetricValue(diff.metric, diff.current)
  const pctAbs = Math.abs(diff.pct_change)
  const sign = diff.delta > 0 ? "+" : diff.delta < 0 ? "" : ""
  const pctStr = `${sign}${diff.pct_change.toFixed(1)}%`

  const dirClass = diff.improved
    ? "text-emerald-600"
    : diff.delta === 0
    ? "text-muted-foreground"
    : "text-rose-600"

  const arrow = diff.direction === "up" ? "↑" : diff.direction === "down" ? "↓" : "→"

  return (
    <tr className="border-b last:border-0">
      <td className="py-1.5 pr-3 text-xs font-medium text-foreground">{label}</td>
      <td className="py-1.5 pr-3 text-xs text-muted-foreground font-mono">{prevFmt}</td>
      <td className="py-1.5 pr-3 text-xs font-mono text-foreground">{curFmt}</td>
      <td className={`py-1.5 text-xs font-medium ${dirClass} whitespace-nowrap`}>
        <span aria-hidden="true">{arrow}</span>{" "}
        {pctAbs >= 0.05 ? pctStr : "±0%"}
      </td>
    </tr>
  )
}

export function DeploymentVersionComparisonCard({ result }: DeploymentVersionComparisonCardProps) {
  // Determine overall border colour based on net improvement
  const improved = result.improved_count ?? 0
  const declined = result.declined_count ?? 0
  const borderClass =
    !result.has_comparison
      ? "border-slate-300"
      : improved > declined
      ? "border-emerald-500/40"
      : declined > improved
      ? "border-rose-500/40"
      : "border-amber-500/40"

  const algoChanged = result.algorithm_changed === true

  return (
    <Card
      data-testid="version-comparison-card"
      role="region"
      aria-label="Deployment version comparison"
      className={borderClass}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <span aria-hidden="true">🔄</span> Version Comparison
          </CardTitle>
          {result.has_comparison && (
            <div className="flex gap-1.5 flex-wrap">
              <Badge className="bg-slate-100 text-slate-700 border-slate-200 text-xs">
                v{result.previous_version} → v{result.current_version}
              </Badge>
              {improved > 0 && (
                <Badge className="bg-emerald-100 text-emerald-800 border-emerald-200 text-xs">
                  {improved} improved
                </Badge>
              )}
              {declined > 0 && (
                <Badge className="bg-rose-100 text-rose-800 border-rose-200 text-xs">
                  {declined} declined
                </Badge>
              )}
            </div>
          )}
        </div>

        {result.has_comparison ? (
          <p className="text-xs text-muted-foreground">
            Comparing{" "}
            <span className="font-medium text-foreground">v{result.previous_version}</span>
            {" "}(deployed {formatDate(result.previous_deployed_at)}) →{" "}
            <span className="font-medium text-foreground">v{result.current_version}</span>
            {" "}(deployed {formatDate(result.current_deployed_at)}).
            {algoChanged && (
              <>
                {" "}
                <span className="text-amber-600 font-medium">Algorithm changed:</span>{" "}
                {result.previous_algorithm} → {result.current_algorithm}.
              </>
            )}
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">{result.summary}</p>
        )}
      </CardHeader>

      {result.has_comparison && (
        <CardContent className="space-y-3">
          {result.metric_diffs && result.metric_diffs.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full" aria-label="Metric comparison between versions">
                <thead>
                  <tr className="border-b">
                    <th className="py-1.5 pr-3 text-left text-xs font-medium text-muted-foreground">Metric</th>
                    <th className="py-1.5 pr-3 text-left text-xs font-medium text-muted-foreground">
                      v{result.previous_version}
                    </th>
                    <th className="py-1.5 pr-3 text-left text-xs font-medium text-muted-foreground">
                      v{result.current_version}
                    </th>
                    <th className="py-1.5 text-left text-xs font-medium text-muted-foreground">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {result.metric_diffs.map((diff) => (
                    <MetricRow key={diff.metric} diff={diff} />
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No comparable metrics found between versions.</p>
          )}

          {/* Summary footer */}
          <p className="text-xs text-muted-foreground border-t pt-2">{result.summary}</p>

          <p className="text-xs text-muted-foreground/70">
            <span className="font-medium text-foreground">Note:</span> For MAE and RMSE, lower is better — a decrease
            in these metrics is an improvement.
          </p>
        </CardContent>
      )}
    </Card>
  )
}
