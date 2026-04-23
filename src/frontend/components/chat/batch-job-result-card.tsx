"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type {
  BatchClassDistributionEntry,
  BatchHistogramBin,
  BatchJobResultsResult,
} from "@/lib/types"

interface BatchJobResultCardProps {
  result: BatchJobResultsResult
}

function fmt(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return n.toFixed(2)
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return iso
  }
}

function RegressionStats({ result }: { result: BatchJobResultsResult }) {
  const stats = [
    { label: "Average", value: fmt(result.avg_prediction!) },
    { label: "Median", value: fmt(result.median_prediction!) },
    { label: "Min", value: fmt(result.min_prediction!) },
    { label: "Max", value: fmt(result.max_prediction!) },
  ]
  const bins: BatchHistogramBin[] = result.histogram ?? []
  const maxCount = Math.max(...bins.map((b) => b.count), 1)

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-2">
        {stats.map((s) => (
          <div
            key={s.label}
            className="rounded-lg bg-sky-50 border border-sky-100 p-2 text-center"
          >
            <div className="text-xs text-muted-foreground">{s.label}</div>
            <div className="text-sm font-semibold text-sky-800">{s.value}</div>
          </div>
        ))}
      </div>
      {bins.length > 0 && (
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-1">
            Prediction Distribution
          </div>
          <div
            className="flex items-end gap-0.5 h-12"
            aria-label="Prediction distribution histogram"
          >
            {bins.map((bin, i) => (
              <div
                key={i}
                className="flex-1 bg-sky-400 rounded-t-sm"
                style={{ height: `${Math.max(4, (bin.count / maxCount) * 100)}%` }}
                title={`${fmt(bin.bin_start)}–${fmt(bin.bin_end)}: ${bin.count}`}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-muted-foreground mt-0.5">
            <span>{fmt(bins[0].bin_start)}</span>
            <span>{fmt(bins[bins.length - 1].bin_end)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

function ClassificationStats({ result }: { result: BatchJobResultsResult }) {
  const dist: BatchClassDistributionEntry[] = result.class_distribution ?? []

  return (
    <div className="space-y-2">
      {result.avg_confidence != null && (
        <div className="text-xs text-muted-foreground">
          Avg confidence:{" "}
          <span className="font-medium text-foreground">
            {result.avg_confidence}%
          </span>
        </div>
      )}
      <div className="text-xs font-medium text-muted-foreground mb-1">
        Prediction Distribution
      </div>
      <div className="space-y-1.5">
        {dist.map((entry) => (
          <div key={entry.class_name} className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-24 truncate" title={entry.class_name}>
              {entry.class_name}
            </span>
            <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-teal-400 rounded-full"
                style={{ width: `${entry.pct}%` }}
              />
            </div>
            <span className="text-xs font-medium w-12 text-right">
              {entry.pct}%
            </span>
            <span className="text-xs text-muted-foreground w-10 text-right">
              ({entry.count})
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function BatchJobResultCard({ result }: BatchJobResultCardProps) {
  if (!result.has_results) {
    return (
      <Card className="border-slate-200 bg-slate-50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">📦</span> Batch Job Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{result.summary}</p>
          <p className="text-xs text-muted-foreground mt-1">
            Schedule a batch prediction run from the Deployment tab or by asking
            &quot;schedule daily batch predictions at 9am&quot;.
          </p>
        </CardContent>
      </Card>
    )
  }

  const isRegression = result.problem_type === "regression"

  return (
    <Card
      role="region"
      className="border-teal-400/40 bg-teal-50/30"
      aria-label="Batch job results"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center justify-between flex-wrap gap-2">
          <span className="flex items-center gap-2">
            <span aria-hidden="true">📦</span> Batch Job Results
          </span>
          <div className="flex items-center gap-1 flex-wrap">
            {result.total_rows != null && (
              <Badge className="bg-teal-100 text-teal-800 text-xs" variant="secondary">
                {result.total_rows.toLocaleString()} records
              </Badge>
            )}
            <Badge className="bg-slate-100 text-slate-700 text-xs" variant="secondary">
              {isRegression ? "Regression" : "Classification"}
            </Badge>
            {result.completed_at && (
              <Badge
                className="bg-slate-100 text-slate-600 text-xs font-normal"
                variant="secondary"
              >
                {fmtDate(result.completed_at)}
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {result.target_column && (
          <div className="text-xs text-muted-foreground">
            Target:{" "}
            <code className="bg-muted px-1 py-0.5 rounded text-foreground">
              {result.target_column}
            </code>
          </div>
        )}

        {isRegression ? (
          <RegressionStats result={result} />
        ) : (
          <ClassificationStats result={result} />
        )}

        <p className="text-xs text-muted-foreground italic">{result.summary}</p>

        <figcaption className="sr-only">
          Batch prediction job completed at{" "}
          {result.completed_at ? fmtDate(result.completed_at) : "unknown time"},{" "}
          scoring {result.total_rows ?? result.row_count ?? 0} records. {result.summary}
        </figcaption>
      </CardContent>
    </Card>
  )
}
