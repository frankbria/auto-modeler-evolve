"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { DeploymentThroughputResult } from "@/lib/types"

interface Props {
  result: DeploymentThroughputResult
}

const VERDICT_COLORS: Record<string, string> = {
  instant: "border-emerald-500",
  fast: "border-emerald-400",
  moderate: "border-amber-400",
  slow: "border-orange-400",
  very_slow: "border-rose-500",
  no_data: "border-muted",
}

const VERDICT_BADGE: Record<string, { label: string; className: string }> = {
  instant: { label: "Instant", className: "bg-emerald-100 text-emerald-800" },
  fast: { label: "Fast", className: "bg-emerald-100 text-emerald-700" },
  moderate: { label: "Moderate", className: "bg-amber-100 text-amber-800" },
  slow: { label: "Slow", className: "bg-orange-100 text-orange-800" },
  very_slow: { label: "Very Slow", className: "bg-rose-100 text-rose-800" },
  no_data: { label: "No Data", className: "bg-muted text-muted-foreground" },
}

function StatBox({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex flex-col items-center gap-0.5 bg-muted rounded-lg px-3 py-2 min-w-[80px]">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-base font-semibold tabular-nums">{value}</span>
      {sub && <span className="text-[10px] text-muted-foreground">{sub}</span>}
    </div>
  )
}

export function DeploymentThroughputCard({ result }: Props) {
  const borderColor = VERDICT_COLORS[result.verdict] ?? "border-muted"
  const badge = VERDICT_BADGE[result.verdict] ?? VERDICT_BADGE.no_data

  const isNoData = result.verdict === "no_data"

  return (
    <Card
      className={`border-2 ${borderColor} mt-2`}
      data-testid="deployment-throughput-card"
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <span aria-hidden="true">⚡</span>
            Throughput Assessment
          </CardTitle>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge className={badge.className} data-testid="verdict-badge">
              {badge.label}
            </Badge>
            {!isNoData && (
              <Badge variant="outline" data-testid="target-n-badge">
                {result.target_n.toLocaleString()} predictions
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {isNoData ? (
          <p className="text-sm text-muted-foreground" data-testid="no-data-message">
            {result.summary}
          </p>
        ) : (
          <>
            <div className="flex items-center gap-2 flex-wrap" data-testid="latency-stats">
              <StatBox
                label="p50 latency"
                value={`${result.p50_ms?.toFixed(0) ?? "—"} ms`}
                sub="median"
              />
              <StatBox
                label="p95 latency"
                value={`${result.p95_ms?.toFixed(0) ?? "—"} ms`}
                sub="95th pct"
              />
              <StatBox
                label="p99 latency"
                value={`${result.p99_ms?.toFixed(0) ?? "—"} ms`}
                sub="99th pct"
              />
              <StatBox
                label="max RPS"
                value={result.max_rps != null ? result.max_rps.toFixed(1) : "—"}
                sub="req/sec"
              />
            </div>

            <div
              className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-sm"
              data-testid="duration-estimate"
            >
              <span aria-hidden="true">🕐</span>
              <span>
                Processing{" "}
                <span className="font-semibold">{result.target_n.toLocaleString()}</span>{" "}
                predictions serially:{" "}
                <span className="font-semibold">~{result.serial_duration}</span>
              </span>
            </div>

            <p className="text-xs text-muted-foreground" data-testid="sample-count">
              Based on {result.n_samples} measured predictions
            </p>
          </>
        )}

        <p className="text-sm text-muted-foreground italic" data-testid="summary">
          {result.summary}
        </p>

        <figcaption className="sr-only">
          Deployment throughput assessment: {result.verdict} verdict.{" "}
          {result.serial_duration
            ? `Processing ${result.target_n} predictions takes ~${result.serial_duration}.`
            : "Insufficient data to estimate throughput."}
        </figcaption>
      </CardContent>
    </Card>
  )
}
