"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ProdPerformanceResult, ProdPerformancePeriod } from "@/lib/types"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"

interface ProdPerformanceCardProps {
  result: ProdPerformanceResult
}

function StatusBadge({ status }: { status: string }) {
  if (status === "stable") {
    return (
      <Badge className="text-xs bg-emerald-100 text-emerald-800 border-emerald-300">
        Stable
      </Badge>
    )
  }
  if (status === "warning") {
    return (
      <Badge className="text-xs bg-amber-100 text-amber-800 border-amber-300">
        Warning
      </Badge>
    )
  }
  if (status === "degrading") {
    return (
      <Badge className="text-xs bg-rose-100 text-rose-800 border-rose-300">
        Degrading
      </Badge>
    )
  }
  return (
    <Badge className="text-xs bg-slate-100 text-slate-600 border-slate-300">
      No Feedback
    </Badge>
  )
}

function DegradationBadge({
  pct,
  direction,
}: {
  pct: number
  direction?: string
}) {
  const improved = direction === "lower_is_better" ? pct < 0 : pct <= 0
  const color = improved
    ? "bg-emerald-100 text-emerald-800 border-emerald-300"
    : Math.abs(pct) < 10
    ? "bg-slate-100 text-slate-700 border-slate-300"
    : Math.abs(pct) < 20
    ? "bg-amber-100 text-amber-800 border-amber-300"
    : "bg-rose-100 text-rose-800 border-rose-300"

  const label =
    direction === "lower_is_better"
      ? pct > 0
        ? `Error +${Math.abs(pct).toFixed(1)}% vs training`
        : `Error ${Math.abs(pct).toFixed(1)}% better than training`
      : pct > 0
      ? `Accuracy -${pct.toFixed(1)}% vs training`
      : `Accuracy ${Math.abs(pct).toFixed(1)}% above training`

  return (
    <Badge className={`text-xs ${color}`} data-testid="degradation-badge">
      {label}
    </Badge>
  )
}

function MetricBox({
  label,
  value,
  pct,
  highlight,
}: {
  label: string
  value: number | null | undefined
  pct?: number
  highlight?: "good" | "bad" | "neutral"
}) {
  const bg =
    highlight === "good"
      ? "bg-emerald-50 border-emerald-200"
      : highlight === "bad"
      ? "bg-rose-50 border-rose-200"
      : "bg-gray-50 border-gray-200"

  return (
    <div
      className={`rounded-lg border p-3 text-center ${bg}`}
      data-testid={`metric-box-${label.toLowerCase().replace(/\s+/g, "-")}`}
    >
      <p className="text-xs text-gray-500 font-medium">{label}</p>
      <p className="text-2xl font-bold text-gray-900 mt-1">
        {value !== null && value !== undefined
          ? pct !== undefined
            ? `${pct}%`
            : value.toFixed(4)
          : "—"}
      </p>
    </div>
  )
}

function Timeline({
  data,
  trainingValue,
  direction,
  metricName,
}: {
  data: ProdPerformancePeriod[]
  trainingValue: number | null
  direction?: string
  metricName: string
}) {
  if (data.length < 2) return null

  const isAccuracy = direction === "higher_is_better"
  const formatted = data.map((d) => ({
    name: d.period.slice(0, 10),
    value: isAccuracy ? d.value : d.value,
    n: d.n,
  }))

  return (
    <figure aria-label={`${metricName} trend over time`}>
      <figcaption className="text-xs font-medium text-gray-600 mb-1">
        Live {metricName} over time
      </figcaption>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formatted}>
            <XAxis
              dataKey="name"
              tick={{ fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis tick={{ fontSize: 10 }} width={40} />
            <Tooltip
              formatter={(val) => {
                const n = typeof val === "number" ? val : 0
                return isAccuracy ? [`${n.toFixed(1)}%`, metricName] : [n.toFixed(4), metricName]
              }}
              labelFormatter={(label) => `Week of ${label}`}
            />
            {trainingValue !== null && trainingValue !== undefined && (
              <ReferenceLine
                y={isAccuracy ? trainingValue * 100 : trainingValue}
                stroke="#6366f1"
                strokeDasharray="4 2"
                label={{ value: "Training", fontSize: 10, fill: "#6366f1" }}
              />
            )}
            <Line
              type="monotone"
              dataKey="value"
              stroke="#0ea5e9"
              dot={{ r: 3 }}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  )
}

export function ProdPerformanceCard({ result }: ProdPerformanceCardProps) {
  const borderColor =
    result.status === "stable"
      ? "border-emerald-300"
      : result.status === "warning"
      ? "border-amber-300"
      : result.status === "degrading"
      ? "border-rose-300"
      : "border-slate-300"

  const bgColor =
    result.status === "stable"
      ? "bg-emerald-50"
      : result.status === "warning"
      ? "bg-amber-50"
      : result.status === "degrading"
      ? "bg-rose-50"
      : "bg-slate-50"

  const isRegression = result.metric_direction === "lower_is_better"

  return (
    <figure
      className={`my-3 rounded-xl border-2 ${borderColor} ${bgColor} overflow-hidden`}
      data-testid="prod-performance-card"
      aria-label="Training vs production performance comparison"
    >
      <Card className="border-0 shadow-none bg-transparent">
        <CardHeader className="pb-2 pt-4 px-4">
          <div className="flex flex-wrap items-center gap-2">
            <span aria-hidden="true" className="text-lg">
              📊
            </span>
            <CardTitle className="text-sm font-semibold text-gray-800">
              Training vs Production Performance
            </CardTitle>
            <StatusBadge status={result.status} />
            {result.target_column && (
              <Badge className="text-xs bg-indigo-100 text-indigo-800 border-indigo-300">
                {result.target_column}
              </Badge>
            )}
            {result.n_feedback !== undefined && result.n_feedback > 0 && (
              <Badge className="text-xs bg-sky-100 text-sky-800 border-sky-300">
                {result.n_feedback} feedback record{result.n_feedback !== 1 ? "s" : ""}
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="px-4 pb-4 space-y-3">
          {!result.has_data ? (
            <div
              className="rounded-md bg-white/60 px-4 py-3 text-sm text-gray-600"
              data-testid="no-feedback-message"
            >
              <p>{result.summary}</p>
            </div>
          ) : (
            <>
              {/* Training vs Live metric boxes */}
              <div className="grid grid-cols-2 gap-3">
                <MetricBox
                  label={`Training ${result.metric_name}`}
                  value={result.training_value}
                  pct={isRegression ? undefined : result.training_pct}
                  highlight="neutral"
                />
                <MetricBox
                  label={`Live ${result.metric_name}`}
                  value={result.live_value}
                  pct={isRegression ? undefined : result.live_pct}
                  highlight={
                    result.status === "stable"
                      ? "good"
                      : result.status === "degrading"
                      ? "bad"
                      : "neutral"
                  }
                />
              </div>

              {/* Degradation badge */}
              {result.degradation_pct !== undefined && (
                <div className="flex flex-wrap gap-2">
                  <DegradationBadge
                    pct={result.degradation_pct}
                    direction={result.metric_direction}
                  />
                </div>
              )}

              {/* Timeline sparkline */}
              {result.weekly_timeline && result.weekly_timeline.length >= 2 && (
                <Timeline
                  data={result.weekly_timeline}
                  trainingValue={result.training_value}
                  direction={result.metric_direction}
                  metricName={result.metric_name}
                />
              )}

              {/* Status explanation */}
              {result.status === "warning" && (
                <div
                  className="rounded-md bg-amber-100 border border-amber-300 px-3 py-2 text-xs text-amber-800"
                  role="alert"
                >
                  Performance has declined since training. Monitor closely and consider
                  retraining if the trend continues.
                </div>
              )}
              {result.status === "degrading" && (
                <div
                  className="rounded-md bg-rose-100 border border-rose-300 px-3 py-2 text-xs text-rose-800"
                  role="alert"
                >
                  Significant performance drop detected. Retraining with recent data is
                  recommended.
                </div>
              )}
            </>
          )}

          {/* Summary */}
          <p className="text-xs text-gray-500 italic">{result.summary}</p>
        </CardContent>
      </Card>
    </figure>
  )
}
