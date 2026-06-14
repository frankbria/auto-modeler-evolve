"use client"

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { PerformanceDecayResult, PerformanceDecayVerdict } from "@/lib/types"

interface PerformanceDecayRateCardProps {
  result: PerformanceDecayResult
}

function verdictColors(v: PerformanceDecayVerdict): { border: string; badge: string } {
  switch (v) {
    case "degrading_fast":
    case "below_threshold":
      return { border: "border-rose-500", badge: "bg-rose-100 text-rose-800" }
    case "degrading_slowly":
      return { border: "border-amber-500", badge: "bg-amber-100 text-amber-800" }
    case "improving":
      return { border: "border-emerald-500", badge: "bg-emerald-100 text-emerald-800" }
    case "stable":
      return { border: "border-sky-500", badge: "bg-sky-100 text-sky-800" }
    default:
      return { border: "border-slate-300", badge: "bg-slate-100 text-slate-600" }
  }
}

function verdictLabel(v: PerformanceDecayVerdict): string {
  switch (v) {
    case "degrading_fast":
      return "Degrading fast"
    case "degrading_slowly":
      return "Degrading slowly"
    case "below_threshold":
      return "Below threshold"
    case "improving":
      return "Improving"
    case "stable":
      return "Stable"
    default:
      return "No data"
  }
}

function slopeLabel(slope: number | null): string {
  if (slope === null) return "—"
  const sign = slope >= 0 ? "+" : ""
  return `${sign}${slope.toFixed(2)}% / week`
}

export function PerformanceDecayRateCard({ result }: PerformanceDecayRateCardProps) {
  const { border, badge } = verdictColors(result.verdict)
  const hasData = result.weekly_data.length > 0

  return (
    <figure
      className={`rounded-lg border-2 ${border} bg-white p-4 space-y-4`}
      data-slot="performance-decay-rate-card"
    >
      <figcaption className="sr-only">Model performance decay rate analysis</figcaption>

      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xl" aria-hidden="true">
            📉
          </span>
          <h3 className="font-semibold text-slate-800 text-sm">Performance Decay Rate</h3>
        </div>
        <div className="flex flex-wrap gap-1">
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badge}`}>
            {verdictLabel(result.verdict)}
          </span>
          {hasData && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
              {result.weekly_data.length} week{result.weekly_data.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      </div>

      {/* Summary */}
      <p className="text-sm text-slate-700">{result.summary}</p>

      {hasData ? (
        <>
          {/* Stats row */}
          <div className="grid grid-cols-3 gap-2">
            <div className="rounded-md bg-slate-50 p-2 text-center">
              <div className="text-xs text-slate-500 mb-0.5">Current accuracy</div>
              <div className="text-base font-semibold text-slate-800">
                {result.current_accuracy_pct !== null
                  ? `${result.current_accuracy_pct.toFixed(1)}%`
                  : "—"}
              </div>
            </div>
            <div className="rounded-md bg-slate-50 p-2 text-center">
              <div className="text-xs text-slate-500 mb-0.5">Slope</div>
              <div
                className={`text-base font-semibold ${
                  result.slope_pct_per_week === null
                    ? "text-slate-800"
                    : result.slope_pct_per_week < -0.5
                      ? "text-rose-700"
                      : result.slope_pct_per_week > 0.5
                        ? "text-emerald-700"
                        : "text-slate-800"
                }`}
              >
                {slopeLabel(result.slope_pct_per_week)}
              </div>
            </div>
            <div className="rounded-md bg-slate-50 p-2 text-center">
              <div className="text-xs text-slate-500 mb-0.5">
                Weeks to {result.threshold_pct.toFixed(0)}%
              </div>
              <div className="text-base font-semibold text-slate-800">
                {result.weeks_until_threshold === null
                  ? "N/A"
                  : result.weeks_until_threshold === 0
                    ? "Now"
                    : `~${result.weeks_until_threshold}`}
              </div>
            </div>
          </div>

          {/* Line chart */}
          <div style={{ height: 180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={result.weekly_data}
                margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="week"
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                  width={38}
                  label={{
                    value: "Accuracy %",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    style: { fontSize: 10, fill: "#94a3b8" },
                  }}
                />
                <Tooltip
                  formatter={(val, name) => [
                    typeof val === "number" && name === "accuracy_pct"
                      ? `${val.toFixed(1)}%`
                      : val,
                    name === "accuracy_pct" ? "Accuracy" : String(name),
                  ]}
                  labelFormatter={(label) => `Week: ${label}`}
                  contentStyle={{ fontSize: 11 }}
                />
                {/* Threshold reference line */}
                <ReferenceLine
                  y={result.threshold_pct}
                  stroke="#f87171"
                  strokeDasharray="4 2"
                  label={{
                    value: `${result.threshold_pct.toFixed(0)}% threshold`,
                    position: "right",
                    fontSize: 9,
                    fill: "#f87171",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy_pct"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#6366f1" }}
                  activeDot={{ r: 5 }}
                  name="accuracy_pct"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Alert callout for degrading/below */}
          {(result.verdict === "degrading_fast" || result.verdict === "below_threshold") && (
            <div
              role="alert"
              className="rounded-md bg-rose-50 border border-rose-200 p-2 text-xs text-rose-800"
            >
              {result.verdict === "below_threshold"
                ? `Accuracy is already below the ${result.threshold_pct.toFixed(0)}% threshold. Retraining is strongly recommended.`
                : result.weeks_until_threshold !== null
                  ? `At the current rate, accuracy will fall below ${result.threshold_pct.toFixed(0)}% in approximately ${result.weeks_until_threshold} week${result.weeks_until_threshold !== 1 ? "s" : ""}.`
                  : `Accuracy is declining fast. Consider retraining soon.`}
            </div>
          )}
        </>
      ) : (
        <p className="text-sm text-slate-500 italic">
          No labeled feedback found. Submit ground-truth outcomes after predictions to track accuracy over time.
        </p>
      )}

      <p className="text-xs text-slate-400">
        Threshold: {result.threshold_pct.toFixed(0)}% · Accuracy computed from labeled feedback records
      </p>
    </figure>
  )
}
