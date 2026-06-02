"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { PredictionValueTrendResult } from "@/lib/types"

interface Props {
  result: PredictionValueTrendResult
}

function directionColor(direction: string): string {
  switch (direction) {
    case "trending_up": return "emerald"
    case "trending_down": return "rose"
    case "no_data":
    case "not_applicable": return "gray"
    default: return "amber"
  }
}

function directionEmoji(direction: string): string {
  switch (direction) {
    case "trending_up": return "↗"
    case "trending_down": return "↘"
    case "no_data":
    case "not_applicable": return "—"
    default: return "→"
  }
}

function formatValue(v: number): string {
  const abs = Math.abs(v)
  if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (abs >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  if (abs < 10) return v.toFixed(2)
  return v.toFixed(1)
}

export function PredictionValueTrendCard({ result }: Props) {
  const color = directionColor(result.direction)
  const borderClass = `border-${color}-300`
  const badgeBg = `bg-${color}-100 text-${color}-700`
  const emoji = directionEmoji(result.direction)

  const changeAbs = Math.abs(result.overall_change_pct)
  const changeSign = result.overall_change_pct >= 0 ? "+" : ""
  const changeLabel = changeAbs >= 0.1
    ? `${changeSign}${result.overall_change_pct.toFixed(1)}%`
    : "no net change"

  if (result.direction === "no_data" || result.direction === "not_applicable" || result.periods.length === 0) {
    return (
      <figure
        className={`my-3 rounded-xl border-2 ${borderClass} bg-white p-4`}
        data-testid="prediction-value-trend-card"
      >
        <div className="flex items-center gap-2 mb-2">
          <span className="text-lg" aria-hidden="true">📈</span>
          <h3 className="text-sm font-semibold text-gray-700">Prediction Value Trend</h3>
        </div>
        <p className="text-sm text-gray-500">{result.summary}</p>
        <figcaption className="sr-only">Prediction value trend: {result.summary}</figcaption>
      </figure>
    )
  }

  return (
    <figure
      className={`my-3 rounded-xl border-2 ${borderClass} bg-white p-4`}
      data-testid="prediction-value-trend-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-lg" aria-hidden="true">📈</span>
        <h3 className="text-sm font-semibold text-gray-700">Prediction Value Trend</h3>
        <span
          className={`ml-auto text-xs font-semibold px-2 py-0.5 rounded-full ${badgeBg}`}
          data-testid="direction-badge"
        >
          {emoji} {result.direction_label}
        </span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="text-center bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500 mb-0.5">First period avg</p>
          <p className="text-sm font-semibold text-gray-700" data-testid="first-period-mean">
            {formatValue(result.first_period_mean)}
          </p>
        </div>
        <div className="text-center bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500 mb-0.5">Last period avg</p>
          <p className="text-sm font-semibold text-gray-700" data-testid="last-period-mean">
            {formatValue(result.last_period_mean)}
          </p>
        </div>
        <div className={`text-center rounded-lg p-2 bg-${color}-50`}>
          <p className="text-xs text-gray-500 mb-0.5">Net change</p>
          <p
            className={`text-sm font-semibold text-${color}-700`}
            data-testid="overall-change-pct"
          >
            {changeLabel}
          </p>
        </div>
      </div>

      {/* Line chart */}
      <div className="h-44" data-testid="trend-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={result.periods}
            margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="period_label"
              tick={{ fontSize: 10, fill: "#6b7280" }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#6b7280" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={formatValue}
              width={50}
            />
            <Tooltip
              formatter={(value) => [formatValue(Number(value)), result.target_col ?? "Prediction"]}
              labelStyle={{ fontSize: 11 }}
              contentStyle={{ fontSize: 11, borderRadius: 8 }}
            />
            <Line
              type="monotone"
              dataKey="mean"
              stroke={
                result.direction === "trending_up"
                  ? "#10b981"
                  : result.direction === "trending_down"
                  ? "#f43f5e"
                  : "#f59e0b"
              }
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Period count and total predictions */}
      <div className="flex items-center gap-3 mt-3 text-xs text-gray-500">
        <span data-testid="period-count">
          {result.n_periods_with_data} {result.period === "week" ? "weeks" : "days"}
        </span>
        <span>•</span>
        <span data-testid="n-total">{result.n_total} predictions</span>
        {result.algorithm && (
          <>
            <span>•</span>
            <span>{result.algorithm}</span>
          </>
        )}
      </div>

      {/* Summary */}
      <p className="text-xs text-gray-600 mt-3 leading-relaxed" data-testid="trend-summary">
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Prediction value trend: {result.direction_label}. {result.summary}
      </figcaption>
    </figure>
  )
}
