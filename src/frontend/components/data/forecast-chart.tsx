"use client"

/**
 * ForecastChart — renders a time-series forecast produced by core/forecaster.py.
 *
 * Layout:
 *   • Header: column name, trend badge (▲ up / ▼ down / → stable), period count
 *   • Recharts ComposedChart:
 *     - Historical line (solid blue)
 *     - Forecast line (dashed blue)
 *     - Confidence band: shaded area between lower/upper (light blue, forecast region only)
 *   • Summary paragraph (plain-English from the backend)
 *
 * The "confidence band" trick: we render the band as an Area on the combined
 * dataset where historical points have lower=value/upper=value (zero-width band)
 * and forecast points have the actual CI bounds.  This keeps the Area component
 * simple — one pass over the data, no two-Area subtraction hack needed.
 */

import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts"
import type { ForecastResult } from "@/lib/types"

interface Props {
  result: ForecastResult
}

/** Compact number formatter (1500 → 1.5K, 1200000 → 1.2M) */
function fmtNum(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}K`
  return v.toFixed(2)
}

function TrendBadge({ trend, growthPct }: { trend: string; growthPct: number }) {
  if (trend === "up") {
    return (
      <span
        className="inline-flex items-center gap-0.5 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700"
        data-testid="trend-badge-up"
      >
        ▲ +{Math.abs(growthPct)}%
      </span>
    )
  }
  if (trend === "down") {
    return (
      <span
        className="inline-flex items-center gap-0.5 rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700"
        data-testid="trend-badge-down"
      >
        ▼ −{Math.abs(growthPct)}%
      </span>
    )
  }
  return (
    <span
      className="inline-flex items-center gap-0.5 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600"
      data-testid="trend-badge-stable"
    >
      → Stable
    </span>
  )
}

export function ForecastChart({ result }: Props) {
  const {
    value_col,
    historical,
    forecast,
    period_label,
    trend,
    growth_pct,
    summary,
  } = result

  // Build a single merged array for Recharts.
  // Historical rows: historical=value, forecast=null, lower=value, upper=value
  // Forecast rows:   historical=null, forecast=value, lower=lower, upper=upper
  const cutoffDate = historical.length > 0 ? historical[historical.length - 1].date : null

  const chartData = [
    ...historical.map((p) => ({
      date: p.date,
      historical: p.value,
      forecast: null as number | null,
      lower: p.value,
      upper: p.value,
    })),
    // Overlap point: connect lines smoothly
    ...(forecast.length > 0 && historical.length > 0
      ? [
          {
            date: historical[historical.length - 1].date,
            historical: historical[historical.length - 1].value,
            forecast: historical[historical.length - 1].value,
            lower: historical[historical.length - 1].value,
            upper: historical[historical.length - 1].value,
          },
        ]
      : []),
    ...forecast.map((p) => ({
      date: p.date,
      historical: null as number | null,
      forecast: p.value,
      lower: p.lower,
      upper: p.upper,
    })),
  ]

  const forecastCount = forecast.length

  return (
    <div
      className="mt-3 rounded-lg border bg-white p-4 shadow-sm"
      data-testid="forecast-chart"
    >
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-gray-900">
            {value_col} — {forecastCount}-{period_label} Forecast
          </p>
          <p className="text-xs text-gray-500">
            Historical + projected values with 95% confidence band
          </p>
        </div>
        <TrendBadge trend={trend} growthPct={growth_pct} />
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11 }}
            tickFormatter={(v: string) => {
              // Shorten long date strings for readability
              if (typeof v === "string" && v.length > 8) return v.slice(0, 8)
              return v
            }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => fmtNum(v)}
            width={55}
          />
          <Tooltip
            formatter={(value, name) => {
              if (value === null || value === undefined) return ["—", String(name)]
              return [fmtNum(Number(value)), name === "historical" ? "Historical" : "Forecast"]
            }}
            labelStyle={{ fontSize: 12 }}
            contentStyle={{ fontSize: 12 }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12 }}
            formatter={(value: string) =>
              value === "historical"
                ? "Historical"
                : value === "forecast"
                  ? "Forecast"
                  : "95% CI"
            }
          />

          {/* Confidence band (forecast region only) */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="#bfdbfe"
            fillOpacity={0.5}
            name="95% CI"
            legendType="none"
            dot={false}
            activeDot={false}
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="white"
            fillOpacity={1}
            name="lower_hidden"
            legendType="none"
            dot={false}
            activeDot={false}
          />

          {/* Historical line */}
          <Line
            type="monotone"
            dataKey="historical"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            name="historical"
            connectNulls={false}
          />

          {/* Forecast line — dashed */}
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#2563eb"
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={{ r: 3, fill: "#2563eb" }}
            name="forecast"
            connectNulls={false}
          />

          {/* Cutoff reference line */}
          {cutoffDate && (
            <ReferenceLine
              x={cutoffDate}
              stroke="#94a3b8"
              strokeDasharray="4 2"
              label={{ value: "Today", position: "insideTopRight", fontSize: 10, fill: "#94a3b8" }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Summary */}
      <p className="mt-2 text-xs text-gray-600 leading-relaxed" data-testid="forecast-summary">
        {summary}
      </p>
    </div>
  )
}
