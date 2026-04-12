"use client"

import type { CalibrationCheckResult, CalibrationPoint } from "@/lib/types"
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

interface CalibrationCheckCardProps {
  result: CalibrationCheckResult
}

function qualityBadge(quality: CalibrationCheckResult["calibration_quality"]) {
  switch (quality) {
    case "excellent":
      return (
        <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800">
          Excellent
        </span>
      )
    case "good":
      return (
        <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
          Good
        </span>
      )
    case "poor":
      return (
        <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">
          Needs attention
        </span>
      )
    default:
      return (
        <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
          Unknown
        </span>
      )
  }
}

function formatBrier(score: number | null): string {
  if (score === null || score === undefined) return "N/A"
  return score.toFixed(3)
}

interface ChartPoint {
  label: string
  actual: number
  perfect: number
}

function buildChartData(curve: CalibrationPoint[]): ChartPoint[] {
  return curve.map((pt) => ({
    label: `${Math.round(pt.predicted * 100)}%`,
    actual: Math.round(pt.actual * 100) / 100,
    perfect: Math.round(pt.predicted * 100) / 100,
  }))
}

export default function CalibrationCheckCard({
  result,
}: CalibrationCheckCardProps) {
  const chartData = buildChartData(result.calibration_curve)

  return (
    <div
      role="region"
      aria-label="Model calibration check"
      className="mt-2 rounded-lg border border-violet-300 bg-violet-50 p-4"
    >
      {/* Header */}
      <div className="mb-3 flex items-center gap-2">
        <span aria-hidden="true">🎯</span>
        <span className="font-semibold text-violet-900">
          Confidence Calibration Check
        </span>
        {qualityBadge(result.calibration_quality)}
        <span className="ml-auto rounded-full bg-white px-2 py-0.5 text-xs font-mono text-violet-700 ring-1 ring-violet-200">
          {result.algorithm}
        </span>
      </div>

      {/* Brier score row */}
      <div className="mb-3 flex items-baseline gap-2">
        <span className="text-xs text-muted-foreground">Brier score:</span>
        <span className="text-sm font-semibold text-violet-900">
          {formatBrier(result.brier_score)}
        </span>
        <span className="text-xs text-muted-foreground">
          (lower is better; 0 = perfect, 0.25 = random)
        </span>
      </div>

      {/* Plain-English summary */}
      {result.summary && (
        <p className="mb-3 text-sm text-violet-800">{result.summary}</p>
      )}

      {/* Reliability diagram */}
      {chartData.length > 0 ? (
        <div className="mb-3">
          <p className="mb-1 text-xs text-muted-foreground">
            Reliability diagram — bars should follow the diagonal for a
            well-calibrated model.
          </p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={chartData}
              margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10 }}
                label={{
                  value: "Predicted confidence",
                  position: "insideBottom",
                  offset: -2,
                  fontSize: 10,
                }}
              />
              <YAxis
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fontSize: 10 }}
                domain={[0, 1]}
              />
              <Tooltip
                formatter={(v, name) => [
                  typeof v === "number" ? `${Math.round(v * 100)}%` : String(v ?? ""),
                  name === "actual" ? "Actual frequency" : "Perfect calibration",
                ]}
              />
              <ReferenceLine
                stroke="#7c3aed"
                strokeDasharray="4 2"
                segment={[
                  { x: chartData[0]?.label, y: chartData[0]?.perfect },
                  {
                    x: chartData[chartData.length - 1]?.label,
                    y: chartData[chartData.length - 1]?.perfect,
                  },
                ]}
              />
              <Bar dataKey="actual" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-1 text-center text-xs italic text-muted-foreground">
            The dashed diagonal = perfect calibration
          </p>
        </div>
      ) : (
        <p className="mb-3 text-sm text-muted-foreground">
          No calibration curve data available for this model.
        </p>
      )}

      {/* Calibration note */}
      {result.calibration_note && (
        <p className="text-xs text-muted-foreground">{result.calibration_note}</p>
      )}
    </div>
  )
}
