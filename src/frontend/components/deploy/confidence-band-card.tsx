"use client"

import type { ConfidenceBandResult } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts"

interface ConfidenceBandCardProps {
  data: ConfidenceBandResult
}

function verdictBadge(verdict: ConfidenceBandResult["verdict"]) {
  switch (verdict) {
    case "stable":
      return (
        <Badge className="bg-emerald-100 text-emerald-800 border-emerald-200 text-[10px]">
          Stable Predictions
        </Badge>
      )
    case "moderate_spread":
      return (
        <Badge className="bg-amber-100 text-amber-800 border-amber-200 text-[10px]">
          Moderate Spread
        </Badge>
      )
    case "high_spread":
      return (
        <Badge className="bg-rose-100 text-rose-800 border-rose-200 text-[10px]">
          High Spread
        </Badge>
      )
    default:
      return (
        <Badge className="bg-muted text-muted-foreground text-[10px]">
          No Data
        </Badge>
      )
  }
}

function borderColor(verdict: ConfidenceBandResult["verdict"]) {
  switch (verdict) {
    case "stable":
      return "border-sky-200"
    case "moderate_spread":
      return "border-amber-200"
    case "high_spread":
      return "border-rose-200"
    default:
      return "border-muted"
  }
}

function formatValue(v: number | null, isClassification: boolean): string {
  if (v === null) return "—"
  return isClassification ? `${(v * 100).toFixed(1)}%` : v.toFixed(2)
}

export function ConfidenceBandCard({ data }: ConfidenceBandCardProps) {
  const isClassification = data.problem_type === "classification"
  const border = borderColor(data.verdict)

  if (data.verdict === "no_data" || data.n_samples === 0) {
    return (
      <Card
        className={border}
        role="region"
        aria-label="Prediction confidence band: no data yet"
      >
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">📊</span> Prediction Confidence Band
          </CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground italic">
          No prediction data yet — the chart will appear after predictions are
          made.
        </CardContent>
      </Card>
    )
  }

  // Build Recharts data: each day is a point with mean, lower, upper
  const chartData = data.days.map((day, i) => ({
    day: day.slice(5), // "MM-DD"
    mean: isClassification ? data.means[i] * 100 : data.means[i],
    lower: isClassification
      ? Math.max(0, data.lowers[i] * 100)
      : data.lowers[i],
    upper: isClassification
      ? Math.min(100, data.uppers[i] * 100)
      : data.uppers[i],
    // Area uses [lower, upper] as a range — encode as band width from lower
    band: [
      isClassification ? Math.max(0, data.lowers[i] * 100) : data.lowers[i],
      isClassification
        ? Math.min(100, data.uppers[i] * 100)
        : data.uppers[i],
    ],
  }))

  const yLabel = isClassification ? "Confidence (%)" : data.value_label

  return (
    <Card
      className={border}
      role="region"
      aria-label={`Prediction confidence band: ${data.verdict.replace("_", " ")}`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-1">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">📊</span> Prediction Confidence Band
          </CardTitle>
          <div className="flex items-center gap-1 flex-wrap">
            {verdictBadge(data.verdict)}
            <Badge className="bg-muted text-muted-foreground text-[10px]">
              {data.n_samples} prediction{data.n_samples !== 1 ? "s" : ""}
            </Badge>
            <Badge className="bg-muted text-muted-foreground text-[10px]">
              {data.n_days} day{data.n_days !== 1 ? "s" : ""}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-3 text-xs">
        {/* Summary */}
        <p className="text-muted-foreground">{data.summary}</p>

        {/* Stats grid */}
        {data.overall_mean !== null && (
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="font-semibold text-sm">
                {formatValue(data.overall_mean, isClassification)}
              </div>
              <div className="text-muted-foreground">Average</div>
            </div>
            <div>
              <div className="font-semibold text-sm">
                {formatValue(data.overall_std, isClassification)}
              </div>
              <div className="text-muted-foreground">Std Dev</div>
            </div>
            <div>
              <div className="font-semibold text-sm">
                {formatValue(data.overall_lower, isClassification)} –{" "}
                {formatValue(data.overall_upper, isClassification)}
              </div>
              <div className="text-muted-foreground">Typical range</div>
            </div>
          </div>
        )}

        {/* Band chart */}
        {chartData.length > 0 && (
          <figure aria-label={`${yLabel} trend with confidence band over ${data.n_days} days`}>
            <div className="h-36">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis
                    dataKey="day"
                    tick={{ fontSize: 9 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 9 }}
                    tickFormatter={(v) =>
                      isClassification ? `${v.toFixed(0)}%` : v.toFixed(1)
                    }
                    width={36}
                  />
                  <Tooltip
                    formatter={(value, name) => {
                      const v = typeof value === "number" ? value : Number(value)
                      const n = String(name ?? "")
                      const formatted = isClassification
                        ? `${v.toFixed(1)}%`
                        : v.toFixed(2)
                      const labels: Record<string, string> = {
                        mean: "Mean",
                        lower: "Lower (−1σ)",
                        upper: "Upper (+1σ)",
                      }
                      return [formatted, labels[n] ?? n]
                    }}
                    contentStyle={{ fontSize: "10px" }}
                  />
                  {/* Shaded band from lower to upper */}
                  <Area
                    type="monotone"
                    dataKey="upper"
                    stroke="none"
                    fill="#0ea5e9"
                    fillOpacity={0.15}
                  />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    stroke="none"
                    fill="#ffffff"
                    fillOpacity={1}
                  />
                  {/* Mean line */}
                  <Line
                    type="monotone"
                    dataKey="mean"
                    stroke="#0ea5e9"
                    strokeWidth={2}
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <figcaption className="sr-only">
              {`${yLabel} trend from ${data.days[0]} to ${data.days[data.days.length - 1]}. Shaded band shows ±1 standard deviation.`}
            </figcaption>
          </figure>
        )}

        <p className="text-muted-foreground text-[10px]">
          Shaded band = mean ± 1 standard deviation per day.{" "}
          {data.verdict === "high_spread"
            ? "High spread may indicate diverse input types or a need for retraining with more varied data."
            : data.verdict === "stable"
            ? "Stable predictions indicate consistent model behaviour in production."
            : "Moderate spread is normal — monitor for increasing variance over time."}
        </p>
      </CardContent>
    </Card>
  )
}
