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
import { PartialDependenceResult } from "@/lib/types"

interface PartialDependenceCardProps {
  result: PartialDependenceResult
}

function fmtVal(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return parseFloat(v.toFixed(4)).toString()
}

const CLASS_COLORS = [
  "#6366f1", // indigo
  "#10b981", // emerald
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // violet
  "#14b8a6", // teal
]

export function PartialDependenceCard({ result }: PartialDependenceCardProps) {
  const {
    feature,
    target_col,
    algorithm,
    problem_type,
    grid_values,
    mean_predictions,
    std_predictions,
    class_curves,
    n_training_rows,
    summary,
  } = result

  const featureLabel = feature.replace(/_/g, " ")
  const targetLabel = (target_col ?? "prediction").replace(/_/g, " ")
  const isRegression = problem_type === "regression"
  const isMulticlass = class_curves !== null && Object.keys(class_curves ?? {}).length > 2

  // Determine trend direction from first to last mean prediction
  const firstPred = mean_predictions[0] ?? 0
  const lastPred = mean_predictions[mean_predictions.length - 1] ?? 0
  const delta = lastPred - firstPred
  const trendLabel =
    Math.abs(delta) < 0.001 * Math.max(Math.abs(firstPred), 1)
      ? "flat"
      : delta > 0
        ? "increases"
        : "decreases"
  const trendColor =
    trendLabel === "flat"
      ? "bg-muted text-muted-foreground border-border"
      : trendLabel === "increases"
        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
        : "bg-rose-50 text-rose-700 border-rose-200"
  const trendArrow = trendLabel === "flat" ? "→" : trendLabel === "increases" ? "↑" : "↓"

  // Build main chart data
  const mainChartData = grid_values.map((x, i) => ({
    x,
    mean: mean_predictions[i],
    upper: mean_predictions[i] + (std_predictions[i] ?? 0),
    lower: mean_predictions[i] - (std_predictions[i] ?? 0),
  }))

  // Build multiclass chart data
  const classNames = class_curves ? Object.keys(class_curves) : []
  const classChartData = isMulticlass
    ? grid_values.map((x, i) => {
        const row: Record<string, number> & { x: number } = { x }
        for (const cls of classNames) {
          row[cls] = (class_curves?.[cls]?.[i] ?? 0)
        }
        return row
      })
    : []

  return (
    <figure
      className="rounded-lg border-2 border-purple-400 bg-purple-50 dark:bg-purple-950/20 dark:border-purple-600 p-4 text-sm my-2"
      aria-label={`Partial dependence plot: ${featureLabel} vs average ${targetLabel}`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-base" aria-hidden="true">📉</span>
        <p className="font-semibold text-foreground">
          Partial Dependence:{" "}
          <span className="capitalize">{featureLabel}</span>
          {" → "}
          <span className="capitalize">{targetLabel}</span>
        </p>
        <span className="rounded-full border border-purple-300 bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-800 dark:bg-purple-900/40 dark:text-purple-300">
          {isRegression ? "Regression" : "Classification"}
        </span>
        <span
          className={`rounded-full border px-2 py-0.5 text-xs font-medium ${trendColor}`}
        >
          {trendArrow} {trendLabel === "flat" ? "Flat" : trendLabel === "increases" ? "Increases" : "Decreases"}
        </span>
        <span className="rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
          {algorithm.replace(/_/g, " ")}
        </span>
      </div>

      {/* Explainer — key differentiator from sensitivity analysis */}
      <p className="text-xs text-muted-foreground mb-3 italic">
        Averaged over {n_training_rows.toLocaleString()} training records — more accurate than fixing other features at their means.
      </p>

      {/* Main chart */}
      {mainChartData.length > 1 && !isMulticlass && (
        <div className="mb-3">
          <figcaption className="sr-only">
            Line chart showing average {targetLabel} as {featureLabel} varies from{" "}
            {fmtVal(grid_values[0])} to {fmtVal(grid_values[grid_values.length - 1])}
          </figcaption>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={mainChartData} margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="x"
                tickFormatter={(v) => fmtVal(v)}
                tick={{ fontSize: 11 }}
                label={{
                  value: featureLabel,
                  position: "insideBottom",
                  offset: -2,
                  fontSize: 11,
                  fill: "#6b7280",
                }}
              />
              <YAxis
                tickFormatter={(v) => fmtVal(v)}
                tick={{ fontSize: 11 }}
                label={{
                  value: isRegression ? `avg ${targetLabel}` : "avg probability",
                  angle: -90,
                  position: "insideLeft",
                  offset: 8,
                  fontSize: 11,
                  fill: "#6b7280",
                }}
                width={56}
              />
              <Tooltip
                formatter={(v) => [
                  typeof v === "number" ? fmtVal(v) : String(v ?? ""),
                  isRegression ? `avg ${targetLabel}` : "avg prob",
                ]}
                labelFormatter={(l) => `${featureLabel}: ${fmtVal(Number(l))}`}
              />
              {/* ±1 std band (upper) */}
              <Line
                dataKey="upper"
                stroke="#a78bfa"
                strokeWidth={1}
                strokeDasharray="4 2"
                dot={false}
                name="+1 std"
              />
              {/* ±1 std band (lower) */}
              <Line
                dataKey="lower"
                stroke="#a78bfa"
                strokeWidth={1}
                strokeDasharray="4 2"
                dot={false}
                name="−1 std"
              />
              {/* Mean PDP line */}
              <Line
                dataKey="mean"
                stroke="#7c3aed"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4 }}
                name={`avg ${targetLabel}`}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-[10px] text-center text-muted-foreground mt-1">
            Dashed band = ±1 standard deviation across training rows
          </p>
        </div>
      )}

      {/* Multiclass per-class curves */}
      {isMulticlass && classChartData.length > 1 && (
        <div className="mb-3">
          <p className="text-xs font-medium text-muted-foreground mb-1">Average class probabilities</p>
          <figcaption className="sr-only">
            Multiclass line chart showing average probability per class as {featureLabel} varies
          </figcaption>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={classChartData} margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="x"
                tickFormatter={(v) => fmtVal(v)}
                tick={{ fontSize: 11 }}
                label={{ value: featureLabel, position: "insideBottom", offset: -2, fontSize: 11, fill: "#6b7280" }}
              />
              <YAxis
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fontSize: 11 }}
                domain={[0, 1]}
                width={44}
              />
              <Tooltip
                formatter={(v, name) => [
                  typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v ?? ""),
                  String(name),
                ]}
                labelFormatter={(l) => `${featureLabel}: ${fmtVal(Number(l))}`}
              />
              {classNames.map((cls, i) => (
                <Line
                  key={cls}
                  dataKey={cls}
                  stroke={CLASS_COLORS[i % CLASS_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  name={cls}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          {/* Colour legend */}
          <div className="flex flex-wrap gap-3 mt-1">
            {classNames.map((cls, i) => (
              <span key={cls} className="flex items-center gap-1 text-[11px] text-muted-foreground">
                <span
                  className="inline-block h-2 w-4 rounded-sm"
                  style={{ background: CLASS_COLORS[i % CLASS_COLORS.length] }}
                  aria-hidden="true"
                />
                {cls}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Single-point (constant feature) fallback */}
      {mainChartData.length === 1 && (
        <p className="text-xs text-muted-foreground mb-3">
          Feature &ldquo;{featureLabel}&rdquo; is constant in the training data (value ={" "}
          {fmtVal(grid_values[0])}) — no dependence curve to display.
        </p>
      )}

      {/* Summary */}
      <p className="text-xs text-foreground border-t border-purple-200 dark:border-purple-700 pt-2">
        {summary}
      </p>
    </figure>
  )
}
