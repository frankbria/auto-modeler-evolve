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
import { SensitivityResult } from "@/lib/types"

interface SensitivityCardProps {
  result: SensitivityResult
}

function fmtVal(v: number | null | undefined): string {
  if (v === null || v === undefined) return "N/A"
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return parseFloat(v.toFixed(4)).toString()
}

export function SensitivityCard({ result }: SensitivityCardProps) {
  const {
    feature,
    target_column,
    problem_type,
    values,
    predictions,
    confidences,
    min_pred,
    max_pred,
    change_pct,
    summary,
  } = result

  const featureLabel = feature.replace(/_/g, " ")
  const targetLabel = (target_column ?? "prediction").replace(/_/g, " ")
  const isRegression = problem_type === "regression"

  // Build chart data
  const chartData = values.map((v, i) => {
    const pred = predictions[i]
    const conf = confidences?.[i]
    return {
      x: parseFloat(v.toFixed(4)),
      y: isRegression && typeof pred === "number" ? pred : conf ?? null,
      label: isRegression ? fmtVal(typeof pred === "number" ? pred : null) : String(pred),
    }
  })

  const hasNumericCurve = chartData.some((d) => d.y !== null)

  const changeBadgeColor =
    change_pct === null
      ? "bg-muted text-muted-foreground border-border"
      : change_pct > 0
        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
        : change_pct < 0
          ? "bg-rose-50 text-rose-700 border-rose-200"
          : "bg-muted text-muted-foreground border-border"

  const changeArrow = change_pct === null ? "→" : change_pct > 0 ? "↑" : change_pct < 0 ? "↓" : "→"

  return (
    <figure
      className="rounded-lg border-2 border-teal-400 bg-teal-50 dark:bg-teal-950/20 dark:border-teal-600 p-4 text-sm my-2"
      aria-label={`Sensitivity analysis: ${featureLabel} vs ${targetLabel}`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-base" aria-hidden="true">🎚️</span>
        <p className="font-semibold text-foreground">
          Sensitivity: <span className="capitalize">{featureLabel}</span> → <span className="capitalize">{targetLabel}</span>
        </p>
        <span className="rounded-full border border-teal-300 bg-teal-100 px-2 py-0.5 text-xs font-medium text-teal-800 dark:bg-teal-900/40 dark:text-teal-300">
          {isRegression ? "Regression" : "Classification"}
        </span>
        {change_pct !== null && (
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${changeBadgeColor}`}
          >
            {changeArrow} {Math.abs(change_pct).toFixed(1)}% range
          </span>
        )}
      </div>

      {/* Min / Max badges for regression */}
      {isRegression && min_pred !== null && max_pred !== null && (
        <div className="flex gap-3 mb-3">
          <div className="rounded-md border border-border bg-white dark:bg-teal-950/30 px-3 py-1.5 text-center">
            <p className="text-xs text-muted-foreground">Min {targetLabel}</p>
            <p className="font-bold text-sm">{fmtVal(min_pred)}</p>
          </div>
          <div className="rounded-md border border-teal-300 bg-teal-100/60 dark:bg-teal-900/20 px-3 py-1.5 text-center">
            <p className="text-xs text-muted-foreground">Max {targetLabel}</p>
            <p className="font-bold text-sm text-teal-700 dark:text-teal-300">{fmtVal(max_pred)}</p>
          </div>
        </div>
      )}

      {/* Line chart (regression or classification confidence) */}
      {hasNumericCurve && (
        <div className="mb-3">
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d1d5db" />
              <XAxis
                dataKey="x"
                tick={{ fontSize: 10 }}
                label={{
                  value: featureLabel,
                  position: "insideBottom",
                  offset: -2,
                  fontSize: 10,
                }}
              />
              <YAxis
                tick={{ fontSize: 10 }}
                width={48}
                label={{
                  value: isRegression ? targetLabel : "confidence",
                  angle: -90,
                  position: "insideLeft",
                  offset: 10,
                  fontSize: 10,
                }}
              />
              <Tooltip
                formatter={(v) => [
                  v !== undefined && v !== null ? fmtVal(Number(v)) : "N/A",
                  isRegression ? targetLabel : "confidence",
                ]}
                labelFormatter={(l) => `${featureLabel}: ${l}`}
              />
              <Line
                type="monotone"
                dataKey="y"
                stroke="#0d9488"
                strokeWidth={2}
                dot={values.length <= 20}
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Classification: prediction labels table (when no numeric curve) */}
      {!isRegression && !hasNumericCurve && (
        <div className="mb-3 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-teal-200">
                <th className="py-1 pr-3 text-left font-medium text-muted-foreground capitalize">{featureLabel}</th>
                <th className="py-1 text-left font-medium text-muted-foreground capitalize">Predicted {targetLabel}</th>
              </tr>
            </thead>
            <tbody>
              {values.map((v, i) => (
                <tr key={i} className="border-b border-teal-100 last:border-0">
                  <td className="py-1 pr-3 tabular-nums">{parseFloat(v.toFixed(4))}</td>
                  <td className="py-1 font-medium">{String(predictions[i])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Summary footer */}
      <figcaption className="mt-2 text-xs text-muted-foreground border-t border-teal-200 pt-2">
        {summary}
      </figcaption>
    </figure>
  )
}
