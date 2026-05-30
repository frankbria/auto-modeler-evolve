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
import type {
  CalibrationCheckResult,
  CalibrationCurvePoint,
  CalibrationPerClassEntry,
} from "@/lib/types"

interface CalibrationCheckCardProps {
  result: CalibrationCheckResult
}

function algoName(algo: string): string {
  const MAP: Record<string, string> = {
    logistic_regression: "Logistic Regression",
    random_forest_classifier: "Random Forest",
    gradient_boosting_classifier: "Gradient Boosting",
    xgboost_classifier: "XGBoost",
    lightgbm_classifier: "LightGBM",
    decision_tree_classifier: "Decision Tree",
    svm_classifier: "Support Vector Machine",
    mlp_classifier: "Neural Network",
    voting_classifier: "Voting Ensemble",
    stacking_classifier: "Stacking Ensemble",
  }
  return MAP[algo] ?? algo
}

function verdictColors(v: CalibrationCheckResult["verdict"]) {
  if (v === "well_calibrated")
    return {
      badge: "text-emerald-700 bg-emerald-100 border-emerald-300",
      border: "border-emerald-300 bg-emerald-50/30",
      bar: "#10b981",
    }
  if (v === "moderate")
    return {
      badge: "text-amber-700 bg-amber-100 border-amber-300",
      border: "border-amber-300 bg-amber-50/30",
      bar: "#f59e0b",
    }
  return {
    badge: "text-rose-700 bg-rose-100 border-rose-300",
    border: "border-rose-300 bg-rose-50/30",
    bar: "#f87171",
  }
}

function buildChartData(points: CalibrationCurvePoint[]) {
  return points.map((pt) => ({
    label: pt.bin_label,
    actual: pt.frac_positive,
    predicted: pt.mean_prob,
    count: pt.count,
  }))
}

interface PerClassRowProps {
  entry: CalibrationPerClassEntry
  n_total: number
}

function PerClassRow({ entry, n_total }: PerClassRowProps) {
  const pct = n_total > 0 ? Math.round((entry.n_positive / n_total) * 100) : 0
  return (
    <div className="flex items-center gap-2 py-1 border-b border-gray-100 last:border-0">
      <span className="text-xs font-medium text-gray-700 min-w-[80px] truncate">
        {entry.class_name}
      </span>
      <span className="text-xs text-gray-500">{pct}%</span>
      <span className="text-xs text-gray-400">ECE {entry.ece.toFixed(3)}</span>
      <span className="text-xs text-gray-400">Brier {entry.brier_score.toFixed(3)}</span>
      <span className="text-xs text-gray-400">{entry.n_positive} samples</span>
    </div>
  )
}

export default function CalibrationCheckCard({ result }: CalibrationCheckCardProps) {
  const colors = verdictColors(result.verdict)
  const chartData = buildChartData(result.curve_points)

  return (
    <figure
      className={`rounded-xl border p-4 my-2 text-sm ${colors.border}`}
      data-testid="calibration-check-card"
      aria-label="Model calibration check"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-base font-semibold text-gray-800" aria-hidden="true">
          📐 Calibration Check
        </span>
        <span
          className={`rounded border px-2 py-0.5 text-xs font-semibold ${colors.badge}`}
          data-testid="calibration-verdict-badge"
        >
          {result.verdict_label}
        </span>
        <span className="rounded border border-gray-200 bg-white px-2 py-0.5 text-xs text-gray-600">
          {algoName(result.algorithm)}
        </span>
        <span className="rounded border border-gray-200 bg-white px-2 py-0.5 text-xs text-gray-600">
          {result.target_col}
        </span>
        <span className="rounded border border-gray-200 bg-white px-2 py-0.5 text-xs text-gray-600">
          {result.n_classes} {result.n_classes === 1 ? "class" : "classes"}
        </span>
        <span className="rounded border border-gray-200 bg-white px-2 py-0.5 text-xs text-gray-600">
          n={result.n_total.toLocaleString()}
        </span>
      </div>

      {/* ECE + Brier stats row */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="rounded-lg border border-gray-200 bg-white px-3 py-2">
          <div className="text-xs text-gray-500 mb-0.5">Expected Calibration Error</div>
          <div className="text-lg font-bold text-gray-800">{result.ece.toFixed(3)}</div>
          <div className="text-xs text-gray-400">
            {result.ece < 0.05 ? "✓ Well calibrated" : result.ece < 0.15 ? "~ Moderate" : "✗ Poor"}
          </div>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white px-3 py-2">
          <div className="text-xs text-gray-500 mb-0.5">Brier Score</div>
          <div className="text-lg font-bold text-gray-800">
            {result.brier_score.toFixed(3)}
          </div>
          <div className="text-xs text-gray-400">lower is better (0 = perfect)</div>
        </div>
      </div>

      {/* Reliability diagram */}
      {chartData.length > 0 && (
        <div className="mb-3">
          <p className="mb-1 text-xs text-gray-500">
            Reliability diagram — data points should follow the dashed diagonal for a
            well-calibrated model.
          </p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart
              data={chartData}
              margin={{ top: 4, right: 8, left: 0, bottom: 24 }}
              aria-label="Reliability diagram showing actual vs predicted probabilities"
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10 }}
                label={{
                  value: "Mean predicted probability",
                  position: "insideBottom",
                  offset: -16,
                  fontSize: 10,
                }}
              />
              <YAxis
                tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
                tick={{ fontSize: 10 }}
                domain={[0, 1]}
                label={{
                  value: "Fraction positive",
                  angle: -90,
                  position: "insideLeft",
                  offset: 10,
                  fontSize: 10,
                }}
              />
              <Tooltip
                formatter={(v: unknown, name: unknown) => [
                  typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v ?? ""),
                  name === "actual" ? "Actual fraction" : "Predicted prob.",
                ]}
              />
              <ReferenceLine
                stroke="#94a3b8"
                strokeDasharray="5 3"
                segment={[
                  { x: chartData[0]?.label, y: 0 },
                  { x: chartData[chartData.length - 1]?.label, y: 1 },
                ]}
                label={{ value: "Perfect", fontSize: 9, fill: "#94a3b8" }}
              />
              <Line
                type="monotone"
                dataKey="actual"
                stroke={colors.bar}
                strokeWidth={2}
                dot={{ r: 3, fill: colors.bar }}
                name="actual"
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="mt-1 text-center text-xs italic text-gray-400">
            Dashed line = perfect calibration
          </p>
        </div>
      )}

      {/* Per-class breakdown (multiclass only) */}
      {result.n_classes > 2 && result.per_class.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium text-gray-600 mb-1">Per-class calibration</p>
          <div className="rounded border border-gray-200 bg-white px-3 py-1">
            {result.per_class.map((entry) => (
              <PerClassRow key={entry.class_name} entry={entry} n_total={result.n_total} />
            ))}
          </div>
        </div>
      )}

      {/* Plain-English summary */}
      <p className="text-sm text-gray-700">{result.summary}</p>

      <figcaption className="sr-only">
        Calibration check for {algoName(result.algorithm)} predicting {result.target_col}.
        Verdict: {result.verdict_label}. ECE: {result.ece.toFixed(3)}. Brier score:{" "}
        {result.brier_score.toFixed(3)}.
      </figcaption>
    </figure>
  )
}
