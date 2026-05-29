"use client"

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { ConfidenceDistributionResult } from "@/lib/types"

interface ConfidenceDistributionCardProps {
  result: ConfidenceDistributionResult
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

function decisivenessColor(d: string): string {
  if (d === "decisive") return "text-emerald-700 bg-emerald-100 border-emerald-300"
  if (d === "moderate") return "text-amber-700 bg-amber-100 border-amber-300"
  return "text-rose-700 bg-rose-100 border-rose-300"
}

function borderColor(d: string): string {
  if (d === "decisive") return "border-emerald-300 bg-emerald-50/30"
  if (d === "moderate") return "border-amber-300 bg-amber-50/30"
  return "border-rose-300 bg-rose-50/30"
}

function barFillForBin(lo: number, hi: number): string {
  const mid = (lo + hi) / 2
  if (mid >= 0.8) return "#10b981"
  if (mid >= 0.5) return "#f59e0b"
  return "#f87171"
}

export function ConfidenceDistributionCard({ result }: ConfidenceDistributionCardProps) {
  const chartData = result.bins.map((b) => ({
    label: b.label,
    count: b.count,
    pct: b.pct,
    fill: barFillForBin(b.lo, b.hi),
  }))

  return (
    <figure
      className={`rounded-xl border p-4 my-2 text-sm ${borderColor(result.decisiveness)}`}
      data-testid="confidence-distribution-card"
    >
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-base font-semibold text-gray-800" aria-hidden="true">
          🎯 Confidence Distribution
        </span>
        <span
          className={`rounded border px-2 py-0.5 text-xs font-semibold ${decisivenessColor(result.decisiveness)}`}
          data-testid="decisiveness-badge"
        >
          {result.decisiveness_label}
        </span>
        <span className="rounded border border-gray-200 bg-white px-2 py-0.5 text-xs text-gray-600">
          {algoName(result.algorithm)}
        </span>
        <span className="rounded border border-gray-200 bg-white px-2 py-0.5 text-xs text-gray-600">
          target: <code className="font-mono">{result.target_col}</code>
        </span>
      </div>

      {/* Segment breakdown */}
      <div className="grid grid-cols-3 gap-2 mb-3" data-testid="confidence-breakdown">
        <div className="rounded border border-emerald-200 bg-emerald-50 p-2 text-center">
          <div className="text-lg font-bold text-emerald-700">{result.pct_high}%</div>
          <div className="text-xs text-emerald-600">High (≥80%)</div>
          <div className="text-xs text-gray-500">{result.n_high} preds</div>
        </div>
        <div className="rounded border border-amber-200 bg-amber-50 p-2 text-center">
          <div className="text-lg font-bold text-amber-700">{result.pct_medium}%</div>
          <div className="text-xs text-amber-600">Medium (50–80%)</div>
          <div className="text-xs text-gray-500">{result.n_medium} preds</div>
        </div>
        <div className="rounded border border-rose-200 bg-rose-50 p-2 text-center">
          <div className="text-lg font-bold text-rose-700">{result.pct_low}%</div>
          <div className="text-xs text-rose-600">Low (&lt;50%)</div>
          <div className="text-xs text-gray-500">{result.n_low} preds</div>
        </div>
      </div>

      {/* Stats row */}
      <div className="flex gap-4 mb-3 text-xs text-gray-600">
        <span>
          Mean confidence:{" "}
          <strong className="text-gray-800">{(result.mean_confidence * 100).toFixed(1)}%</strong>
        </span>
        <span>
          Median:{" "}
          <strong className="text-gray-800">{(result.median_confidence * 100).toFixed(1)}%</strong>
        </span>
        <span>
          N:{" "}
          <strong className="text-gray-800">{result.n_total.toLocaleString()}</strong>
        </span>
      </div>

      {/* Histogram */}
      <div className="h-36 mb-3" aria-label="Confidence histogram">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 2, right: 4, left: 0, bottom: 2 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 9 }}
              interval={0}
              angle={-30}
              textAnchor="end"
              height={30}
            />
            <YAxis tick={{ fontSize: 10 }} width={28} />
            <Tooltip
              formatter={(v, name) => {
                if (name === "pct") return [`${v}%`, "% of predictions"]
                return [v as number, "count"]
              }}
            />
            <Bar dataKey="count" name="count" radius={[2, 2, 0, 0]}>
              {chartData.map((entry, idx) => (
                <Cell key={idx} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-class mean confidence */}
      {result.per_class_mean.length > 0 && (
        <div className="mb-3">
          <div className="text-xs font-semibold text-gray-500 uppercase mb-1">
            Mean confidence by class
          </div>
          <div className="space-y-1">
            {result.per_class_mean.map((cls) => (
              <div key={cls.class_name} className="flex items-center gap-2 text-xs" data-testid={`class-confidence-${cls.class_name}`}>
                <span className="w-20 truncate text-gray-700 font-mono">{cls.class_name}</span>
                <div className="flex-1 bg-gray-100 rounded h-2 overflow-hidden">
                  <div
                    className="h-full bg-indigo-400 rounded"
                    style={{ width: `${Math.min(cls.mean_confidence * 100, 100)}%` }}
                    role="progressbar"
                    aria-valuenow={Math.round(cls.mean_confidence * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
                <span className="w-10 text-right text-gray-600">
                  {(cls.mean_confidence * 100).toFixed(0)}%
                </span>
                <span className="text-gray-400">({cls.count})</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      <p className="text-xs text-gray-600 italic" data-testid="confidence-summary">
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Model confidence distribution showing {result.n_total} training predictions.
        Decisiveness verdict: {result.decisiveness_label}.
        {result.pct_high}% are high-confidence (≥80%), {result.pct_medium}% medium, {result.pct_low}% low.
      </figcaption>
    </figure>
  )
}
