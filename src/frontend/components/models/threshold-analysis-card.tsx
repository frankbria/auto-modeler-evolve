"use client"

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { ThresholdAnalysisResult, ThresholdRecommendation } from "@/lib/types"

interface ThresholdAnalysisCardProps {
  result: ThresholdAnalysisResult
}

function RecommendationRow({ rec, isBest }: { rec: ThresholdRecommendation; isBest?: boolean }) {
  return (
    <div
      className={`rounded-lg border p-3 ${isBest ? "border-amber-300 bg-amber-50" : "border-gray-200 bg-gray-50"}`}
      data-testid="threshold-recommendation"
    >
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <span className="font-semibold text-sm text-gray-800">{rec.label}</span>
        {isBest && (
          <span className="rounded px-1.5 py-0.5 text-xs font-semibold bg-amber-200 text-amber-800">
            Recommended
          </span>
        )}
        <span className="rounded px-1.5 py-0.5 text-xs font-mono bg-white border border-gray-300 text-gray-700">
          {Math.round(rec.threshold * 100)}%
        </span>
      </div>
      <div className="flex gap-4 mb-2 text-xs text-gray-600">
        <span>Precision: <strong>{Math.round(rec.precision * 100)}%</strong></span>
        <span>Recall: <strong>{Math.round(rec.recall * 100)}%</strong></span>
        <span>F1: <strong>{rec.f1.toFixed(2)}</strong></span>
      </div>
      <p className="text-xs text-gray-600 leading-relaxed">{rec.description}</p>
    </div>
  )
}

export function ThresholdAnalysisCard({ result }: ThresholdAnalysisCardProps) {
  const chartData = result.sweep.map((pt) => ({
    threshold: `${Math.round(pt.threshold * 100)}%`,
    Precision: Math.round(pt.precision * 100),
    Recall: Math.round(pt.recall * 100),
    F1: Math.round(pt.f1 * 100),
  }))

  const bestF1Pct = Math.round(result.recommendations.max_f1.threshold * 100)
  const currentPrecPct = Math.round(result.current_metrics.precision * 100)
  const currentRecPct = Math.round(result.current_metrics.recall * 100)
  const prevPct = Math.round(result.prevalence * 100)

  return (
    <figure
      className="rounded-xl border border-amber-300 bg-amber-50 p-4 space-y-4"
      role="region"
      aria-label="Classification threshold analysis"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-lg" aria-hidden="true">🎯</span>
        <span className="font-semibold text-gray-800">Threshold Advisor</span>
        <span className="rounded-full px-2.5 py-0.5 text-xs font-medium bg-amber-200 text-amber-800">
          Classification
        </span>
        <span className="rounded-full px-2.5 py-0.5 text-xs font-medium bg-gray-200 text-gray-700">
          Target: {result.target_col}
        </span>
        <span className="rounded-full px-2.5 py-0.5 text-xs font-medium bg-gray-200 text-gray-700">
          {prevPct}% positive ({result.n_positive}/{result.n_total} rows)
        </span>
      </div>

      {/* Summary */}
      <p className="text-sm text-gray-700 leading-relaxed">{result.summary}</p>

      {/* Current threshold stats */}
      <div className="rounded-lg border border-gray-200 bg-white p-3">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          At Default 50% Threshold
        </p>
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-gray-500">Precision:</span>{" "}
            <strong className="text-gray-800">{currentPrecPct}%</strong>
          </div>
          <div>
            <span className="text-gray-500">Recall:</span>{" "}
            <strong className="text-gray-800">{currentRecPct}%</strong>
          </div>
          <div>
            <span className="text-gray-500">F1:</span>{" "}
            <strong className="text-gray-800">{result.current_metrics.f1.toFixed(2)}</strong>
          </div>
        </div>
      </div>

      {/* Precision-Recall-F1 curve */}
      <div>
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          Threshold Trade-off Curve
        </p>
        <div
          aria-label={`Precision, recall, and F1 score at thresholds from 5% to 95%. Best F1 at ${bestF1Pct}%.`}
        >
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="threshold"
                tick={{ fontSize: 10 }}
                label={{ value: "Threshold", position: "insideBottom", offset: -2, fontSize: 10 }}
              />
              <YAxis
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 10 }}
                width={36}
              />
              <Tooltip formatter={(val) => (typeof val === "number" ? `${val}%` : val)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine
                x={`${bestF1Pct}%`}
                stroke="#d97706"
                strokeDasharray="4 2"
                label={{ value: `Best F1`, position: "top", fontSize: 9, fill: "#d97706" }}
              />
              <Line
                type="monotone"
                dataKey="Precision"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="Recall"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="F1"
                stroke="#d97706"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recommendations */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
          Threshold Options
        </p>
        <RecommendationRow rec={result.recommendations.max_f1} isBest />
        <RecommendationRow rec={result.recommendations.high_recall} />
        <RecommendationRow rec={result.recommendations.high_precision} />
      </div>

      {/* Guidance */}
      <div className="rounded-lg bg-white border border-amber-200 p-3">
        <p className="text-xs font-semibold text-amber-800 mb-1">Which threshold is right for me?</p>
        <ul className="text-xs text-gray-600 space-y-1 list-disc list-inside">
          <li><strong>Churn / fraud / medical</strong> — use High Recall to avoid missing cases</li>
          <li><strong>Costly interventions / outbound calls</strong> — use High Precision to avoid wasted effort</li>
          <li><strong>General use / balanced goal</strong> — use Max F1 (best balance)</li>
        </ul>
      </div>

      <figcaption className="sr-only">
        Classification threshold analysis for {result.target_col}: {result.summary}
      </figcaption>
    </figure>
  )
}
