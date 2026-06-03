"use client"

import type { ProductionThresholdOptimizerResult, ProductionThresholdSweepPoint } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts"

interface Props {
  result: ProductionThresholdOptimizerResult
}

const VERDICT_CONFIG: Record<string, { border: string; badge: string; label: string }> = {
  improved: {
    border: "border-amber-200",
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    label: "⚡ Optimal ≠ Default",
  },
  same: {
    border: "border-emerald-200",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    label: "✓ Default Is Optimal",
  },
  no_data: {
    border: "border-gray-200",
    badge: "bg-gray-100 text-gray-600 border-gray-200",
    label: "⊘ Not Enough Data",
  },
  not_applicable: {
    border: "border-gray-200",
    badge: "bg-gray-100 text-gray-600 border-gray-200",
    label: "N/A",
  },
}

function StatBox({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color?: string
}) {
  return (
    <div className="flex flex-col items-center p-2 rounded bg-gray-50 border border-gray-100">
      <span className={`text-base font-bold ${color ?? "text-gray-800"}`}>{value}</span>
      <span className="text-[10px] text-muted-foreground text-center">{label}</span>
    </div>
  )
}

export function ProductionThresholdOptimizerCard({ result }: Props) {
  const cfg = VERDICT_CONFIG[result.verdict] ?? VERDICT_CONFIG.no_data

  if (result.verdict === "no_data" || result.verdict === "not_applicable") {
    return (
      <Card
        className={`${cfg.border} border`}
        role="region"
        aria-label="Production threshold optimizer: insufficient data"
      >
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm flex items-center gap-2">
              <span aria-hidden="true">🎯</span> Production Threshold Optimizer
            </CardTitle>
            <Badge className={`${cfg.badge} text-[10px]`}>{cfg.label}</Badge>
          </div>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          <p>{result.summary}</p>
          {result.n_feedback !== undefined && (
            <p className="mt-1 text-gray-400">
              Feedback records with labels: {result.n_feedback}
            </p>
          )}
        </CardContent>
        <figcaption className="sr-only">
          Production threshold optimizer: {result.summary}
        </figcaption>
      </Card>
    )
  }

  const { sweep = [], optimal_threshold = 0.5, current_threshold = 0.5 } = result

  const chartData = sweep.map((pt: ProductionThresholdSweepPoint) => ({
    threshold: `${Math.round(pt.threshold * 100)}%`,
    thresholdNum: pt.threshold,
    F1: Math.round(pt.f1 * 1000) / 1000,
    Precision: Math.round(pt.precision * 1000) / 1000,
    Recall: Math.round(pt.recall * 1000) / 1000,
    Coverage: Math.round(pt.coverage * 1000) / 1000,
  }))

  const optimalLabel = `${Math.round(optimal_threshold * 100)}%`
  const optimalF1 = result.optimal_f1 ?? 0
  const optimalPrecision = result.optimal_precision ?? 0
  const optimalRecall = result.optimal_recall ?? 0
  const optimalCoverage = result.optimal_coverage ?? 0
  const currentF1 = result.current_f1 ?? 0
  const overallAccuracy = result.overall_accuracy ?? 0
  const nFeedback = result.n_feedback ?? 0

  const f1Gain = optimalF1 - currentF1

  return (
    <Card
      className={`${cfg.border} border`}
      role="region"
      aria-label={`Production threshold optimizer: ${result.verdict_label}`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">🎯</span> Production Threshold Optimizer
          </CardTitle>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge className={`${cfg.badge} text-[10px]`}>{cfg.label}</Badge>
            <Badge className="bg-blue-50 text-blue-700 border-blue-200 text-[10px]" variant="outline">
              {nFeedback} outcomes
            </Badge>
            {result.algorithm && (
              <Badge className="bg-gray-100 text-gray-600 border-gray-200 text-[10px]" variant="outline">
                {result.algorithm}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Key stats at optimal threshold */}
        <div>
          <p className="text-[10px] text-muted-foreground mb-1.5 uppercase tracking-wide font-medium">
            At Optimal Threshold ({optimalLabel})
          </p>
          <div className="grid grid-cols-4 gap-2">
            <StatBox
              label="F1 Score"
              value={optimalF1.toFixed(3)}
              color={result.verdict === "improved" ? "text-amber-700" : "text-emerald-700"}
            />
            <StatBox label="Precision" value={optimalPrecision.toFixed(3)} />
            <StatBox label="Recall" value={optimalRecall.toFixed(3)} />
            <StatBox
              label="Coverage"
              value={`${Math.round(optimalCoverage * 100)}%`}
            />
          </div>
        </div>

        {/* Comparison row */}
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="rounded border border-gray-100 bg-gray-50 p-2 text-center">
            <div className="font-mono text-gray-800 font-bold">
              {(overallAccuracy * 100).toFixed(1)}%
            </div>
            <div className="text-[10px] text-muted-foreground">Overall Accuracy</div>
          </div>
          <div className="rounded border border-gray-100 bg-gray-50 p-2 text-center">
            <div className="font-mono text-gray-600 font-bold">{currentF1.toFixed(3)}</div>
            <div className="text-[10px] text-muted-foreground">
              F1 @ {Math.round(current_threshold * 100)}% (default)
            </div>
          </div>
          <div
            className={`rounded border p-2 text-center ${
              f1Gain > 0.01
                ? "border-amber-200 bg-amber-50"
                : "border-emerald-200 bg-emerald-50"
            }`}
          >
            <div
              className={`font-mono font-bold ${
                f1Gain > 0.01 ? "text-amber-700" : "text-emerald-700"
              }`}
            >
              {f1Gain > 0 ? `+${f1Gain.toFixed(3)}` : f1Gain.toFixed(3)}
            </div>
            <div className="text-[10px] text-muted-foreground">F1 Gain</div>
          </div>
        </div>

        {/* Sweep chart */}
        {chartData.length > 0 && (
          <div>
            <p className="text-[10px] text-muted-foreground mb-1.5 uppercase tracking-wide font-medium">
              Threshold Sweep (F1 / Precision / Recall)
            </p>
            <div
              aria-label={`Threshold sweep chart. Optimal threshold: ${optimalLabel}, F1=${optimalF1.toFixed(3)}`}
            >
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <XAxis
                    dataKey="threshold"
                    tick={{ fontSize: 9 }}
                    interval={3}
                  />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 9 }} />
                  <Tooltip
                    formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
                    labelFormatter={(l) => `Threshold: ${l}`}
                  />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <ReferenceLine
                    x={optimalLabel}
                    stroke="#f59e0b"
                    strokeDasharray="3 3"
                    label={{ value: "Optimal", fontSize: 9, fill: "#b45309" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="F1"
                    stroke="#7c3aed"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="Precision"
                    stroke="#2563eb"
                    strokeWidth={1.5}
                    dot={false}
                    strokeDasharray="4 2"
                  />
                  <Line
                    type="monotone"
                    dataKey="Recall"
                    stroke="#059669"
                    strokeWidth={1.5}
                    dot={false}
                    strokeDasharray="4 2"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Summary */}
        <p className="text-xs text-muted-foreground leading-relaxed border-t border-gray-100 pt-3">
          {result.summary}
        </p>

        <figcaption className="sr-only">
          Production threshold optimizer based on {nFeedback} real-world outcomes.
          Optimal threshold: {optimalLabel}, F1={optimalF1.toFixed(3)}.{" "}
          {result.summary}
        </figcaption>
      </CardContent>
    </Card>
  )
}
