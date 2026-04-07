"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import type { LearningCurveResult } from "@/lib/types"

interface LearningCurveCardProps {
  result: LearningCurveResult
}

export function LearningCurveCard({ result }: LearningCurveCardProps) {
  const {
    sizes_pct,
    train_scores,
    val_scores,
    converged,
    plateau_pct,
    best_val_score,
    metric_label,
    n_total,
    algorithm_name,
    recommendation,
    summary,
  } = result

  // Build chart data
  const chartData = sizes_pct.map((pct, i) => ({
    pct,
    train: train_scores[i],
    val: val_scores[i],
  }))

  const isPercent = metric_label !== "R²"
  const fmt = (v: number) =>
    isPercent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3)

  const convergeBadge = converged ? (
    <Badge className="bg-emerald-100 text-emerald-800 border-emerald-300 text-xs">
      Converged
    </Badge>
  ) : (
    <Badge className="bg-amber-100 text-amber-800 border-amber-300 text-xs">
      Still Learning
    </Badge>
  )

  return (
    <figure aria-label="Learning curve analysis" className="not-prose">
      <Card className="border-indigo-500/30">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <span aria-hidden="true">📈</span> Learning Curve Analysis
            </CardTitle>
            {convergeBadge}
            <Badge variant="outline" className="text-xs ml-auto">
              {n_total.toLocaleString()} rows · {algorithm_name}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Summary */}
          <p className="text-xs text-muted-foreground">{summary}</p>

          {/* Chart */}
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">
              {metric_label} vs. Training Data Size
            </p>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart
                data={chartData}
                margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="pct"
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fontSize: 10 }}
                  label={{
                    value: "% of training data",
                    position: "insideBottom",
                    offset: -2,
                    fontSize: 10,
                  }}
                />
                <YAxis
                  tickFormatter={fmt}
                  tick={{ fontSize: 10 }}
                  width={48}
                  label={{
                    value: metric_label,
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 10,
                  }}
                />
                <Tooltip
                  formatter={(value) => [fmt(Number(value))]}
                  labelFormatter={(v) => `${v}% of data`}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line
                  type="monotone"
                  dataKey="train"
                  name="Train score"
                  stroke="hsl(var(--primary))"
                  strokeDasharray="4 2"
                  dot={{ r: 3 }}
                  strokeWidth={1.5}
                />
                <Line
                  type="monotone"
                  dataKey="val"
                  name="Validation score"
                  stroke="#6366f1"
                  dot={{ r: 3 }}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Best score + plateau */}
          <div className="flex gap-2 flex-wrap">
            <div className="rounded bg-indigo-50 border border-indigo-200 px-3 py-1.5 text-xs">
              <span className="text-muted-foreground">Best {metric_label}</span>
              <p className="font-semibold text-indigo-800">{fmt(best_val_score)}</p>
            </div>
            {converged && plateau_pct != null && (
              <div className="rounded bg-emerald-50 border border-emerald-200 px-3 py-1.5 text-xs">
                <span className="text-muted-foreground">Converged at</span>
                <p className="font-semibold text-emerald-800">{plateau_pct}% of data</p>
              </div>
            )}
          </div>

          {/* Recommendation */}
          <div className="rounded border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Recommendation: </span>
            {recommendation}
          </div>
        </CardContent>
      </Card>
    </figure>
  )
}
