"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import type { PredictionAnalyticsChatResult } from "@/lib/types"

interface PredictionAnalyticsChatCardProps {
  result: PredictionAnalyticsChatResult
}

export function PredictionAnalyticsChatCard({
  result,
}: PredictionAnalyticsChatCardProps) {
  const {
    total_predictions,
    predictions_last_7_days,
    predictions_last_30_days,
    predictions_today,
    predictions_by_day,
    peak_day,
    class_counts,
    avg_prediction,
    problem_type,
    summary,
  } = result

  const maxCount = Math.max(...predictions_by_day.map((d) => d.count))
  const hasAnyPredictions = maxCount > 0

  const classEntries = class_counts
    ? Object.entries(class_counts).sort(([, a], [, b]) => b - a)
    : null
  const classTotal = classEntries
    ? classEntries.reduce((s, [, n]) => s + n, 0)
    : 0

  const shortDate = (d: string) => {
    const [, m, day] = d.split("-")
    return `${m}/${day}`
  }

  return (
    <figure aria-label="Prediction usage analytics" className="not-prose">
      <Card className="border-sky-500/30">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <span aria-hidden="true">📊</span> Prediction Usage
            </CardTitle>
            <Badge variant="outline" className="text-xs">
              {total_predictions.toLocaleString()} total
            </Badge>
            {problem_type && (
              <Badge variant="outline" className="text-xs capitalize ml-auto">
                {problem_type}
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Summary */}
          <p className="text-xs text-muted-foreground">{summary}</p>

          {/* Stats grid */}
          <div className="grid grid-cols-3 gap-2">
            <div className="rounded bg-sky-50 border border-sky-200 px-3 py-2 text-xs text-center">
              <p className="text-muted-foreground">7-day</p>
              <p className="font-semibold text-sky-800 text-base">
                {predictions_last_7_days.toLocaleString()}
              </p>
            </div>
            <div className="rounded bg-sky-50 border border-sky-200 px-3 py-2 text-xs text-center">
              <p className="text-muted-foreground">30-day</p>
              <p className="font-semibold text-sky-800 text-base">
                {predictions_last_30_days.toLocaleString()}
              </p>
            </div>
            <div className="rounded bg-sky-50 border border-sky-200 px-3 py-2 text-xs text-center">
              <p className="text-muted-foreground">Today</p>
              <p className="font-semibold text-sky-800 text-base">
                {predictions_today.toLocaleString()}
              </p>
            </div>
          </div>

          {/* Daily sparkline — last 14 days */}
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">
              Daily predictions (last 14 days)
            </p>
            {!hasAnyPredictions ? (
              <p className="text-xs text-muted-foreground italic">
                No predictions in this period.
              </p>
            ) : (
              <ResponsiveContainer width="100%" height={80}>
                <BarChart
                  data={predictions_by_day}
                  margin={{ top: 2, right: 4, left: 0, bottom: 2 }}
                >
                  <XAxis
                    dataKey="date"
                    tickFormatter={shortDate}
                    tick={{ fontSize: 9 }}
                    interval={2}
                  />
                  <YAxis hide />
                  <Tooltip
                    formatter={(v) => [v, "predictions"]}
                    labelFormatter={(d) => `Date: ${d}`}
                  />
                  <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                    {predictions_by_day.map((entry) => (
                      <Cell
                        key={entry.date}
                        fill={
                          peak_day && entry.date === peak_day.date
                            ? "#0284c7"
                            : "#bae6fd"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Peak day */}
          {peak_day && peak_day.count > 0 && (
            <div className="rounded bg-sky-50 border border-sky-200 px-3 py-1.5 text-xs">
              <span className="text-muted-foreground">Peak day: </span>
              <span className="font-semibold text-sky-800">
                {peak_day.date}
              </span>
              <span className="text-muted-foreground ml-1">
                ({peak_day.count} predictions)
              </span>
            </div>
          )}

          {/* Classification: top predicted classes */}
          {classEntries && classEntries.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">
                Predicted class distribution
              </p>
              <div className="space-y-1">
                {classEntries.slice(0, 5).map(([cls, cnt]) => {
                  const pct = classTotal > 0 ? Math.round((cnt / classTotal) * 100) : 0
                  return (
                    <div key={cls} className="flex items-center gap-2 text-xs">
                      <span className="w-20 truncate text-muted-foreground font-mono">
                        {cls}
                      </span>
                      <div className="flex-1 bg-muted rounded-full h-2">
                        <div
                          className="bg-sky-500 h-2 rounded-full"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="w-10 text-right text-foreground font-medium">
                        {pct}%
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Regression: average prediction */}
          {avg_prediction !== null && avg_prediction !== undefined && (
            <div className="rounded bg-sky-50 border border-sky-200 px-3 py-1.5 text-xs">
              <span className="text-muted-foreground">Avg prediction (last 30 days): </span>
              <span className="font-semibold text-sky-800">
                {avg_prediction.toLocaleString()}
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </figure>
  )
}
