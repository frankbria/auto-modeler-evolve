"use client"

import type { FeedbackAccuracyReportResult } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"

interface FeedbackAccuracyCardProps {
  result: FeedbackAccuracyReportResult
}

const VERDICT_CONFIG: Record<
  string,
  { label: string; badge: string; border: string }
> = {
  excellent: {
    label: "✓ Excellent",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    border: "border-emerald-200",
  },
  good: {
    label: "✓ Good",
    badge: "bg-green-100 text-green-800 border-green-200",
    border: "border-green-200",
  },
  moderate: {
    label: "⚠ Moderate",
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    border: "border-amber-200",
  },
  poor: {
    label: "✗ Poor",
    badge: "bg-red-100 text-red-800 border-red-200",
    border: "border-red-200",
  },
}

const TREND_CONFIG: Record<string, { label: string; color: string }> = {
  improving: { label: "↑ Improving", color: "text-emerald-700" },
  stable: { label: "→ Stable", color: "text-gray-600" },
  declining: { label: "↓ Declining", color: "text-red-600" },
}

export function FeedbackAccuracyCard({ result }: FeedbackAccuracyCardProps) {
  if (!result.has_data) {
    return (
      <Card
        className="border-gray-200"
        role="region"
        aria-label="Feedback accuracy: no data yet"
      >
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">🎯</span> Real-World Accuracy
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs text-muted-foreground">
          <p>{result.summary}</p>
          {result.status === "no_feedback" && (
            <p className="italic">
              Tip: After making predictions on the dashboard, record the actual outcomes
              in the Deployment tab to start tracking real-world accuracy.
            </p>
          )}
        </CardContent>
      </Card>
    )
  }

  const verdict = result.verdict ?? "moderate"
  const cfg = VERDICT_CONFIG[verdict] ?? VERDICT_CONFIG.moderate
  const trendCfg =
    TREND_CONFIG[result.trend_direction ?? "stable"] ?? TREND_CONFIG.stable

  const isRegression = result.problem_type === "regression"
  const chartData = result.weekly_trend ?? []

  return (
    <Card
      className={cfg.border}
      role="region"
      aria-label={`Real-world accuracy: ${verdict}`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">🎯</span> Real-World Accuracy
          </CardTitle>
          <Badge className={`${cfg.badge} text-[10px]`}>{cfg.label}</Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3 text-xs">
        {/* Primary metric */}
        <div className="grid grid-cols-3 gap-2 text-center">
          {isRegression ? (
            <>
              <div>
                <div className="font-semibold text-base">
                  {result.mae !== undefined ? result.mae.toFixed(4) : "—"}
                </div>
                <div className="text-muted-foreground">MAE</div>
              </div>
              <div>
                <div className="font-semibold text-base">
                  {result.pct_error !== undefined
                    ? `${result.pct_error.toFixed(1)}%`
                    : "—"}
                </div>
                <div className="text-muted-foreground">% Error</div>
              </div>
              <div>
                <div className="font-semibold text-base">
                  {result.paired_count ?? 0}
                </div>
                <div className="text-muted-foreground">Matched</div>
              </div>
            </>
          ) : (
            <>
              <div>
                <div className="font-semibold text-base">
                  {result.accuracy_pct !== undefined
                    ? `${result.accuracy_pct}%`
                    : "—"}
                </div>
                <div className="text-muted-foreground">Accuracy</div>
              </div>
              <div>
                <div className="font-semibold text-base text-emerald-700">
                  {result.correct_count ?? 0}
                </div>
                <div className="text-muted-foreground">Correct</div>
              </div>
              <div>
                <div className="font-semibold text-base text-red-600">
                  {result.incorrect_count ?? 0}
                </div>
                <div className="text-muted-foreground">Incorrect</div>
              </div>
            </>
          )}
        </div>

        {/* Trend row */}
        {result.trend_direction && (
          <div className="flex items-center justify-between text-muted-foreground">
            <span>
              Weekly trend:{" "}
              <strong className={trendCfg.color}>{trendCfg.label}</strong>
            </span>
            <span className="text-[10px]">
              {result.total_feedback} feedback record
              {result.total_feedback !== 1 ? "s" : ""}
            </span>
          </div>
        )}

        {/* Weekly trend chart */}
        {chartData.length > 1 && (
          <figure
            className="h-20"
            aria-label={`Weekly ${isRegression ? "MAE" : "accuracy"} trend over ${chartData.length} weeks`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis dataKey="week_start" hide />
                <YAxis hide />
                <Tooltip
                  formatter={(v) =>
                    isRegression
                      ? [Number(v).toFixed(4), "MAE"]
                      : [`${Number(v).toFixed(1)}%`, "Accuracy"]
                  }
                  contentStyle={{ fontSize: "10px" }}
                />
                <Line
                  type="monotone"
                  dataKey={isRegression ? "mae" : "accuracy"}
                  stroke={
                    verdict === "excellent" || verdict === "good"
                      ? "#16a34a"
                      : verdict === "poor"
                        ? "#dc2626"
                        : "#f59e0b"
                  }
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
            <figcaption className="sr-only">
              Weekly {isRegression ? "mean absolute error" : "accuracy"} over time
            </figcaption>
          </figure>
        )}

        {/* Verdict message */}
        {result.verdict_msg && (
          <p className="text-muted-foreground text-[10px]">{result.verdict_msg}</p>
        )}

        {/* Summary */}
        {result.summary && (
          <p className="text-muted-foreground text-[10px]">{result.summary}</p>
        )}
      </CardContent>
    </Card>
  )
}
