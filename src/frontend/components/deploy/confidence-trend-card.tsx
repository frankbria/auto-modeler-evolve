"use client"

import type { ConfidenceTrendResult } from "@/lib/types"
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

interface ConfidenceTrendCardProps {
  result: ConfidenceTrendResult
}

const DIRECTION_CONFIG = {
  improving: { label: "↑ Improving", badge: "bg-green-100 text-green-800 border-green-200" },
  declining: { label: "↓ Declining", badge: "bg-red-100 text-red-800 border-red-200" },
  stable:    { label: "→ Stable",    badge: "bg-gray-100 text-gray-700 border-gray-200" },
}

const BORDER_COLOR: Record<string, string> = {
  improving: "border-green-200",
  declining: "border-red-200",
  stable:    "border-teal-200",
}

export function ConfidenceTrendCard({ result }: ConfidenceTrendCardProps) {
  if (!result.has_data) {
    return (
      <Card
        className="border-teal-200"
        role="region"
        aria-label="Confidence trend: no data yet"
      >
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">📉</span> Confidence Trend
          </CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground italic">
          No confidence data yet — trend will appear after the first predictions.
        </CardContent>
      </Card>
    )
  }

  const dir = result.trend_direction as "improving" | "declining" | "stable"
  const cfg = DIRECTION_CONFIG[dir] ?? DIRECTION_CONFIG.stable
  const borderColor = BORDER_COLOR[dir] ?? "border-teal-200"

  const rateLabel =
    result.trend_rate_per_day !== null
      ? `${result.trend_rate_per_day > 0 ? "+" : ""}${result.trend_rate_per_day.toFixed(2)}%/day`
      : null

  return (
    <Card
      className={borderColor}
      role="region"
      aria-label={`Confidence trend: ${dir}`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">📉</span> Confidence Trend
          </CardTitle>
          <Badge className={`${cfg.badge} text-[10px]`}>{cfg.label}</Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3 text-xs">
        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="font-semibold text-base">
              {result.overall_avg !== null ? `${result.overall_avg.toFixed(1)}%` : "—"}
            </div>
            <div className="text-muted-foreground">Avg confidence</div>
          </div>
          <div>
            <div className="font-semibold text-base text-green-700">
              {result.peak_value !== null ? `${result.peak_value.toFixed(1)}%` : "—"}
            </div>
            <div className="text-muted-foreground">Peak day</div>
          </div>
          <div>
            <div className="font-semibold text-base text-red-600">
              {result.low_value !== null ? `${result.low_value.toFixed(1)}%` : "—"}
            </div>
            <div className="text-muted-foreground">Low day</div>
          </div>
        </div>

        {/* Rate + sample count */}
        {rateLabel && (
          <div className="flex items-center justify-between text-muted-foreground">
            <span>
              Trend:{" "}
              <strong
                className={`${dir === "improving" ? "text-green-700" : dir === "declining" ? "text-red-600" : "text-foreground"}`}
              >
                {rateLabel}
              </strong>
            </span>
            <span className="text-[10px]">
              {result.sample_count} prediction{result.sample_count !== 1 ? "s" : ""}
            </span>
          </div>
        )}

        {/* Trend sparkline */}
        {result.daily_stats.length > 1 && (
          <figure
            className="h-20"
            aria-label={`Daily confidence trend over ${result.daily_stats.length} days`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={result.daily_stats}>
                <XAxis dataKey="date" hide />
                <YAxis domain={[0, 100]} hide />
                <Tooltip
                  formatter={(v) => [`${Number(v).toFixed(1)}%`, "Avg confidence"]}
                  contentStyle={{ fontSize: "10px" }}
                />
                <Line
                  type="monotone"
                  dataKey="avg_confidence"
                  stroke={dir === "improving" ? "#16a34a" : dir === "declining" ? "#dc2626" : "#0d9488"}
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
            <figcaption className="sr-only">Daily average confidence over time</figcaption>
          </figure>
        )}

        {/* Summary sentence */}
        {result.summary && (
          <p className="text-muted-foreground text-[10px]">{result.summary}</p>
        )}
      </CardContent>
    </Card>
  )
}
