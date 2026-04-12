"use client"

/**
 * SlaCard — chat-inline latency card for "how fast is my model?" queries.
 *
 * Reuses the SlaData type from deployment-panel but renders in the
 * conversational context with role="region" aria-label for accessibility.
 */

import type { SlaData } from "@/lib/types"
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

interface SlaCardProps {
  sla: SlaData
}

export function SlaCard({ sla }: SlaCardProps) {
  if (sla.sample_count === 0) {
    return (
      <Card
        className="border-sky-200"
        role="region"
        aria-label="Prediction latency: no data yet"
      >
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">⚡</span> Prediction Latency
          </CardTitle>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground italic">
          No timing data yet — latency will appear after the first prediction.
        </CardContent>
      </Card>
    )
  }

  const alertColor = sla.alert ? "border-red-300" : "border-sky-200"

  return (
    <Card
      className={alertColor}
      role="region"
      aria-label={`Prediction latency: ${sla.alert ? "alert" : "healthy"}`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">⚡</span> Prediction Latency
          </CardTitle>
          {sla.alert ? (
            <Badge className="bg-red-100 text-red-800 border-red-200 text-[10px]">
              p95 &gt; 500ms
            </Badge>
          ) : (
            <Badge className="bg-green-100 text-green-800 border-green-200 text-[10px]">
              Healthy
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3 text-xs">
        {/* Percentile grid */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="font-semibold text-base">{sla.p50_ms}ms</div>
            <div className="text-muted-foreground">p50 (median)</div>
          </div>
          <div>
            <div
              className={`font-semibold text-base ${
                sla.p95_ms !== null && sla.p95_ms > 500 ? "text-red-600" : ""
              }`}
            >
              {sla.p95_ms}ms
            </div>
            <div className="text-muted-foreground">p95</div>
          </div>
          <div>
            <div className="font-semibold text-base">{sla.p99_ms}ms</div>
            <div className="text-muted-foreground">p99</div>
          </div>
        </div>

        {/* Average + sparkline */}
        {sla.avg_ms !== null && (
          <div className="flex items-center justify-between text-muted-foreground">
            <span>
              Avg:{" "}
              <strong className="text-foreground">{sla.avg_ms}ms</strong>
            </span>
            <span className="text-[10px]">
              {sla.sample_count} prediction{sla.sample_count !== 1 ? "s" : ""}
            </span>
          </div>
        )}

        {/* Trend sparkline */}
        {sla.latency_by_day.length > 1 && (
          <div
            className="h-16"
            aria-label={`Latency trend over ${sla.latency_by_day.length} days`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sla.latency_by_day}>
                <XAxis dataKey="date" hide />
                <YAxis hide />
                <Tooltip
                  formatter={(v) => [`${v}ms`, "Avg latency"]}
                  contentStyle={{ fontSize: "10px" }}
                />
                <Line
                  type="monotone"
                  dataKey="avg_ms"
                  stroke={sla.alert ? "#dc2626" : "#0ea5e9"}
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Alert message */}
        {sla.alert && sla.alert_message && (
          <p
            className="text-red-700 bg-red-50 rounded p-2"
            role="alert"
          >
            {sla.alert_message}
          </p>
        )}

        {/* SLA target footnote */}
        <p className="text-muted-foreground text-[10px]">
          SLA target: p95 ≤ 500ms. Ask me to &quot;switch to a faster model&quot; if
          you need lower latency.
        </p>
      </CardContent>
    </Card>
  )
}
