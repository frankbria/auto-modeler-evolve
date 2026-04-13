"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ABTestChatResult, ABVariantMetrics, ABSignificance } from "@/lib/types"

// ---------------------------------------------------------------------------
// ABTestChatCard — inline chat card for A/B test status / action confirmations.
//
// Read-only: shows traffic split, per-variant metrics, and statistical
// significance when a test is active.  Also renders promoted/ended
// confirmations and a "no test" empty state with guidance.
// ---------------------------------------------------------------------------

interface ABTestChatCardProps {
  result: ABTestChatResult
}

function MetricsColumn({
  label,
  color,
  metrics,
}: {
  label: string
  color: "purple" | "amber"
  metrics: ABVariantMetrics
}) {
  const border = color === "purple" ? "border-purple-200" : "border-amber-200"
  const bg = color === "purple" ? "bg-purple-50" : "bg-amber-50"
  const text = color === "purple" ? "text-purple-700" : "text-amber-700"

  return (
    <div className={`rounded border ${border} ${bg} p-2 space-y-0.5`}>
      <p className={`text-xs font-semibold ${text} mb-1`}>{label}</p>
      <dl className="space-y-0.5 text-xs">
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Requests</dt>
          <dd className="font-medium">{metrics.request_count}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Avg confidence</dt>
          <dd className="font-medium">
            {metrics.avg_confidence != null
              ? `${(metrics.avg_confidence * 100).toFixed(1)}%`
              : "—"}
          </dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">p95 latency</dt>
          <dd className="font-medium">
            {metrics.p95_ms != null ? `${metrics.p95_ms}ms` : "—"}
          </dd>
        </div>
        {metrics.avg_prediction != null && (
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Avg prediction</dt>
            <dd className="font-medium">{metrics.avg_prediction.toFixed(2)}</dd>
          </div>
        )}
      </dl>
    </div>
  )
}

function SignificanceRow({ sig }: { sig: ABSignificance }) {
  if (sig.p_value == null) {
    return <p className="text-xs text-muted-foreground italic">{sig.note}</p>
  }
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <Badge
        variant="outline"
        className={
          sig.significant
            ? "bg-green-50 text-green-700 border-green-300"
            : "bg-muted text-muted-foreground"
        }
      >
        {sig.significant ? "Statistically significant" : "Not yet significant"}
      </Badge>
      <span className="text-muted-foreground">
        p&nbsp;=&nbsp;{sig.p_value} · {sig.note}
      </span>
    </div>
  )
}

export function ABTestChatCard({ result }: ABTestChatCardProps) {
  const isStatus = result.action === "status"
  const isPromoted = result.action === "promoted"
  const isEnded = result.action === "ended"
  const isNone = result.action === "none"

  const ariaLabel = isStatus
    ? "A/B test status"
    : isPromoted
    ? "Challenger promoted"
    : isEnded
    ? "A/B test ended"
    : "No active A/B test"

  return (
    <Card
      className="border-purple-300 bg-purple-50/30"
      role="region"
      aria-label={ariaLabel}
    >
      <CardHeader className="pb-2 pt-3 px-4">
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm font-semibold">
          <span aria-hidden="true">⚗️</span>
          {isStatus && "A/B Test Status"}
          {isPromoted && "Challenger Promoted"}
          {isEnded && "A/B Test Ended"}
          {isNone && "No Active A/B Test"}
          {isStatus && result.is_active && (
            <Badge className="ml-auto text-xs bg-purple-100 text-purple-800 border-purple-300" variant="outline">
              Live
            </Badge>
          )}
          {isPromoted && (
            <Badge className="text-xs bg-green-100 text-green-700 border-green-300" variant="outline">
              Promoted ✓
            </Badge>
          )}
          {isEnded && (
            <Badge className="text-xs bg-muted text-muted-foreground" variant="outline">
              Ended
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="px-4 pb-3 space-y-3 text-sm">
        {/* Status view: traffic split + metrics + significance */}
        {isStatus &&
          result.champion_split_pct != null &&
          result.challenger_split_pct != null && (
            <>
              {/* Traffic split bar */}
              <div>
                <div className="flex justify-between text-xs mb-1 font-medium">
                  <span>Champion ({result.champion_algorithm ?? "Model"})</span>
                  <span>Challenger ({result.challenger_algorithm ?? "Model"})</span>
                </div>
                <div className="flex h-2 rounded overflow-hidden">
                  <div
                    className="bg-purple-500"
                    style={{ width: `${result.champion_split_pct}%` }}
                    title={`Champion: ${result.champion_split_pct}%`}
                  />
                  <div
                    className="bg-amber-400"
                    style={{ width: `${result.challenger_split_pct}%` }}
                    title={`Challenger: ${result.challenger_split_pct}%`}
                  />
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-0.5">
                  <span>{result.champion_split_pct}% of traffic</span>
                  <span>{result.challenger_split_pct}% of traffic</span>
                </div>
              </div>

              {/* Per-variant metrics */}
              {result.champion_metrics && result.challenger_metrics && (
                <div className="grid grid-cols-2 gap-2">
                  <MetricsColumn
                    label="Champion"
                    color="purple"
                    metrics={result.champion_metrics}
                  />
                  <MetricsColumn
                    label="Challenger"
                    color="amber"
                    metrics={result.challenger_metrics}
                  />
                </div>
              )}

              {/* Significance */}
              {result.significance && <SignificanceRow sig={result.significance} />}

              <p className="text-xs text-muted-foreground">
                To promote the challenger or end the test, ask me or use the Deployment panel.
              </p>
            </>
          )}

        {/* Promoted / ended confirmation */}
        {(isPromoted || isEnded) && (
          <p className="text-sm text-foreground">{result.summary}</p>
        )}
        {isPromoted && (
          <p className="text-xs text-muted-foreground">
            Your prediction URL is unchanged — any links you shared continue to work.
          </p>
        )}
        {isEnded && (
          <p className="text-xs text-muted-foreground">
            The champion model handles all prediction traffic. Start a new A/B test anytime.
          </p>
        )}

        {/* No active test */}
        {isNone && (
          <>
            <p className="text-sm text-muted-foreground">{result.summary}</p>
            <p className="text-xs text-muted-foreground">
              Train a second model, deploy it, then open the Deployment panel to configure traffic splitting.
            </p>
          </>
        )}
      </CardContent>
    </Card>
  )
}
