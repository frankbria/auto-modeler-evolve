"use client"

import { Badge } from "@/components/ui/badge"
import type { PairCorrelationResult } from "@/lib/types"

interface PairCorrelationCardProps {
  result: PairCorrelationResult
}

function strengthColor(strength: string): string {
  switch (strength) {
    case "very strong":
      return "bg-emerald-100 text-emerald-800 border-emerald-300"
    case "strong":
      return "bg-blue-100 text-blue-800 border-blue-300"
    case "moderate":
      return "bg-amber-100 text-amber-800 border-amber-300"
    case "weak":
      return "bg-orange-100 text-orange-800 border-orange-300"
    default:
      return "bg-muted text-muted-foreground"
  }
}

function directionColor(direction: string, r: number | null): string {
  if (r === null) return "text-muted-foreground"
  return direction === "positive" ? "text-blue-600" : "text-rose-600"
}

export function PairCorrelationCard({ result }: PairCorrelationCardProps) {
  const hasData = result.r !== null
  const absR = hasData ? Math.abs(result.r!) : 0
  const barWidth = Math.round(absR * 100)

  return (
    <div
      className="rounded-lg border-2 border-violet-300 bg-card p-4 mt-2"
      role="region"
      aria-label={`Correlation analysis between ${result.col1} and ${result.col2}`}
    >
      <div className="flex items-center gap-2 mb-1 flex-wrap">
        <span aria-hidden="true" className="text-violet-600 font-bold">∼</span>
        <span className="font-semibold text-sm">
          Correlation:{" "}
          <span className="font-mono">{result.col1.replace(/_/g, " ")}</span>
          {" "}vs{" "}
          <span className="font-mono">{result.col2.replace(/_/g, " ")}</span>
        </span>
        {hasData && (
          <Badge className={strengthColor(result.strength)}>
            {result.strength}
          </Badge>
        )}
        {hasData && (
          <Badge
            variant="secondary"
            className={`text-xs ${result.direction === "positive" ? "text-blue-700" : "text-rose-700"}`}
          >
            {result.direction}
          </Badge>
        )}
      </div>

      <p className="text-xs text-muted-foreground mb-3">
        Based on {result.n.toLocaleString()} paired observations
      </p>

      {hasData ? (
        <>
          {/* r value display */}
          <div className="flex items-center gap-4 mb-3">
            <div>
              <span className="text-xs text-muted-foreground uppercase tracking-wide">Pearson r</span>
              <p
                className={`text-2xl font-bold tabular-nums ${directionColor(result.direction, result.r)}`}
                aria-label={`Pearson r = ${result.r}`}
              >
                {result.r! >= 0 ? "+" : ""}{result.r!.toFixed(3)}
              </p>
            </div>
            <div className="flex-1">
              <div
                className="h-3 w-full rounded-full bg-muted overflow-hidden"
                aria-label={`Correlation strength bar: ${barWidth}%`}
              >
                <div
                  className={`h-full rounded-full transition-all ${result.direction === "positive" ? "bg-blue-500" : "bg-rose-500"}`}
                  style={{ width: `${barWidth}%` }}
                  aria-hidden="true"
                />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {barWidth}% of maximum possible correlation
              </p>
            </div>
          </div>

          {/* p-value and significance */}
          {result.p_value !== null && (
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-muted-foreground">p-value:</span>
              <span className="text-xs font-mono tabular-nums">
                {result.p_value < 0.001 ? "< 0.001" : result.p_value.toFixed(4)}
              </span>
              <Badge variant="outline" className="text-xs">
                {result.significant}
              </Badge>
            </div>
          )}

          {/* Plain-English interpretation */}
          {result.interpretation && (
            <p className="text-xs text-foreground mb-2 bg-muted/40 rounded p-2">
              {result.interpretation}
            </p>
          )}
        </>
      ) : (
        <p className="text-sm text-muted-foreground italic">{result.summary}</p>
      )}

      <p className="mt-2 text-xs text-muted-foreground border-t border-border/50 pt-2">
        {result.summary}
      </p>
    </div>
  )
}
