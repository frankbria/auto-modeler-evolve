"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { FeatureSweepEntry, FeatureSweepResult } from "@/lib/types"

interface FeatureSweepCardProps {
  result: FeatureSweepResult
}

function formatValue(v: number | string, type: "numeric" | "categorical"): string {
  if (type === "categorical") return String(v)
  const n = Number(v)
  if (isNaN(n)) return String(v)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) >= 1) return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return n.toPrecision(3)
}

function formatPred(v: number | string): string {
  const n = Number(v)
  if (isNaN(n)) return String(v)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) >= 1) return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return n.toPrecision(3)
}

function DeltaBar({ delta, maxDelta }: { delta: number; maxDelta: number }) {
  const pct = maxDelta > 0 ? Math.round((delta / maxDelta) * 100) : 0
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-teal-500 rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground w-12 text-right shrink-0">
        {formatPred(delta)}
      </span>
    </div>
  )
}

function FeatureRow({
  entry,
  maxDelta,
  isClassification,
}: {
  entry: FeatureSweepEntry
  maxDelta: number
  isClassification: boolean
}) {
  const bestLabel = formatValue(entry.best_value, entry.feature_type)
  const worstLabel = formatValue(entry.worst_value, entry.feature_type)
  const bestPred = isClassification
    ? `${(Number(entry.best_prediction) * 100).toFixed(0)}%`
    : formatPred(entry.best_prediction)
  const worstPred = isClassification
    ? `${(Number(entry.worst_prediction) * 100).toFixed(0)}%`
    : formatPred(entry.worst_prediction)

  return (
    <div className="py-2 border-b border-border last:border-0">
      <div className="flex items-center gap-2 mb-1">
        <Badge variant="outline" className="text-xs font-mono w-6 h-5 flex items-center justify-center p-0 shrink-0">
          {entry.rank}
        </Badge>
        <span className="text-sm font-medium truncate">{entry.feature_name.replace(/_/g, " ")}</span>
        <Badge variant="secondary" className="text-xs shrink-0">
          {entry.feature_type}
        </Badge>
      </div>
      <div className="flex gap-3 text-xs text-muted-foreground mb-1 pl-8">
        <span>
          <span className="text-emerald-600 font-medium">{bestLabel}</span>
          {" → "}
          <span className="text-emerald-600">{bestPred}</span>
        </span>
        <span className="text-muted-foreground">|</span>
        <span>
          <span className="text-rose-500 font-medium">{worstLabel}</span>
          {" → "}
          <span className="text-rose-500">{worstPred}</span>
        </span>
      </div>
      <div className="pl-8">
        <DeltaBar delta={entry.delta} maxDelta={maxDelta} />
      </div>
    </div>
  )
}

export function FeatureSweepCard({ result }: FeatureSweepCardProps) {
  const isMax = result.direction === "maximize"
  const isClassification = result.problem_type === "classification"
  const topFeatures = result.features.slice(0, 10)
  const maxDelta = topFeatures.length > 0 ? topFeatures[0].delta : 0

  const optimalPredDisplay = isClassification
    ? `${(Number(result.optimal_prediction) * 100).toFixed(1)}%`
    : formatPred(result.optimal_prediction)

  const optimalConfigEntries = Object.entries(result.optimal_config).slice(0, 8)

  return (
    <Card className="border-teal-500 border-2 mt-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2 flex-wrap">
          <span>🔭</span>
          <span>Feature Impact Sweep</span>
          <Badge className={isMax ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"}>
            {isMax ? "Maximize" : "Minimize"}
          </Badge>
          <Badge variant="secondary">{result.n_features} features</Badge>
          <Badge variant="outline">{result.target_column?.replace(/_/g, " ")}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Optimal configuration */}
        <div className="bg-muted/50 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Optimal {isMax ? "Maximum" : "Minimum"} Prediction
            </span>
          </div>
          <div className="text-2xl font-bold text-teal-600 mb-2">{optimalPredDisplay}</div>
          <div className="flex flex-wrap gap-1">
            {optimalConfigEntries.map(([feat, val]) => (
              <Badge key={feat} variant="secondary" className="text-xs font-mono">
                {feat.replace(/_/g, " ")}={formatValue(val, typeof val === "number" ? "numeric" : "categorical")}
              </Badge>
            ))}
            {Object.keys(result.optimal_config).length > 8 && (
              <Badge variant="outline" className="text-xs">
                +{Object.keys(result.optimal_config).length - 8} more
              </Badge>
            )}
          </div>
        </div>

        {/* Feature impact ranking */}
        {topFeatures.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
              Impact Ranking
              <span className="ml-2 font-normal normal-case">
                (emerald = {isMax ? "highest" : "lowest"} prediction value · rose = {isMax ? "lowest" : "highest"})
              </span>
            </p>
            <div>
              {topFeatures.map((entry) => (
                <FeatureRow
                  key={entry.feature_name}
                  entry={entry}
                  maxDelta={maxDelta}
                  isClassification={isClassification}
                />
              ))}
            </div>
          </div>
        )}

        {topFeatures.length === 0 && (
          <p className="text-sm text-muted-foreground italic">{result.summary}</p>
        )}

        <p className="text-xs text-muted-foreground italic border-t border-border pt-2">
          Each feature is swept independently while others are held at training-data defaults.
          Setting all features to their optimal values together yields the prediction shown above.
        </p>
      </CardContent>
    </Card>
  )
}
