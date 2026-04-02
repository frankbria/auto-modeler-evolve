"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ClassDistributionEntry, ClassImbalanceResult } from "@/lib/types"

interface ImbalanceCardProps {
  data: ClassImbalanceResult
  selectedStrategy: string | null
  onStrategyChange: (strategy: string | null) => void
}

const STRATEGY_INFO: Record<
  string,
  { label: string; description: string; color: string }
> = {
  class_weight: {
    label: "Class Weighting",
    description:
      "Tells the model to pay more attention to the minority class during training. Fast, no new data created. Best starting point.",
    color: "bg-blue-100 border-blue-300 text-blue-800",
  },
  smote: {
    label: "SMOTE Oversampling",
    description:
      "Creates synthetic minority examples by interpolating between real ones. Balances training data without simply duplicating rows.",
    color: "bg-violet-100 border-violet-300 text-violet-800",
  },
  threshold: {
    label: "Threshold Tuning",
    description:
      "Trains normally, then finds the decision threshold that maximises F1 score on held-out data. Good when you want to control the precision/recall trade-off.",
    color: "bg-amber-100 border-amber-300 text-amber-800",
  },
}

function DistributionBar({ entries }: { entries: ClassDistributionEntry[] }) {
  const maxRatio = Math.max(...entries.map((e) => e.ratio))
  return (
    <div
      className="space-y-1.5"
      role="list"
      aria-label="Class distribution"
    >
      {entries.map((entry) => {
        const pct = Math.round(entry.ratio * 100)
        const barWidth = maxRatio > 0 ? Math.round((entry.ratio / maxRatio) * 100) : 0
        const isMinority = entry.ratio < 0.2
        return (
          <div key={entry.class} className="flex items-center gap-2" role="listitem">
            <span className="w-28 truncate text-xs font-medium text-foreground" title={entry.class}>
              {entry.class}
            </span>
            <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden" aria-hidden="true">
              <div
                className={`h-full rounded-full ${isMinority ? "bg-rose-500" : "bg-primary"}`}
                style={{ width: `${barWidth}%` }}
              />
            </div>
            <span className={`w-12 text-right text-xs font-semibold ${isMinority ? "text-rose-600" : "text-foreground"}`}>
              {pct}%
            </span>
            <span className="w-16 text-right text-xs text-muted-foreground">
              {entry.count.toLocaleString()} rows
            </span>
            {isMinority && (
              <Badge className="bg-rose-100 text-rose-700 border-rose-300 text-[10px] px-1 py-0">
                minority
              </Badge>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function ImbalanceCard({
  data,
  selectedStrategy,
  onStrategyChange,
}: ImbalanceCardProps) {
  if (!data.is_imbalanced) {
    return (
      <Card className="border-emerald-300 bg-emerald-50/50 mb-4">
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className="text-sm font-semibold text-emerald-800 flex items-center gap-2">
            <span aria-hidden="true">✅</span>
            <span>Balanced Classes</span>
            <Badge className="bg-emerald-100 text-emerald-700 border-emerald-300 ml-1">No action needed</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3">
          {data.class_distribution.length > 0 && (
            <DistributionBar entries={data.class_distribution} />
          )}
          <p className="text-xs text-muted-foreground mt-2">{data.explanation}</p>
        </CardContent>
      </Card>
    )
  }

  const recommended = data.recommended_strategy
  const strategies = Object.keys(STRATEGY_INFO) as Array<keyof typeof STRATEGY_INFO>

  return (
    <Card className="border-rose-300 bg-rose-50/40 mb-4">
      <CardHeader className="pb-2 pt-3 px-4">
        <CardTitle className="text-sm font-semibold text-rose-800 flex items-center gap-2">
          <span aria-hidden="true">⚠️</span>
          <span>Class Imbalance Detected</span>
          <Badge className="bg-rose-100 text-rose-700 border-rose-300 ml-1">
            {Math.round((data.minority_ratio ?? 0) * 100)}% minority
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4 space-y-3">
        <DistributionBar entries={data.class_distribution} />

        <p className="text-xs text-muted-foreground leading-relaxed">{data.explanation}</p>

        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-foreground">Choose an imbalance strategy:</p>
          {strategies.map((strategy) => {
            const info = STRATEGY_INFO[strategy]
            const isSelected = selectedStrategy === strategy
            const isRec = strategy === recommended
            return (
              <button
                key={strategy}
                type="button"
                onClick={() => onStrategyChange(isSelected ? null : strategy)}
                aria-pressed={isSelected}
                className={`w-full text-left rounded-lg border px-3 py-2 text-xs transition-all
                  ${isSelected
                    ? `${info.color} ring-2 ring-offset-1 ring-current font-semibold`
                    : "border-muted bg-background hover:bg-muted/50 text-foreground"
                  }`}
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className="font-semibold">{info.label}</span>
                  <div className="flex items-center gap-1">
                    {isRec && (
                      <Badge className="bg-sky-100 text-sky-700 border-sky-300 text-[10px] px-1 py-0">
                        recommended
                      </Badge>
                    )}
                    {isSelected && (
                      <Badge className="bg-primary text-primary-foreground text-[10px] px-1 py-0">
                        selected
                      </Badge>
                    )}
                  </div>
                </div>
                <p className="text-muted-foreground font-normal leading-relaxed">
                  {info.description}
                </p>
              </button>
            )
          })}
        </div>

        {selectedStrategy && (
          <p className="text-xs text-muted-foreground italic">
            Strategy will apply to all selected algorithms. Click the selected strategy again to remove it.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
