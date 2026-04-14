"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ClassDistributionEntry, ClassImbalanceResult } from "@/lib/types"

interface ClassImbalanceChatCardProps {
  data: ClassImbalanceResult
  onSwitchTab?: (tab: string) => void
}

const STRATEGY_LABELS: Record<string, { label: string; color: string; hint: string }> = {
  class_weight: {
    label: "Class Weighting",
    color: "bg-blue-100 text-blue-800 border-blue-300",
    hint: "Tells the model to pay more attention to the minority class. Fast, effective, no new data needed.",
  },
  smote: {
    label: "SMOTE Oversampling",
    color: "bg-violet-100 text-violet-800 border-violet-300",
    hint: "Creates synthetic minority examples by interpolating between real ones. Best for severe imbalance.",
  },
  threshold: {
    label: "Threshold Tuning",
    color: "bg-amber-100 text-amber-800 border-amber-300",
    hint: "Trains normally, then finds the decision boundary that maximises F1. Good when precision/recall balance matters.",
  },
  none: {
    label: "No action needed",
    color: "bg-emerald-100 text-emerald-800 border-emerald-300",
    hint: "Classes are balanced — standard training will work well.",
  },
}

function DistributionBar({ entries }: { entries: ClassDistributionEntry[] }) {
  const maxRatio = Math.max(...entries.map((e) => e.ratio))
  return (
    <div className="space-y-1.5" role="list" aria-label="Class distribution">
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

export function ClassImbalanceChatCard({ data, onSwitchTab }: ClassImbalanceChatCardProps) {
  const strategy = data.recommended_strategy
  const strategyInfo = STRATEGY_LABELS[strategy] ?? STRATEGY_LABELS.none

  if (data.problem_type !== "classification") {
    return (
      <Card
        className="border-muted bg-muted/20 mb-2"
        role="region"
        aria-label="Class imbalance check"
      >
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2">
            <span aria-hidden="true">⚖️</span>
            <span>Class Imbalance Check</span>
            <Badge className="bg-muted text-muted-foreground border-muted ml-1">Regression — N/A</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3">
          <p className="text-xs text-muted-foreground">{data.explanation}</p>
        </CardContent>
      </Card>
    )
  }

  if (!data.is_imbalanced) {
    return (
      <Card
        className="border-emerald-300 bg-emerald-50/50 mb-2"
        role="region"
        aria-label="Class imbalance check"
      >
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className="text-sm font-semibold text-emerald-800 flex items-center gap-2">
            <span aria-hidden="true">✅</span>
            <span>Balanced Classes</span>
            <Badge className="bg-emerald-100 text-emerald-700 border-emerald-300 ml-1">No action needed</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3 space-y-2">
          {data.class_distribution.length > 0 && (
            <DistributionBar entries={data.class_distribution} />
          )}
          <p className="text-xs text-muted-foreground">{data.explanation}</p>
        </CardContent>
      </Card>
    )
  }

  const minorityPct = data.minority_ratio != null ? Math.round(data.minority_ratio * 100) : 0

  return (
    <Card
      className="border-rose-300 bg-rose-50/40 mb-2"
      role="region"
      aria-label="Class imbalance check"
    >
      <CardHeader className="pb-2 pt-3 px-4">
        <CardTitle className="text-sm font-semibold text-rose-800 flex items-center gap-2">
          <span aria-hidden="true">⚠️</span>
          <span>Class Imbalance Detected</span>
          <Badge className="bg-rose-100 text-rose-700 border-rose-300 ml-1">
            {minorityPct}% minority
          </Badge>
          {data.target_column && (
            <Badge className="bg-muted text-muted-foreground border-muted ml-1 text-[10px]">
              target: {data.target_column}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4 space-y-3">
        {data.class_distribution.length > 0 && (
          <DistributionBar entries={data.class_distribution} />
        )}

        <p className="text-xs text-muted-foreground leading-relaxed">{data.explanation}</p>

        <div className="rounded-lg border px-3 py-2 text-xs space-y-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-foreground">Recommended strategy:</span>
            <Badge className={`${strategyInfo.color} text-[10px] px-1.5 py-0`}>
              {strategyInfo.label}
            </Badge>
          </div>
          <p className="text-muted-foreground leading-relaxed">{strategyInfo.hint}</p>
        </div>

        {onSwitchTab && (
          <button
            type="button"
            onClick={() => onSwitchTab("models")}
            className="text-xs text-primary underline underline-offset-2 hover:text-primary/80 transition-colors"
          >
            Go to Models tab to apply this strategy before training →
          </button>
        )}
      </CardContent>
    </Card>
  )
}
