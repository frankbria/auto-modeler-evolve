"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type {
  LowAccuracyGuidanceResult,
  LowAccuracyGuidanceTip,
} from "@/lib/types"

interface LowAccuracyGuidanceCardProps {
  result: LowAccuracyGuidanceResult
  onActionClick?: (action: string) => void
}

function TipRow({
  tip,
  index,
  onActionClick,
}: {
  tip: LowAccuracyGuidanceTip
  index: number
  onActionClick?: (action: string) => void
}) {
  return (
    <div
      className="rounded-md border border-rose-200 bg-rose-50/50 p-3 space-y-1.5"
      data-testid={`guidance-tip-${index}`}
    >
      <div className="flex items-center gap-2">
        <span aria-hidden="true" className="text-base">
          {tip.icon}
        </span>
        <span className="text-sm font-semibold text-rose-900">{tip.title}</span>
      </div>
      <p className="text-xs text-rose-800 leading-relaxed">{tip.description}</p>
      {onActionClick && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs text-rose-700 hover:text-rose-900 hover:bg-rose-100 -ml-2"
          onClick={() => onActionClick(tip.action)}
          aria-label={`Try: ${tip.action}`}
          data-testid={`guidance-action-${index}`}
        >
          Try this →
        </Button>
      )}
    </div>
  )
}

export function LowAccuracyGuidanceCard({
  result,
  onActionClick,
}: LowAccuracyGuidanceCardProps) {
  const { metric_name, score_label, is_very_low, tips, summary } = result

  return (
    <figure
      className="rounded-lg border-2 border-rose-300 bg-white p-4 space-y-3 text-sm"
      aria-label="Low accuracy guidance"
      data-testid="low-accuracy-guidance-card"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2">
        <span aria-hidden="true" className="text-xl">
          🔬
        </span>
        <h3 className="font-semibold text-rose-900 text-base">
          Improve Your Model Accuracy
        </h3>
        <Badge
          className={`text-xs border-0 ${
            is_very_low
              ? "bg-rose-100 text-rose-800"
              : "bg-amber-100 text-amber-800"
          }`}
          data-testid="guidance-score-badge"
        >
          {metric_name} {score_label}
        </Badge>
        {is_very_low && (
          <Badge
            className="text-xs bg-rose-200 text-rose-900 border-0"
            data-testid="guidance-very-low-badge"
          >
            Very Low
          </Badge>
        )}
      </div>

      {/* Summary */}
      <p className="text-sm text-muted-foreground leading-relaxed">{summary}</p>

      {/* Tips list */}
      <div className="space-y-2" role="list" aria-label="Improvement suggestions">
        {tips.map((tip, i) => (
          <div key={tip.title} role="listitem">
            <TipRow tip={tip} index={i} onActionClick={onActionClick} />
          </div>
        ))}
      </div>

      {/* Accessibility caption */}
      <figcaption className="sr-only">
        Low accuracy guidance card: {metric_name} {score_label}.{" "}
        {tips.length} improvement tips provided.
      </figcaption>
    </figure>
  )
}
