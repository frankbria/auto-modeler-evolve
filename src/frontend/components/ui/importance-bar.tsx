"use client"

/**
 * Shared importance bar component used in model-card-view and feature-suggestions.
 * Accepts `importance` in the range 0..1 (already normalized to max=1).
 * The caller is responsible for normalization.
 */

interface ImportanceBarProps {
  /** Normalized importance in the range 0..1 (where 1 = full bar width). */
  importance: number
  feature?: string
  rank?: number
  /** Override the displayed label text (defaults to widthPct%). */
  label?: string
}

export function ImportanceBar({ importance, feature, rank, label }: ImportanceBarProps) {
  const widthPct = Math.round(Math.min(Math.max(importance * 100, 0), 100))
  const displayLabel = label ?? `${widthPct}%`
  const fillStyle = rank
    ? { backgroundColor: `hsl(${220 - rank * 20}, 70%, ${55 + rank * 5}%)` }
    : undefined

  return (
    <div className="flex items-center gap-2 text-xs" data-testid="importance-bar">
      {rank !== undefined && (
        <span className="w-5 text-muted-foreground font-mono text-right">{rank}</span>
      )}
      {feature !== undefined && (
        <span className="flex-1 min-w-0 truncate">{feature}</span>
      )}
      <div className="w-28 flex-shrink-0 flex items-center gap-1">
        <div className="flex-1 bg-muted rounded-full h-2">
          <div
            className="h-2 rounded-full bg-primary transition-all"
            style={{ width: `${widthPct}%`, ...fillStyle }}
          />
        </div>
        <span className="w-10 text-right text-muted-foreground">
          {displayLabel}
        </span>
      </div>
    </div>
  )
}
