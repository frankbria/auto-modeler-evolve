'use client'

import { Badge } from '@/components/ui/badge'
import type {
  ConfidenceHeatmapResult,
  ConfidenceHeatmapCell,
  ConfidenceHeatmapVerdict,
} from '@/lib/types'

function verdictBorderClass(verdict: ConfidenceHeatmapVerdict): string {
  switch (verdict) {
    case 'gaps_found': return 'border-rose-400'
    case 'uniform_high': return 'border-emerald-400'
    case 'uniform_moderate': return 'border-amber-400'
    default: return 'border-muted'
  }
}

function verdictBadgeClass(verdict: ConfidenceHeatmapVerdict): string {
  switch (verdict) {
    case 'gaps_found': return 'bg-rose-100 text-rose-800'
    case 'uniform_high': return 'bg-emerald-100 text-emerald-800'
    case 'uniform_moderate': return 'bg-amber-100 text-amber-800'
    default: return 'bg-muted text-muted-foreground'
  }
}

function verdictLabel(verdict: ConfidenceHeatmapVerdict): string {
  switch (verdict) {
    case 'gaps_found': return 'Gaps Found'
    case 'uniform_high': return 'Uniformly High'
    case 'uniform_moderate': return 'Moderate'
    case 'insufficient_data': return 'Insufficient Data'
    case 'no_data': return 'No Data'
  }
}

function cellBgClass(avgConf: number, hasSamples: boolean): string {
  if (!hasSamples) return 'bg-slate-50 text-slate-300'
  if (avgConf >= 0.85) return 'bg-emerald-100 text-emerald-900'
  if (avgConf >= 0.70) return 'bg-lime-100 text-lime-900'
  if (avgConf >= 0.55) return 'bg-amber-100 text-amber-900'
  return 'bg-rose-100 text-rose-900'
}

interface HeatmapGridProps {
  result: ConfidenceHeatmapResult
}

function HeatmapGrid({ result }: HeatmapGridProps) {
  const { x_labels, y_labels, cells } = result

  if (x_labels.length === 0 || y_labels.length === 0) return null

  // Build lookup map: (xi, yi) -> cell
  const cellMap = new Map<string, ConfidenceHeatmapCell>()
  for (const cell of cells) {
    cellMap.set(`${cell.x_idx},${cell.y_idx}`, cell)
  }

  const colCount = x_labels.length

  return (
    <div
      className="overflow-x-auto"
      data-testid="heatmap-grid"
      aria-label={`Confidence heatmap: ${result.feature_x} × ${result.feature_y}`}
    >
      {/* Column header row */}
      <div
        className="grid gap-1 mb-1"
        style={{ gridTemplateColumns: `80px repeat(${colCount}, minmax(60px, 1fr))` }}
      >
        <div className="text-xs text-muted-foreground font-medium px-1 pt-1 truncate">
          {result.feature_y} ↓ / {result.feature_x} →
        </div>
        {x_labels.map((xl) => (
          <div
            key={xl}
            className="text-xs text-center font-medium text-slate-600 px-1 truncate"
            title={xl}
          >
            {xl}
          </div>
        ))}
      </div>

      {/* Data rows */}
      {y_labels.map((yl, yi) => (
        <div
          key={yl}
          className="grid gap-1 mb-1"
          style={{ gridTemplateColumns: `80px repeat(${colCount}, minmax(60px, 1fr))` }}
        >
          <div
            className="text-xs font-medium text-slate-600 px-1 py-2 flex items-center truncate"
            title={yl}
          >
            {yl}
          </div>
          {x_labels.map((xl, xi) => {
            const cell = cellMap.get(`${xi},${yi}`)
            const bg = cell ? cellBgClass(cell.avg_confidence, true) : 'bg-slate-50 text-slate-300'
            return (
              <div
                key={xl}
                data-testid={`cell-${xi}-${yi}`}
                className={`rounded p-1.5 text-center text-xs font-medium ${bg}`}
                title={
                  cell
                    ? `${result.feature_x}=${xl}, ${result.feature_y}=${yl}: ${(cell.avg_confidence * 100).toFixed(0)}% avg confidence, ${cell.n_samples} samples`
                    : `${result.feature_x}=${xl}, ${result.feature_y}=${yl}: no data`
                }
              >
                {cell ? (
                  <span>
                    <span className="block">{(cell.avg_confidence * 100).toFixed(0)}%</span>
                    <span className="text-[10px] opacity-70">n={cell.n_samples}</span>
                  </span>
                ) : (
                  <span className="opacity-40">—</span>
                )}
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}

interface ConfidenceHeatmapCardProps {
  result: ConfidenceHeatmapResult
}

export function ConfidenceHeatmapCard({ result }: ConfidenceHeatmapCardProps) {
  const borderClass = verdictBorderClass(result.verdict)
  const badgeClass = verdictBadgeClass(result.verdict)
  const label = verdictLabel(result.verdict)

  return (
    <figure
      data-testid="confidence-heatmap-card"
      className={`rounded-lg border-2 ${borderClass} bg-card p-4 space-y-3`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 flex-wrap">
        <span aria-hidden="true">🗺️</span>
        <h3 className="font-semibold text-sm">Confidence Heatmap</h3>
        <Badge className={badgeClass} data-testid="verdict-badge">
          {label}
        </Badge>
        <Badge variant="outline" data-testid="feature-pair-badge">
          {result.feature_x} × {result.feature_y}
        </Badge>
        <Badge variant="outline" data-testid="n-samples-badge">
          {result.n_samples.toLocaleString()} predictions
        </Badge>
        {result.n_cells_populated > 0 && (
          <Badge variant="outline" data-testid="cells-badge">
            {result.n_cells_populated}/{result.n_cells_total} cells populated
          </Badge>
        )}
      </div>

      {/* Summary */}
      <p className="text-sm text-muted-foreground" data-testid="summary-text">
        {result.summary}
      </p>

      {/* Stats row */}
      {result.overall_mean !== null && (
        <div className="grid grid-cols-3 gap-2 text-xs" data-testid="stats-row">
          <div className="bg-slate-50 rounded p-2 text-center">
            <span className="block font-medium text-slate-800">
              {result.overall_min !== null ? `${(result.overall_min * 100).toFixed(0)}%` : '—'}
            </span>
            <span className="text-muted-foreground">Min</span>
          </div>
          <div className="bg-slate-50 rounded p-2 text-center">
            <span className="block font-medium text-slate-800">
              {`${(result.overall_mean * 100).toFixed(0)}%`}
            </span>
            <span className="text-muted-foreground">Mean</span>
          </div>
          <div className="bg-slate-50 rounded p-2 text-center">
            <span className="block font-medium text-slate-800">
              {result.overall_max !== null ? `${(result.overall_max * 100).toFixed(0)}%` : '—'}
            </span>
            <span className="text-muted-foreground">Max</span>
          </div>
        </div>
      )}

      {/* Heatmap grid */}
      {result.cells.length > 0 && (
        <HeatmapGrid result={result} />
      )}

      {/* Color legend */}
      {result.cells.length > 0 && (
        <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap" data-testid="legend">
          <span>Confidence:</span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-rose-200" />
            &lt;55%
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-amber-200" />
            55–70%
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-lime-200" />
            70–85%
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-emerald-200" />
            ≥85%
          </span>
        </div>
      )}

      {/* Low confidence zones callout */}
      {result.low_confidence_zones.length > 0 && (
        <div
          className="bg-rose-50 border border-rose-200 rounded p-3 space-y-1"
          role="alert"
          data-testid="low-confidence-zones"
        >
          <p className="text-xs font-semibold text-rose-800">
            ⚠ Low-confidence zones ({result.low_confidence_zones.length})
          </p>
          {result.low_confidence_zones.map((zone, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-rose-700">
              <span>
                {result.feature_x}=<strong>{zone.x_label}</strong> +{' '}
                {result.feature_y}=<strong>{zone.y_label}</strong>
              </span>
              <span className="text-rose-500">→</span>
              <span className="font-medium">{(zone.avg_confidence * 100).toFixed(0)}% avg</span>
              <span className="text-rose-400">({zone.n_samples} predictions)</span>
            </div>
          ))}
        </div>
      )}

      {/* No data / insufficient data states */}
      {(result.verdict === 'no_data' || result.verdict === 'insufficient_data') && (
        <p className="text-xs text-muted-foreground italic" data-testid="empty-state">
          {result.summary}
        </p>
      )}

      <figcaption className="sr-only">
        Confidence heatmap for {result.feature_x} × {result.feature_y}.
        Verdict: {label}. {result.summary}
        {result.low_confidence_zones.length > 0 &&
          ` ${result.low_confidence_zones.length} low-confidence zone(s) detected.`}
      </figcaption>

      <p className="text-xs text-muted-foreground italic">
        Each cell shows the average model confidence for that feature value combination.
        Low-confidence zones may benefit from additional training data or feature engineering.
      </p>
    </figure>
  )
}
