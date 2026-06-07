'use client'

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { Badge } from '@/components/ui/badge'
import type { SegmentConfTrendResult, SegmentConfTrendEntry, SegmentConfTrendVerdict } from '@/lib/types'

function verdictBorderClass(verdict: SegmentConfTrendVerdict): string {
  switch (verdict) {
    case 'calibration_gap': return 'border-amber-500'
    case 'uniform_decline': return 'border-rose-500'
    case 'uniform_improving': return 'border-emerald-500'
    case 'mixed': return 'border-sky-500'
    case 'stable': return 'border-muted'
    default: return 'border-muted'
  }
}

function verdictBadgeClass(verdict: SegmentConfTrendVerdict): string {
  switch (verdict) {
    case 'calibration_gap': return 'bg-amber-100 text-amber-800'
    case 'uniform_decline': return 'bg-rose-100 text-rose-800'
    case 'uniform_improving': return 'bg-emerald-100 text-emerald-800'
    case 'mixed': return 'bg-sky-100 text-sky-800'
    case 'stable': return 'bg-muted text-muted-foreground'
    default: return 'bg-muted text-muted-foreground'
  }
}

function directionBadgeClass(direction: string): string {
  switch (direction) {
    case 'deteriorating': return 'bg-rose-100 text-rose-800'
    case 'improving': return 'bg-emerald-100 text-emerald-800'
    default: return 'bg-muted text-muted-foreground'
  }
}

function directionArrow(direction: string): string {
  if (direction === 'improving') return '↑'
  if (direction === 'deteriorating') return '↓'
  return '→'
}

function metricLabel(metric: string): string {
  return metric === 'confidence' ? 'Confidence' : 'Spread (CV%)'
}

function SegmentConfRow({
  seg,
  metric,
}: {
  seg: SegmentConfTrendEntry
  metric: string
}) {
  const label = metricLabel(metric)
  const sign = seg.change_pct >= 0 ? '+' : ''
  const lineColor =
    seg.direction === 'improving'
      ? '#10b981'
      : seg.direction === 'deteriorating'
      ? '#f43f5e'
      : '#94a3b8'

  return (
    <div data-testid={`segment-row-${seg.name}`} className="border rounded-md p-3 space-y-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="font-medium text-sm" data-testid={`segment-name-${seg.name}`}>
          {seg.name}
        </span>
        <div className="flex items-center gap-2">
          <Badge
            className={directionBadgeClass(seg.direction)}
            data-testid={`segment-direction-${seg.name}`}
          >
            {directionArrow(seg.direction)} {seg.direction_label}
          </Badge>
          <span className="text-xs text-muted-foreground" data-testid={`segment-change-${seg.name}`}>
            {sign}{seg.change_pct.toFixed(1)}% change
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
        <div>
          <span className="block font-medium text-foreground">First</span>
          {seg.first_mean.toFixed(3)}
        </div>
        <div>
          <span className="block font-medium text-foreground">Latest</span>
          {seg.last_mean.toFixed(3)}
        </div>
        <div>
          <span className="block font-medium text-foreground">Samples</span>
          {seg.n_samples.toLocaleString()}
        </div>
      </div>
      {seg.daily_stats.length >= 2 && (
        <div
          className="h-16"
          data-testid={`segment-chart-${seg.name}`}
          aria-label={`${label} trend for segment ${seg.name}`}
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={seg.daily_stats}>
              <XAxis dataKey="date" hide />
              <YAxis hide domain={['auto', 'auto']} />
              <Tooltip formatter={(value) => [Number(value).toFixed(3), label]} />
              <Line
                type="monotone"
                dataKey="mean"
                dot={false}
                strokeWidth={2}
                stroke={lineColor}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

interface SegmentConfidenceTrendCardProps {
  result: SegmentConfTrendResult
}

export function SegmentConfidenceTrendCard({ result }: SegmentConfidenceTrendCardProps) {
  const borderClass = verdictBorderClass(result.verdict)
  const badgeClass = verdictBadgeClass(result.verdict)
  const verdictLabel = result.verdict.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
  const label = metricLabel(result.metric)

  return (
    <figure
      data-testid="segment-conf-trend-card"
      className={`rounded-lg border-2 ${borderClass} bg-card p-4 space-y-3`}
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span aria-hidden="true">🎯</span>
        <h3 className="font-semibold text-sm">Segment Confidence Trend</h3>
        <Badge className={badgeClass} data-testid="verdict-badge">
          {verdictLabel}
        </Badge>
        <Badge variant="outline" data-testid="segment-col-badge">
          by {result.segment_column}
        </Badge>
        <Badge variant="outline" data-testid="n-segments-badge">
          {result.n_segments} segment{result.n_segments !== 1 ? 's' : ''}
        </Badge>
        <Badge variant="outline" data-testid="n-days-badge">
          {result.n_days}d window
        </Badge>
        <Badge variant="outline" data-testid="metric-badge">
          {label}
        </Badge>
      </div>

      <p className="text-sm text-muted-foreground" data-testid="summary-text">
        {result.summary}
      </p>

      {result.calibration_gap && (
        <div
          className="flex gap-2 text-xs text-amber-700 bg-amber-50 rounded px-2 py-1"
          data-testid="calibration-gap-warning"
        >
          <span className="font-medium">⚠ Calibration gap:</span>
          <span>
            Model confidence varies significantly across segments — predictions may be
            less reliable for &lsquo;{result.least_confident_segment}&rsquo;.
          </span>
        </div>
      )}

      {result.most_confident_segment && (
        <div className="flex gap-2 text-xs" data-testid="most-confident">
          <span className="text-emerald-600 font-medium">↑ Most confident:</span>
          <span>{result.most_confident_segment}</span>
        </div>
      )}
      {result.least_confident_segment && result.least_confident_segment !== result.most_confident_segment && (
        <div className="flex gap-2 text-xs" data-testid="least-confident">
          <span className="text-rose-600 font-medium">↓ Least confident:</span>
          <span>{result.least_confident_segment}</span>
        </div>
      )}

      {result.segments.length === 0 ? (
        <p className="text-xs text-muted-foreground italic" data-testid="empty-state">
          No segment data available. Make sure the deployment has received predictions
          that include the &lsquo;{result.segment_column}&rsquo; feature.
        </p>
      ) : (
        <div className="space-y-2" data-testid="segments-list">
          {result.segments.map((seg) => (
            <SegmentConfRow key={seg.name} seg={seg} metric={result.metric} />
          ))}
        </div>
      )}

      <figcaption className="sr-only">
        Segment confidence trend by {result.segment_column}: {result.summary}
        Verdict: {result.verdict}. Metric: {label}. Segments analysed: {result.n_segments}.
        {result.most_confident_segment && ` Most confident: ${result.most_confident_segment}.`}
        {result.least_confident_segment && ` Least confident: ${result.least_confident_segment}.`}
        {result.calibration_gap && ' Calibration gap detected.'}
      </figcaption>
    </figure>
  )
}
