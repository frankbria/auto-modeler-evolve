'use client'

import { Badge } from '@/components/ui/badge'
import type { ApiUptimeSummaryResult, ApiUptimeDayStat, ApiUptimeVerdict } from '@/lib/types'

function verdictBorderClass(verdict: ApiUptimeVerdict): string {
  switch (verdict) {
    case 'healthy': return 'border-emerald-500'
    case 'degraded': return 'border-amber-500'
    case 'mostly_idle': return 'border-slate-400'
    default: return 'border-muted'
  }
}

function verdictBadgeClass(verdict: ApiUptimeVerdict): string {
  switch (verdict) {
    case 'healthy': return 'bg-emerald-100 text-emerald-800'
    case 'degraded': return 'bg-amber-100 text-amber-800'
    case 'mostly_idle': return 'bg-slate-100 text-slate-700'
    default: return 'bg-muted text-muted-foreground'
  }
}

function verdictLabel(verdict: ApiUptimeVerdict): string {
  switch (verdict) {
    case 'healthy': return 'Healthy'
    case 'degraded': return 'Degraded'
    case 'mostly_idle': return 'Mostly Idle'
    case 'idle': return 'Idle'
    default: return 'No Data'
  }
}

function dayStatusClass(status: string): string {
  switch (status) {
    case 'active': return 'bg-emerald-500'
    case 'degraded': return 'bg-amber-400'
    default: return 'bg-slate-200'
  }
}

function dayStatusLabel(status: string): string {
  switch (status) {
    case 'active': return 'active'
    case 'degraded': return 'degraded (high latency)'
    default: return 'silent'
  }
}

function DayCell({ day }: { day: ApiUptimeDayStat }) {
  const shortDate = day.date.slice(5) // "MM-DD"
  return (
    <div
      className="flex flex-col items-center gap-1"
      title={`${day.date}: ${day.n_predictions} predictions, status: ${dayStatusLabel(day.status)}`}
    >
      <div
        className={`h-6 w-6 rounded-sm ${dayStatusClass(day.status)}`}
        aria-label={`${day.date}: ${dayStatusLabel(day.status)}, ${day.n_predictions} predictions`}
      />
      <span className="text-[9px] text-muted-foreground">{shortDate}</span>
      <span className="text-[9px] font-medium">{day.n_predictions > 0 ? day.n_predictions : '—'}</span>
    </div>
  )
}

export function ApiUptimeSummaryCard({ result }: { result: ApiUptimeSummaryResult }) {
  const { verdict, n_days, total_predictions, n_active_days, n_silent_days, degraded_days,
    daily_stats, overall_avg_latency_ms, overall_p95_latency_ms, peak_latency_ms,
    peak_latency_date, note, summary } = result

  return (
    <figure
      className={`rounded-lg border-2 ${verdictBorderClass(verdict)} bg-card p-4 shadow-sm`}
      data-testid="api-uptime-summary-card"
    >
      {/* Header */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span aria-hidden="true" className="text-lg">📡</span>
        <span className="font-semibold text-sm">API Uptime Summary</span>
        <Badge className={verdictBadgeClass(verdict)} data-testid="uptime-verdict-badge">
          {verdictLabel(verdict)}
        </Badge>
        <Badge variant="secondary" data-testid="uptime-n-days-badge">
          Last {n_days} days
        </Badge>
        <Badge variant="outline" data-testid="uptime-total-badge">
          {total_predictions} predictions
        </Badge>
      </div>

      {/* Daily activity grid */}
      {daily_stats.length > 0 && (
        <div className="mb-4" data-testid="uptime-daily-grid">
          <p className="mb-2 text-xs text-muted-foreground font-medium">Daily Activity</p>
          <div className="flex flex-wrap gap-3">
            {daily_stats.map((day) => (
              <DayCell key={day.date} day={day} />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-3 text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" aria-hidden="true" />
              Active
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-amber-400" aria-hidden="true" />
              High latency
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm bg-slate-200" aria-hidden="true" />
              Silent
            </span>
          </div>
        </div>
      )}

      {/* Activity summary row */}
      <div className="mb-3 flex flex-wrap gap-3 text-xs" data-testid="uptime-activity-row">
        <span>
          <span className="font-semibold text-emerald-700">{n_active_days}</span>
          <span className="text-muted-foreground"> active days</span>
        </span>
        <span>
          <span className="font-semibold text-slate-500">{n_silent_days}</span>
          <span className="text-muted-foreground"> silent days</span>
        </span>
        {degraded_days > 0 && (
          <span>
            <span className="font-semibold text-amber-600">{degraded_days}</span>
            <span className="text-muted-foreground"> degraded days</span>
          </span>
        )}
      </div>

      {/* Latency stats */}
      {(overall_avg_latency_ms !== null || overall_p95_latency_ms !== null) && (
        <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-3" data-testid="uptime-latency-stats">
          {overall_avg_latency_ms !== null && (
            <div className="rounded bg-muted/40 p-2 text-center">
              <div className="text-sm font-semibold">{overall_avg_latency_ms.toFixed(0)} ms</div>
              <div className="text-[10px] text-muted-foreground">Avg latency</div>
            </div>
          )}
          {overall_p95_latency_ms !== null && (
            <div className="rounded bg-muted/40 p-2 text-center">
              <div className="text-sm font-semibold">{overall_p95_latency_ms.toFixed(0)} ms</div>
              <div className="text-[10px] text-muted-foreground">p95 latency</div>
            </div>
          )}
          {peak_latency_ms !== null && (
            <div className="rounded bg-muted/40 p-2 text-center">
              <div className="text-sm font-semibold">{peak_latency_ms.toFixed(0)} ms</div>
              <div className="text-[10px] text-muted-foreground">
                Peak{peak_latency_date ? ` (${peak_latency_date.slice(5)})` : ''}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Summary */}
      <p className="mb-2 text-sm text-foreground" data-testid="uptime-summary">{summary}</p>

      {/* Honesty note about silent days */}
      <p className="text-[11px] text-muted-foreground italic" data-testid="uptime-note">{note}</p>

      <figcaption className="sr-only">
        API uptime summary: {verdictLabel(verdict)}, {n_active_days} active days out of {n_days},
        {total_predictions} total predictions. {summary}
      </figcaption>
    </figure>
  )
}
