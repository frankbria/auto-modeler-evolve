"use client"

import { RetrainingReadinessResult, RetrainingReadinessSignal } from "@/lib/types"

interface Props {
  data: RetrainingReadinessResult
  onActionClick?: (text: string) => void
}

function VerdictColor(verdict: string): string {
  switch (verdict) {
    case "retrain_now": return "rose"
    case "retrain_soon": return "orange"
    case "monitor": return "amber"
    default: return "emerald"
  }
}

function VerdictEmoji(verdict: string): string {
  switch (verdict) {
    case "retrain_now": return "🚨"
    case "retrain_soon": return "⚠️"
    case "monitor": return "👀"
    default: return "✅"
  }
}

function SignalRow({ signal }: { signal: RetrainingReadinessSignal }) {
  const barWidth = Math.min(100, Math.round((signal.score_contribution / 40) * 100))
  return (
    <div
      className="flex items-start gap-3 py-2 border-b border-gray-100 last:border-0"
      data-testid={`signal-row-${signal.name}`}
    >
      <span
        className={`mt-0.5 text-xs font-semibold px-1.5 py-0.5 rounded ${
          signal.is_firing
            ? "bg-rose-100 text-rose-700"
            : "bg-emerald-100 text-emerald-700"
        }`}
        aria-label={signal.is_firing ? "Signal firing" : "Signal clear"}
      >
        {signal.is_firing ? "!" : "✓"}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm font-medium text-gray-700">{signal.label}</span>
          <span className="text-xs text-gray-500 shrink-0">{signal.value_str}</span>
        </div>
        <p className="text-xs text-gray-500 mt-0.5">{signal.detail}</p>
        {signal.is_firing && (
          <div className="mt-1.5 h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-rose-400 rounded-full transition-all"
              style={{ width: `${barWidth}%` }}
              role="progressbar"
              aria-valuenow={signal.score_contribution}
              aria-valuemin={0}
              aria-valuemax={40}
              aria-label={`${signal.label} contribution`}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export function RetrainingReadinessCard({ data, onActionClick }: Props) {
  const color = VerdictColor(data.verdict)
  const emoji = VerdictEmoji(data.verdict)

  const borderClass = {
    rose: "border-rose-300 bg-rose-50/30",
    orange: "border-orange-300 bg-orange-50/30",
    amber: "border-amber-300 bg-amber-50/30",
    emerald: "border-emerald-300 bg-emerald-50/30",
  }[color]

  const badgeClass = {
    rose: "bg-rose-100 text-rose-800",
    orange: "bg-orange-100 text-orange-800",
    amber: "bg-amber-100 text-amber-800",
    emerald: "bg-emerald-100 text-emerald-800",
  }[color]

  const scoreBarClass = {
    rose: "bg-rose-500",
    orange: "bg-orange-500",
    amber: "bg-amber-500",
    emerald: "bg-emerald-500",
  }[color]

  return (
    <figure
      className={`rounded-xl border-2 p-4 space-y-3 ${borderClass}`}
      data-testid="retraining-readiness-card"
      aria-label="Retraining Readiness Assessment"
    >
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-lg">{emoji}</span>
          <span className="font-semibold text-gray-800">Retraining Readiness</span>
        </div>
        <div className="flex gap-2 flex-wrap">
          <span
            className={`text-xs font-semibold px-2 py-0.5 rounded-full ${badgeClass}`}
            data-testid="verdict-badge"
          >
            {data.verdict_label}
          </span>
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
            Score: {data.score}/100
          </span>
          {data.signals_available > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
              {data.signals_firing}/{data.signals_available} signals firing
            </span>
          )}
        </div>
      </div>

      {/* Score bar */}
      <div>
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Urgency</span>
          <span>{data.score}%</span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${scoreBarClass}`}
            style={{ width: `${data.score}%` }}
            role="progressbar"
            aria-valuenow={data.score}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Retraining urgency score"
            data-testid="score-bar"
          />
        </div>
      </div>

      {/* Signals */}
      {data.signals.length > 0 ? (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
            Monitoring Signals
          </p>
          <div>
            {data.signals.map((signal) => (
              <SignalRow key={signal.name} signal={signal} />
            ))}
          </div>
        </div>
      ) : (
        <p className="text-sm text-gray-500 italic">
          No monitoring data available yet. Run predictions and submit feedback to build a picture of model health.
        </p>
      )}

      {/* Recommendations */}
      {data.recommendations.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
            Recommendations
          </p>
          <ul className="space-y-1">
            {data.recommendations.map((rec, i) => (
              <li key={i} className="text-sm text-gray-700 flex gap-1.5">
                <span className="text-gray-400 mt-0.5">→</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Retrain CTA */}
      {(data.verdict === "retrain_now" || data.verdict === "retrain_soon") &&
        onActionClick && (
          <button
            onClick={() => onActionClick("retrain my model")}
            className="w-full text-sm font-medium py-2 px-3 rounded-lg bg-rose-600 text-white hover:bg-rose-700 transition-colors"
            data-testid="retrain-cta-button"
          >
            Retrain Model Now
          </button>
        )}

      {/* Summary */}
      <p className="text-sm text-gray-600 italic" data-testid="readiness-summary">
        {data.summary}
      </p>

      <figcaption className="sr-only">
        Retraining readiness score: {data.score}/100. Verdict: {data.verdict_label}.{" "}
        {data.summary}
      </figcaption>
    </figure>
  )
}
