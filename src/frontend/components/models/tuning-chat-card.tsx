"use client"

import { Badge } from "@/components/ui/badge"
import type { TuningChatResult } from "@/lib/types"

interface TuningChatCardProps {
  result: TuningChatResult
}

function MetricDelta({ original, tuned }: { original: number; tuned: number }) {
  const delta = tuned - original
  const pct = original !== 0 ? (delta / Math.abs(original)) * 100 : 0
  if (Math.abs(delta) < 0.0001) {
    return <span className="text-xs text-muted-foreground">—</span>
  }
  const positive = delta > 0
  return (
    <span
      className={`text-xs font-medium ${positive ? "text-emerald-700" : "text-rose-700"}`}
    >
      {positive ? "+" : ""}
      {pct.toFixed(1)}%
    </span>
  )
}

function MetricsTable({
  original,
  tuned,
}: {
  original: Record<string, number>
  tuned: Record<string, number>
}) {
  const keys = Object.keys(original)
  if (keys.length === 0) return null
  return (
    <table
      className="w-full text-xs border-collapse"
      aria-label="Before and after metrics"
      data-testid="tuning-metrics-table"
    >
      <thead>
        <tr className="text-left text-muted-foreground">
          <th className="py-1 pr-3 font-medium">Metric</th>
          <th className="py-1 pr-3 font-medium">Before</th>
          <th className="py-1 pr-3 font-medium">After</th>
          <th className="py-1 font-medium">Change</th>
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => {
          const orig = original[k]
          const tune = tuned[k] ?? orig
          return (
            <tr key={k} className="border-t border-border/50">
              <td className="py-1 pr-3 text-foreground capitalize">
                {k.replace(/_/g, " ")}
              </td>
              <td className="py-1 pr-3 text-muted-foreground">
                {orig.toFixed(4)}
              </td>
              <td
                className={`py-1 pr-3 font-medium ${
                  tune > orig ? "text-emerald-700" : tune < orig ? "text-rose-700" : ""
                }`}
              >
                {tune.toFixed(4)}
              </td>
              <td className="py-1">
                <MetricDelta original={orig} tuned={tune} />
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function BestParamsDisplay({ params }: { params: Record<string, unknown> }) {
  const entries = Object.entries(params)
  if (entries.length === 0) return null
  return (
    <div
      className="bg-muted/40 rounded-md px-3 py-2 space-y-1"
      data-testid="tuning-best-params"
    >
      <p className="text-xs font-medium text-foreground mb-1">Best parameters found:</p>
      <div className="flex flex-wrap gap-x-4 gap-y-0.5">
        {entries.map(([k, v]) => (
          <span key={k} className="text-xs text-muted-foreground font-mono">
            <span className="text-foreground">{k}</span>
            {" = "}
            <span className="text-emerald-700">{String(v)}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

export function TuningChatCard({ result }: TuningChatCardProps) {
  const {
    tunable,
    algorithm_name,
    problem_type,
    original_metrics,
    tuned_metrics,
    best_params,
    improved,
    improvement_pct,
    summary,
  } = result

  // Border/accent color: emerald if improved, amber if unchanged, rose if declined, slate if not tunable
  const borderColor = !tunable
    ? "border-slate-200"
    : improved
      ? "border-emerald-200"
      : "border-amber-200"
  const bgColor = !tunable
    ? "bg-slate-50"
    : improved
      ? "bg-emerald-50"
      : "bg-amber-50"
  const headingColor = !tunable
    ? "text-slate-900"
    : improved
      ? "text-emerald-900"
      : "text-amber-900"

  const hasMetrics =
    tunable &&
    original_metrics &&
    Object.keys(original_metrics).length > 0 &&
    tuned_metrics &&
    Object.keys(tuned_metrics).length > 0

  const hasParams = tunable && best_params && Object.keys(best_params).length > 0

  return (
    <figure
      className={`border ${borderColor} rounded-lg p-4 bg-white my-2 space-y-3`}
      aria-label="Hyperparameter tuning result"
      data-testid="tuning-chat-card"
    >
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-lg">🔧</span>
          <h3 className={`text-sm font-semibold ${headingColor}`}>
            Hyperparameter Tuning
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {problem_type && (
            <Badge className="text-xs bg-muted text-muted-foreground border-0 capitalize">
              {problem_type}
            </Badge>
          )}
          <Badge className="text-xs bg-muted text-muted-foreground border-0">
            {algorithm_name}
          </Badge>
          {tunable && improved !== undefined && (
            <Badge
              className={`text-xs border-0 ${
                improved
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-amber-100 text-amber-800"
              }`}
              data-testid="tuning-improvement-badge"
            >
              {improved ? "Improved" : "Unchanged"}
            </Badge>
          )}
          {tunable && improvement_pct != null && Math.abs(improvement_pct) >= 0.01 && (
            <Badge
              className={`text-xs border-0 ${
                improvement_pct > 0
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-rose-100 text-rose-800"
              }`}
              data-testid="tuning-pct-badge"
            >
              {improvement_pct > 0 ? "+" : ""}
              {improvement_pct.toFixed(1)}%
            </Badge>
          )}
        </div>
      </div>

      {/* Not tunable explanation */}
      {!tunable && (
        <div className={`${bgColor} border border-slate-200 rounded-md px-3 py-2`}>
          <p className="text-xs text-slate-700 leading-relaxed">
            <strong>{algorithm_name}</strong> does not have hyperparameters that benefit
            from automated search. Consider switching to a tree-based model like{" "}
            <em>Random Forest</em> or <em>Gradient Boosting</em> for tuning.
          </p>
        </div>
      )}

      {/* Before/After metrics */}
      {hasMetrics && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-foreground">Before vs. After:</p>
          <MetricsTable
            original={original_metrics!}
            tuned={tuned_metrics!}
          />
        </div>
      )}

      {/* Best params */}
      {hasParams && <BestParamsDisplay params={best_params!} />}

      {/* Summary */}
      <p
        className="text-xs text-muted-foreground leading-relaxed"
        data-testid="tuning-summary"
      >
        {summary}
      </p>

      <figcaption className="text-xs text-muted-foreground border-t pt-2 mt-1">
        Tuning ran RandomizedSearchCV (10 iterations, 3-fold cross-validation). The
        improved model is saved and ready to use for predictions.
      </figcaption>
    </figure>
  )
}
