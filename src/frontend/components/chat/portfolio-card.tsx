import type { PortfolioResult } from "@/lib/types"

interface PortfolioCardProps {
  result: PortfolioResult
}

function metricLabel(name: string | null): string {
  if (name === "r2") return "R²"
  if (name === "accuracy") return "Accuracy"
  return name ?? "Score"
}

function algorithmLabel(alg: string | null): string {
  if (!alg) return "—"
  return alg.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function pct(value: number | null): string {
  if (value == null) return "—"
  return `${Math.round(value * 100)}%`
}

export function PortfolioCard({ result }: PortfolioCardProps) {
  const { total_projects, active_deployments, total_predictions, best_performer, projects, summary } = result

  return (
    <div
      role="region"
      aria-label="Portfolio overview card"
      className="mt-2 rounded-lg border border-purple-200 bg-purple-50/50 p-3 text-sm"
    >
      {/* Header */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span aria-hidden="true">🗂️</span>
        <span className="font-semibold text-purple-900">Model Portfolio</span>
        <span className="rounded bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-700">
          {total_projects} project{total_projects !== 1 ? "s" : ""}
        </span>
        <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
          {active_deployments} deployed
        </span>
        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
          {total_predictions.toLocaleString()} prediction{total_predictions !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Summary */}
      <p className="mb-3 text-xs text-purple-700">{summary}</p>

      {/* Best performer highlight */}
      {best_performer && (
        <div className="mb-3 rounded-md border border-emerald-200 bg-emerald-50 p-2">
          <div className="mb-1 flex items-center gap-1.5">
            <span aria-hidden="true">🏆</span>
            <span className="text-xs font-semibold text-emerald-800">Best Performer</span>
          </div>
          <p className="text-xs text-emerald-700">
            <span className="font-medium">{best_performer.name}</span>
            {" — "}
            {algorithmLabel(best_performer.algorithm)} predicting{" "}
            <code className="rounded bg-emerald-100 px-1 font-mono">{best_performer.target_column}</code>
            {" · "}
            <span className="font-medium">
              {pct(best_performer.metric_value)} {metricLabel(best_performer.metric_name)}
            </span>
          </p>
        </div>
      )}

      {/* Per-project rows */}
      {projects.length === 0 ? (
        <p className="text-xs italic text-slate-500">No projects yet. Create a project to get started.</p>
      ) : (
        <div className="space-y-1.5">
          {projects.map((proj) => (
            <div
              key={proj.project_id}
              className="flex flex-wrap items-start gap-2 rounded border border-purple-100 bg-white p-2"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs font-medium text-slate-800">{proj.name}</p>
                {proj.dataset_filename && (
                  <p className="truncate text-xs text-slate-500">
                    {proj.dataset_filename}
                    {proj.row_count != null ? ` · ${proj.row_count.toLocaleString()} rows` : ""}
                  </p>
                )}
                {proj.best_target_column && (
                  <p className="text-xs text-slate-500">
                    Predicts{" "}
                    <code className="rounded bg-slate-100 px-0.5 font-mono">{proj.best_target_column}</code>
                    {" · "}
                    {algorithmLabel(proj.best_algorithm)}
                  </p>
                )}
              </div>
              <div className="flex flex-col items-end gap-1">
                {proj.best_metric_value != null && (
                  <span className="rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
                    {pct(proj.best_metric_value)} {metricLabel(proj.best_metric_name)}
                  </span>
                )}
                {proj.has_deployment ? (
                  <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700">
                    ● Live
                  </span>
                ) : proj.model_count > 0 ? (
                  <span className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-700">
                    Trained
                  </span>
                ) : (
                  <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                    No model
                  </span>
                )}
                {proj.prediction_count > 0 && (
                  <span className="text-xs text-slate-400">
                    {proj.prediction_count.toLocaleString()} predictions
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
