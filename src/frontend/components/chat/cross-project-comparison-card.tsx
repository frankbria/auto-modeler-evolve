import type { CrossProjectComparisonResult, CrossProjectComparisonRow } from "@/lib/types"

interface CrossProjectComparisonCardProps {
  result: CrossProjectComparisonResult
}

const RANK_MEDALS: Record<number, string> = { 1: "🥇", 2: "🥈", 3: "🥉" }

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score)
  const color =
    pct >= 80
      ? "bg-emerald-500"
      : pct >= 60
        ? "bg-sky-500"
        : pct >= 40
          ? "bg-amber-500"
          : "bg-rose-500"
  return (
    <div
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Performance score: ${pct} out of 100`}
      className="relative h-2 w-full overflow-hidden rounded-full bg-slate-200"
    >
      <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

function ProblemTypeBadge({ type }: { type: string }) {
  const isReg = type === "regression"
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-xs font-medium ${
        isReg
          ? "bg-violet-100 text-violet-700"
          : "bg-amber-100 text-amber-700"
      }`}
    >
      {isReg ? "Regression" : "Classification"}
    </span>
  )
}

function ProjectRow({ row }: { row: CrossProjectComparisonRow }) {
  const medal = RANK_MEDALS[row.rank]
  const metricLabel =
    row.metric_name === "r2"
      ? "R²"
      : row.metric_name === "accuracy"
        ? "Acc"
        : (row.metric_name ?? "Score")
  const metricDisplay =
    row.metric_value != null
      ? row.metric_name === "r2" || row.metric_name === "accuracy"
        ? `${Math.round(row.metric_value * 100)}%`
        : row.metric_value.toFixed(4)
      : "—"

  return (
    <div className="grid grid-cols-[2rem_1fr_auto] items-center gap-2 rounded border border-indigo-100 bg-white p-2">
      {/* Rank */}
      <div className="flex items-center justify-center text-base" aria-label={`Rank ${row.rank}`}>
        {medal ?? <span className="text-xs font-bold text-slate-400">#{row.rank}</span>}
      </div>

      {/* Project info */}
      <div className="min-w-0 space-y-1">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="truncate text-xs font-semibold text-slate-800">{row.name}</span>
          <ProblemTypeBadge type={row.problem_type} />
        </div>
        <p className="truncate text-xs text-slate-500">
          Predicts{" "}
          <code className="rounded bg-slate-100 px-0.5 font-mono">{row.target_column || "—"}</code>
          {" · "}
          {row.algorithm_plain || row.algorithm.replace(/_/g, " ")}
        </p>
        <ScoreBar score={row.performance_score} />
        <p className="text-xs text-slate-400">
          Score: <span className="font-medium text-slate-600">{row.performance_score.toFixed(0)}/100</span>
          {row.metric_value != null && (
            <>
              {" · "}
              {metricLabel}: <span className="font-medium">{metricDisplay}</span>
            </>
          )}
        </p>
      </div>

      {/* Status */}
      <div className="flex flex-col items-end gap-1 shrink-0">
        {row.has_deployment ? (
          <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700">● Live</span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">Not deployed</span>
        )}
        {row.prediction_count > 0 && (
          <span className="text-xs text-slate-400">
            {row.prediction_count.toLocaleString()} pred
          </span>
        )}
      </div>
    </div>
  )
}

export function CrossProjectComparisonCard({ result }: CrossProjectComparisonCardProps) {
  const { n_projects, n_with_models, winner, projects_compared, insights, summary } = result

  return (
    <div
      role="region"
      aria-label="Cross-project model comparison card"
      className="mt-2 rounded-lg border border-indigo-200 bg-indigo-50/50 p-3 text-sm"
    >
      {/* Header */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span aria-hidden="true">🏆</span>
        <span className="font-semibold text-indigo-900">Cross-Project Comparison</span>
        <span className="rounded bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
          {n_with_models} model{n_with_models !== 1 ? "s" : ""} compared
        </span>
        {n_projects > n_with_models && (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            {n_projects - n_with_models} without model
          </span>
        )}
      </div>

      {/* Summary */}
      <p className="mb-3 text-xs text-indigo-700">{summary}</p>

      {/* No-models empty state */}
      {n_with_models === 0 && (
        <p className="text-xs italic text-slate-500">
          Train models in your projects to compare their performance here.
        </p>
      )}

      {/* Winner highlight */}
      {winner && (
        <div className="mb-3 rounded-md border border-emerald-200 bg-emerald-50 p-2">
          <div className="mb-1 flex items-center gap-1.5">
            <span aria-hidden="true">🥇</span>
            <span className="text-xs font-semibold text-emerald-800">Top Performer</span>
            <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
              {winner.performance_score.toFixed(0)}/100
            </span>
          </div>
          <p className="text-xs text-emerald-700">
            <span className="font-medium">{winner.name}</span>
            {" — "}
            {winner.algorithm_plain || winner.algorithm.replace(/_/g, " ")} predicting{" "}
            <code className="rounded bg-emerald-100 px-1 font-mono">{winner.target_column}</code>
          </p>
        </div>
      )}

      {/* Ranked comparison table */}
      {projects_compared.length > 0 && (
        <div className="mb-3 space-y-1.5">
          {projects_compared.map((row) => (
            <ProjectRow key={row.project_id} row={row} />
          ))}
        </div>
      )}

      {/* Insights */}
      {insights.length > 0 && (
        <div className="rounded border border-indigo-100 bg-white p-2">
          <p className="mb-1 text-xs font-semibold text-indigo-800">Insights</p>
          <ul className="space-y-1">
            {insights.map((insight, i) => (
              <li key={i} className="text-xs text-slate-600 before:mr-1.5 before:content-['→']">
                {insight}
              </li>
            ))}
          </ul>
        </div>
      )}

      <figcaption className="sr-only">
        Cross-project model comparison showing normalized performance scores for{" "}
        {n_with_models} project{n_with_models !== 1 ? "s" : ""}.
        {winner ? ` Top performer: ${winner.name} with score ${winner.performance_score.toFixed(0)}/100.` : ""}
      </figcaption>
    </div>
  )
}
