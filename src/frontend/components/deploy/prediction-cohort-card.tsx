"use client"

import { PredictionCohortResult, CohortCategoricalProfile, CohortNumericProfile } from "@/lib/types"

interface PredictionCohortCardProps {
  result: PredictionCohortResult
}

function dirLabel(direction: string) {
  return direction === "highest" ? "Highest" : "Lowest"
}

function ratioBadge(ratio: number | null): string {
  if (ratio === null) return "bg-slate-100 text-slate-600"
  if (ratio >= 1.5) return "bg-rose-100 text-rose-800"
  if (ratio >= 1.2) return "bg-amber-100 text-amber-800"
  if (ratio <= 0.7) return "bg-sky-100 text-sky-700"
  if (ratio <= 0.85) return "bg-indigo-100 text-indigo-700"
  return "bg-slate-100 text-slate-600"
}

function ratioText(ratio: number | null): string {
  if (ratio === null) return "—"
  if (ratio > 1) return `${ratio.toFixed(1)}× more`
  if (ratio < 1) return `${ratio.toFixed(1)}× less`
  return "≈ same"
}

function CategoricalRow({ profile }: { profile: CohortCategoricalProfile }) {
  return (
    <div className="mb-3 last:mb-0">
      <p className="mb-1 text-xs font-semibold text-slate-600">{profile.column.replace(/_/g, " ")}</p>
      {profile.categories.map((cat) => (
        <div key={cat.value} className="mb-1 flex items-center gap-2">
          <span className="w-24 truncate text-xs text-slate-700" title={cat.value}>
            {cat.value}
          </span>
          <div className="flex flex-1 flex-col gap-0.5">
            <div className="relative h-2 rounded bg-slate-100">
              <div
                className="absolute left-0 top-0 h-full rounded bg-indigo-400"
                style={{ width: `${Math.min(cat.top_pct, 100)}%` }}
              />
            </div>
            <div className="relative h-2 rounded bg-slate-100">
              <div
                className="absolute left-0 top-0 h-full rounded bg-slate-300"
                style={{ width: `${Math.min(cat.overall_pct, 100)}%` }}
              />
            </div>
          </div>
          <div className="w-16 text-right text-xs">
            <span className="font-semibold text-indigo-700">{cat.top_pct}%</span>
            <span className="text-slate-400"> / {cat.overall_pct}%</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function NumericRow({ profile }: { profile: CohortNumericProfile }) {
  const badge = ratioBadge(profile.ratio)
  const text = ratioText(profile.ratio)
  const arrow = profile.direction === "higher" ? "↑" : profile.direction === "lower" ? "↓" : "≈"
  return (
    <div className="flex items-center gap-2 py-1">
      <span className="w-28 truncate text-xs font-medium text-slate-700" title={profile.column}>
        {profile.column.replace(/_/g, " ")}
      </span>
      <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ${badge}`}>
        {arrow} {text}
      </span>
      <span className="ml-auto text-xs text-slate-500">
        top avg: <span className="font-medium text-slate-700">{profile.top_mean.toLocaleString()}</span>
        {" vs "}
        <span className="text-slate-400">{profile.overall_mean.toLocaleString()}</span>
      </span>
    </div>
  )
}

export function PredictionCohortCard({ result }: PredictionCohortCardProps) {
  const hasCat = result.categorical_profile.length > 0
  const hasNum = result.numeric_profile.length > 0

  return (
    <figure aria-label={`Prediction cohort profile: top ${result.n} ${result.direction} ${result.target_column}`}>
      <div className="rounded-lg border-2 border-indigo-400 bg-white p-4 shadow-sm">
        {/* Header */}
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span aria-hidden="true" className="text-lg">🔍</span>
          <h3 className="font-semibold text-slate-800">
            Cohort Profile — Top {result.n} {result.target_column}
          </h3>
          <span className="rounded bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-800">
            {dirLabel(result.direction)}
          </span>
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
            {result.n} of {result.total_scored.toLocaleString()} rows
          </span>
        </div>

        {/* Characterization */}
        <p className="mb-4 rounded bg-indigo-50 px-3 py-2 text-sm font-medium text-indigo-900">
          {result.characterization}
        </p>

        {/* Categorical profile */}
        {hasCat && (
          <div className="mb-4">
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Categorical Breakdown
            </h4>
            <div className="mb-1 flex gap-4 text-xs text-slate-500">
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-3 rounded bg-indigo-400" />
                Top {result.n} group
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-3 rounded bg-slate-300" />
                All rows
              </span>
            </div>
            {result.categorical_profile.map((cp) => (
              <CategoricalRow key={cp.column} profile={cp} />
            ))}
          </div>
        )}

        {/* Numeric profile */}
        {hasNum && (
          <div>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Numeric Averages
            </h4>
            <div className="divide-y divide-slate-100 rounded border border-slate-200">
              {result.numeric_profile.map((np) => (
                <div key={np.column} className="px-3">
                  <NumericRow profile={np} />
                </div>
              ))}
            </div>
          </div>
        )}

        {!hasCat && !hasNum && (
          <p className="text-xs text-slate-500 italic">
            No additional categorical or numeric features to profile.
          </p>
        )}
      </div>
    </figure>
  )
}
