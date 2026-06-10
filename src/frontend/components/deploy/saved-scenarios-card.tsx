"use client"

import { SavedScenariosResult, SavedScenarioEntry } from "@/lib/types"

interface SavedScenariosCardProps {
  result: SavedScenariosResult
}

function fmtPred(entry: SavedScenarioEntry): string {
  if (entry.prediction_value) return entry.prediction_value
  if (entry.prediction_numeric !== null && entry.prediction_numeric !== undefined) {
    const v = entry.prediction_numeric
    if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
    if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
    return parseFloat(v.toFixed(3)).toString()
  }
  return "—"
}

function fmtInputs(inputs: Record<string, number | string>): string {
  const entries = Object.entries(inputs)
  if (entries.length === 0) return "baseline"
  return entries
    .slice(0, 4)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ") + (entries.length > 4 ? ` +${entries.length - 4} more` : "")
}

function barWidth(entry: SavedScenarioEntry, scenarios: SavedScenarioEntry[]): number {
  const numerics = scenarios
    .map((s) => s.prediction_numeric)
    .filter((v): v is number => v !== null && v !== undefined)
  if (numerics.length < 2 || entry.prediction_numeric === null || entry.prediction_numeric === undefined) {
    return 50
  }
  const min = Math.min(...numerics)
  const max = Math.max(...numerics)
  if (max === min) return 50
  return Math.round(((entry.prediction_numeric - min) / (max - min)) * 90 + 5)
}

function barColor(entry: SavedScenarioEntry, best: SavedScenarioEntry | null, worst: SavedScenarioEntry | null): string {
  if (best && entry.id === best.id) return "bg-emerald-400"
  if (worst && entry.id === worst.id) return "bg-rose-400"
  return "bg-sky-300"
}

export function SavedScenariosCard({ result }: SavedScenariosCardProps) {
  const {
    scenarios,
    n_scenarios,
    problem_type,
    target_column,
    best_scenario,
    worst_scenario,
    spread,
    summary,
  } = result

  const isRegression = problem_type === "regression"

  return (
    <figure
      className="rounded-xl border border-sky-300 bg-sky-50 p-4 text-sm"
      aria-label="Saved scenario comparison"
      data-testid="saved-scenarios-card"
    >
      {/* Header */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-lg" aria-hidden="true">📋</span>
        <span className="font-semibold text-sky-900">Saved Scenarios</span>
        <span className="rounded-full bg-sky-200 px-2 py-0.5 text-xs font-medium text-sky-800">
          {n_scenarios} scenario{n_scenarios !== 1 ? "s" : ""}
        </span>
        {target_column && (
          <span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs font-medium text-slate-700">
            target: {target_column}
          </span>
        )}
        <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
          {isRegression ? "Regression" : "Classification"}
        </span>
      </div>

      {/* Empty state */}
      {n_scenarios === 0 && (
        <p className="py-4 text-center text-slate-500">
          No saved scenarios yet. Say <em>&quot;save discount=0.1 as Low Discount&quot;</em> to add one.
        </p>
      )}

      {/* Scenarios list */}
      {n_scenarios > 0 && (
        <div className="space-y-2">
          {scenarios.map((entry) => (
            <div
              key={entry.id}
              className={`rounded-lg border p-3 ${
                best_scenario?.id === entry.id
                  ? "border-emerald-300 bg-emerald-50"
                  : worst_scenario?.id === entry.id
                  ? "border-rose-300 bg-rose-50"
                  : "border-slate-200 bg-white"
              }`}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold text-slate-800">{entry.name}</span>
                {best_scenario?.id === entry.id && n_scenarios > 1 && (
                  <span className="rounded-full bg-emerald-200 px-2 py-0.5 text-xs font-medium text-emerald-800">
                    Best
                  </span>
                )}
                {worst_scenario?.id === entry.id && n_scenarios > 1 && (
                  <span className="rounded-full bg-rose-200 px-2 py-0.5 text-xs font-medium text-rose-800">
                    Worst
                  </span>
                )}
                <span className="ml-auto font-mono text-sm font-semibold text-slate-700">
                  {fmtPred(entry)}
                </span>
              </div>

              {/* Inputs row */}
              {Object.keys(entry.inputs).length > 0 && (
                <p className="mt-1 text-xs text-slate-500">
                  {fmtInputs(entry.inputs)}
                </p>
              )}

              {/* Regression bar */}
              {isRegression && entry.prediction_numeric !== null && (
                <div className="mt-2" role="img" aria-label={`Prediction bar for ${entry.name}`}>
                  <div className="h-2 w-full rounded-full bg-slate-200">
                    <div
                      className={`h-2 rounded-full transition-all ${barColor(entry, best_scenario, worst_scenario)}`}
                      style={{ width: `${barWidth(entry, scenarios)}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Classification confidence */}
              {!isRegression && entry.confidence !== null && entry.confidence !== undefined && (
                <p className="mt-1 text-xs text-slate-500">
                  confidence: {Math.round(entry.confidence * 100)}%
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Spread callout for regression */}
      {isRegression && spread !== null && n_scenarios >= 2 && (
        <div className="mt-3 rounded-lg bg-white p-2 text-xs text-slate-600">
          <span className="font-medium text-slate-700">Range: </span>
          {spread.toLocaleString()} between best and worst scenario
        </div>
      )}

      {/* Summary */}
      <p className="mt-3 text-slate-700">{summary}</p>

      <figcaption className="sr-only">
        Saved scenario comparison showing {n_scenarios} named what-if configurations and their predicted outputs.
      </figcaption>
    </figure>
  )
}
