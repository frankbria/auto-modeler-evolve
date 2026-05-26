import type {
  CohortEvolutionResult,
  CohortEvolutionShift,
  CohortCategoricalShift,
  CohortEvolutionPeriod,
} from "@/lib/types"

interface CohortEvolutionCardProps {
  result: CohortEvolutionResult
}

/** A single shift row showing ▲ or ▼ change for one category */
function ShiftRow({ shift }: { shift: CohortCategoricalShift }) {
  const isRising = shift.direction === "rising"
  const arrow = isRising ? "▲" : "▼"
  const colourClass = isRising
    ? "text-emerald-700 bg-emerald-50 border-emerald-200"
    : "text-rose-700 bg-rose-50 border-rose-200"
  const changeSign = isRising ? "+" : ""
  const colPlain = shift.column.replace(/_/g, " ")

  return (
    <div
      className={`flex items-center gap-2 px-2 py-1 rounded border text-xs ${colourClass}`}
      data-testid="cohort-shift-row"
    >
      <span aria-hidden="true">{arrow}</span>
      <span className="font-medium">{colPlain}</span>
      <span className="text-gray-500">=</span>
      <span className="font-mono">&quot;{shift.category}&quot;</span>
      <span>
        {shift.from_pct}% → {shift.to_pct}%
      </span>
      <span className="ml-auto font-semibold">
        ({changeSign}
        {shift.change}pp)
      </span>
    </div>
  )
}

/** A single period node showing the top categorical breakdowns */
function PeriodNode({ period }: { period: CohortEvolutionPeriod }) {
  const topCats = period.categorical_profile.slice(0, 2)

  return (
    <div
      className="flex-1 min-w-0 bg-white border border-gray-200 rounded-lg p-3"
      data-testid="cohort-period-node"
    >
      <div className="font-medium text-xs text-gray-700 truncate mb-2">
        {period.period_label}
      </div>
      <div className="text-xs text-gray-500 mb-2">
        {period.total_rows.toLocaleString()} rows
      </div>
      {topCats.length > 0 ? (
        <ul className="space-y-1" aria-label={`Top cohort traits for ${period.period_label}`}>
          {topCats.map((cp) =>
            cp.categories.slice(0, 1).map((cat) => {
              const colPlain = cp.column.replace(/_/g, " ")
              const barWidth = Math.min(100, Math.round(cat.top_pct))
              return (
                <li key={`${cp.column}-${cat.value}`} className="space-y-0.5">
                  <div className="flex justify-between text-xs text-gray-600">
                    <span className="truncate max-w-[100px]">
                      {colPlain} = &quot;{cat.value}&quot;
                    </span>
                    <span className="font-medium">{cat.top_pct}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-violet-400 rounded-full transition-all"
                      style={{ width: `${barWidth}%` }}
                      aria-label={`${cat.top_pct}% of cohort is ${cp.column} = ${cat.value}`}
                    />
                  </div>
                </li>
              )
            })
          )}
        </ul>
      ) : (
        <p className="text-xs text-gray-400 italic">No categorical data</p>
      )}
    </div>
  )
}

/** Arrow connector between two periods showing the key shift */
function ShiftConnector({ shift }: { shift: CohortEvolutionShift }) {
  const topShift = shift.categorical_shifts[0]
  const hasShifts = shift.categorical_shifts.length > 0

  return (
    <div
      className="flex flex-col items-center justify-center px-1 shrink-0"
      data-testid="cohort-shift-connector"
      aria-label={hasShifts ? shift.summary : "Stable composition"}
    >
      <div className="text-gray-400 text-sm">→</div>
      {topShift && (
        <div
          className={`text-[10px] text-center max-w-[60px] leading-tight mt-0.5 ${
            topShift.direction === "rising" ? "text-emerald-600" : "text-rose-600"
          }`}
        >
          {topShift.direction === "rising" ? "▲" : "▼"} {topShift.category}
        </div>
      )}
    </div>
  )
}

export function CohortEvolutionCard({ result }: CohortEvolutionCardProps) {
  const directionLabel = result.direction === "highest" ? "Highest" : "Lowest"

  // Collect all shifts for the detail section
  const allShifts = result.shifts.flatMap((s) => s.categorical_shifts)
  const topOverallShifts = allShifts
    .slice()
    .sort((a, b) => Math.abs(b.change) - Math.abs(a.change))
    .slice(0, 6)

  return (
    <figure
      className="rounded-xl border border-violet-300 bg-violet-50 p-4 my-2 text-sm"
      data-testid="cohort-evolution-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-base">
          📈
        </span>
        <span className="font-semibold text-violet-900">Cohort Evolution</span>
        <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-violet-100 text-violet-700 border border-violet-200">
          {result.n_periods} periods
        </span>
        <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-700 border border-gray-200">
          Top-{result.n} {directionLabel}
        </span>
        <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200">
          {result.target_column}
        </span>
      </div>

      {/* Summary sentence */}
      <p className="text-xs text-violet-800 mb-4 leading-relaxed">{result.summary}</p>

      {/* Period timeline */}
      <div
        className="flex items-stretch gap-1 mb-4 overflow-x-auto pb-1"
        role="list"
        aria-label="Cohort composition per period"
      >
        {result.periods.map((period, idx) => (
          <div key={period.dataset_id} className="flex items-center gap-1 shrink-0">
            <div role="listitem" className="w-36">
              <PeriodNode period={period} />
            </div>
            {idx < result.periods.length - 1 && result.shifts[idx] && (
              <ShiftConnector shift={result.shifts[idx]} />
            )}
          </div>
        ))}
      </div>

      {/* Shift detail section */}
      {topOverallShifts.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-gray-700 mb-2">
            Notable Composition Shifts
          </div>
          <ul className="space-y-1.5" aria-label="Notable cohort composition shifts">
            {topOverallShifts.map((shift, idx) => (
              <li key={`${shift.column}-${shift.category}-${idx}`}>
                <ShiftRow shift={shift} />
              </li>
            ))}
          </ul>
        </div>
      )}

      {topOverallShifts.length === 0 && (
        <p className="text-xs text-gray-500 italic">
          No categorical segments shifted more than 5 percentage points across these periods.
        </p>
      )}

      <figcaption className="sr-only">
        Predictive cohort evolution tracking how the top-{result.n} {result.direction}{" "}
        {result.target_column} predictions changed across {result.n_periods} dataset uploads.{" "}
        {result.summary}
      </figcaption>
    </figure>
  )
}
