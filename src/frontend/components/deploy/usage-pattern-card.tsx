import type { UsagePatternResult } from "@/lib/types"

interface UsagePatternCardProps {
  result: UsagePatternResult
}

function fmt12h(hour: number): string {
  if (hour === 0) return "12am"
  if (hour < 12) return `${hour}am`
  if (hour === 12) return "12pm"
  return `${hour - 12}pm`
}

function HourBar({ count, max, hour }: { count: number; max: number; hour: number }) {
  const pct = max > 0 ? Math.round((count / max) * 100) : 0
  const isPeak = count === max && max > 0
  const color = isPeak ? "bg-indigo-500" : pct >= 60 ? "bg-indigo-300" : pct >= 20 ? "bg-indigo-200" : "bg-gray-100"
  return (
    <div className="flex flex-col items-center gap-0.5" title={`${fmt12h(hour)}: ${count} predictions`}>
      <div className="w-4 flex items-end" style={{ height: 40 }}>
        <div
          className={`w-full rounded-t transition-all ${color}`}
          style={{ height: `${Math.max(2, pct)}%` }}
          aria-label={`${fmt12h(hour)}: ${count} predictions`}
        />
      </div>
      {hour % 6 === 0 && (
        <span className="text-[9px] text-gray-400">{fmt12h(hour)}</span>
      )}
    </div>
  )
}

function DayBar({ count, max, label }: { count: number; max: number; label: string }) {
  const pct = max > 0 ? Math.round((count / max) * 100) : 0
  const isPeak = count === max && max > 0
  const color = isPeak ? "bg-indigo-500" : pct >= 60 ? "bg-indigo-300" : pct >= 20 ? "bg-indigo-200" : "bg-gray-100"
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="w-6 flex items-end" style={{ height: 36 }}>
        <div
          className={`w-full rounded-t transition-all ${color}`}
          style={{ height: `${Math.max(2, pct)}%` }}
          aria-label={`${label}: ${count} predictions`}
        />
      </div>
      <span className="text-[10px] text-gray-500 font-medium">{label}</span>
      <span className="text-[9px] text-gray-400">{count}</span>
    </div>
  )
}

export function UsagePatternCard({ result }: UsagePatternCardProps) {
  const isEmpty = result.total_predictions === 0

  const maxHour = Math.max(...result.hour_counts, 1)
  const maxDay = Math.max(...result.day_counts, 1)
  const dayNames = result.day_names ?? ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

  const quietWindowSuggestion =
    result.quiet_hours.length > 0
      ? result.quiet_hours.slice(0, 3).map(fmt12h).join(", ") +
        (result.quiet_hours.length > 3 ? ` +${result.quiet_hours.length - 3} more` : "")
      : null

  return (
    <figure
      className="rounded-lg border border-indigo-300 bg-indigo-50 p-4 my-2 space-y-3"
      aria-label="Prediction usage pattern analysis"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-lg" aria-hidden="true">🕐</span>
        <span className="font-semibold text-sm">Prediction Usage Patterns</span>
        {!isEmpty && result.peak_hour !== null && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 font-medium">
            Peak: {fmt12h(result.peak_hour)} UTC
          </span>
        )}
        {!isEmpty && result.peak_day_short && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 font-medium">
            Busiest: {result.peak_day_name}s
          </span>
        )}
        <span className="text-xs px-2 py-0.5 rounded-full bg-sky-100 text-sky-700">
          {result.total_predictions.toLocaleString()} predictions
        </span>
      </div>

      {isEmpty ? (
        <p className="text-sm text-gray-500 italic">
          No predictions recorded yet. Usage patterns will appear once the model starts receiving requests.
        </p>
      ) : (
        <>
          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">Hour of Day (UTC)</p>
            <div className="flex items-end gap-0.5" aria-label="Hour-of-day prediction distribution">
              {result.hour_counts.map((count, hour) => (
                <HourBar key={hour} count={count} max={maxHour} hour={hour} />
              ))}
            </div>
          </div>

          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">Day of Week</p>
            <div className="flex items-end gap-2" aria-label="Day-of-week prediction distribution">
              {result.day_counts.map((count, idx) => (
                <DayBar key={idx} count={count} max={maxDay} label={dayNames[idx]} />
              ))}
            </div>
          </div>

          {result.busiest_period && (
            <div className="flex items-start gap-2 text-xs text-indigo-800 bg-indigo-100 rounded p-2">
              <span aria-hidden="true">📈</span>
              <span>Busiest period: <strong>{result.busiest_period}</strong></span>
            </div>
          )}

          {quietWindowSuggestion && (
            <div className="flex items-start gap-2 text-xs text-emerald-800 bg-emerald-50 border border-emerald-200 rounded p-2">
              <span aria-hidden="true">🔧</span>
              <span>
                Suggested maintenance window: <strong>{quietWindowSuggestion} UTC</strong> — lowest usage
              </span>
            </div>
          )}

          <p className="text-xs text-gray-600 italic">{result.summary}</p>
        </>
      )}

      <figcaption className="sr-only">
        Prediction usage pattern: {result.total_predictions} total predictions.
        {result.peak_hour !== null && ` Peak hour: ${fmt12h(result.peak_hour)} UTC.`}
        {result.peak_day_name && ` Busiest day: ${result.peak_day_name}.`}
      </figcaption>
    </figure>
  )
}
