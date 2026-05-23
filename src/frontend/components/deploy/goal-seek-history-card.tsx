/**
 * GoalSeekHistoryCard — shows the last N goal-seek scenarios for a deployment.
 *
 * Analysts can say "show my past goal seek results" or "compare my scenarios"
 * to see a timeline of targets they've explored, side-by-side, so they can
 * understand how different targets require different input changes.
 */
"use client"

import { CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { GoalSeekHistoryResult, GoalSeekHistoryEntry, GoalSeekSuggestion } from "@/lib/types"

// Format a display value — truncate very long strings
function fmtVal(v: string): string {
  const n = Number(v)
  if (!isNaN(n) && v.trim() !== "") {
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  return v
}

// Human-readable relative time
function relativeTime(isoStr: string): string {
  try {
    const diff = Date.now() - new Date(isoStr + "Z").getTime()
    const mins = Math.round(diff / 60_000)
    if (mins < 1) return "just now"
    if (mins < 60) return `${mins}m ago`
    const hrs = Math.round(mins / 60)
    if (hrs < 24) return `${hrs}h ago`
    return `${Math.round(hrs / 24)}d ago`
  } catch {
    return isoStr
  }
}

function TopSuggestions({ suggestions }: { suggestions: GoalSeekSuggestion[] }) {
  if (suggestions.length === 0) {
    return <p className="text-xs italic text-muted-foreground">No suggestions.</p>
  }
  return (
    <ul className="space-y-0.5">
      {suggestions.slice(0, 3).map((s) => {
        const arrow =
          s.direction === "increase" ? "↑" : s.direction === "decrease" ? "↓" : "→"
        const color =
          s.direction === "increase"
            ? "text-emerald-700"
            : s.direction === "decrease"
              ? "text-rose-700"
              : "text-slate-500"
        return (
          <li key={s.feature} className="flex items-center gap-1.5 text-xs">
            <span className={`font-bold ${color}`} aria-hidden="true">
              {arrow}
            </span>
            <span className="font-medium text-foreground truncate max-w-[100px]" title={s.feature}>
              {s.feature.replace(/_/g, " ")}
            </span>
            <span className="text-muted-foreground font-mono">
              → {fmtVal(String(s.suggested_value))}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

function EntryCard({
  entry,
  index,
}: {
  entry: GoalSeekHistoryEntry
  index: number
}) {
  const borderColor = entry.achieved
    ? "border-emerald-200 bg-emerald-50/40"
    : "border-amber-200 bg-amber-50/30"

  const achievedBadge = entry.achieved ? (
    <Badge
      data-testid={`history-achieved-badge-${index}`}
      className="bg-emerald-100 text-emerald-800 border-emerald-200 text-[10px]"
    >
      ✓ Achieved
    </Badge>
  ) : (
    <Badge
      data-testid={`history-best-effort-badge-${index}`}
      className="bg-amber-100 text-amber-800 border-amber-200 text-[10px]"
    >
      Best effort
    </Badge>
  )

  return (
    <div
      className={`rounded-lg border p-3 space-y-2 ${borderColor}`}
      data-testid={`history-entry-${index}`}
    >
      {/* Header row */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold text-muted-foreground">#{index + 1}</span>
          <span className="text-xs text-muted-foreground">{relativeTime(entry.created_at)}</span>
        </div>
        {achievedBadge}
      </div>

      {/* Target vs Achieved */}
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded bg-white/70 p-1.5 text-center border border-border/30">
          <div className="text-[9px] text-muted-foreground uppercase tracking-wide mb-0.5">
            Target
          </div>
          <div
            data-testid={`history-target-${index}`}
            className="text-sm font-bold font-mono text-foreground"
          >
            {fmtVal(entry.target_value_str)}
          </div>
        </div>
        <div className="rounded bg-white/70 p-1.5 text-center border border-border/30">
          <div className="text-[9px] text-muted-foreground uppercase tracking-wide mb-0.5">
            Achieved
          </div>
          <div
            data-testid={`history-achieved-${index}`}
            className={`text-sm font-bold font-mono ${
              entry.achieved ? "text-emerald-700" : "text-amber-700"
            }`}
          >
            {fmtVal(entry.achieved_value_str)}
          </div>
        </div>
      </div>

      {/* Gap */}
      {entry.gap_pct !== null && !entry.achieved && (
        <p className="text-[10px] text-center text-amber-700 bg-amber-50 rounded px-2 py-0.5">
          {entry.gap_pct}% gap from target
        </p>
      )}

      {/* Top suggestions */}
      {entry.suggestions.length > 0 && (
        <div>
          <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Key changes needed
          </div>
          <TopSuggestions suggestions={entry.suggestions} />
        </div>
      )}

      {/* Fixed features */}
      {Object.keys(entry.fixed_features).length > 0 && (
        <p className="text-[10px] text-muted-foreground">
          <span className="font-medium">Pinned: </span>
          {Object.entries(entry.fixed_features)
            .map(([k, v]) => `${k.replace(/_/g, " ")}=${v}`)
            .join(", ")}
        </p>
      )}
    </div>
  )
}

interface GoalSeekHistoryCardProps {
  result: GoalSeekHistoryResult
}

export function GoalSeekHistoryCard({ result }: GoalSeekHistoryCardProps) {
  const hasEntries = result.count > 0

  return (
    <figure
      className="rounded-xl border-2 border-violet-300 bg-card p-4 shadow-sm w-full"
      aria-label="Goal seek history"
      data-testid="goal-seek-history-card"
    >
      {/* Header */}
      <CardHeader className="p-0 pb-3">
        <CardTitle className="text-sm flex items-center gap-2 flex-wrap">
          <span aria-hidden="true">🎯</span>
          <span>Goal Seek History</span>
          {hasEntries && (
            <Badge
              data-testid="history-count-badge"
              className="bg-violet-100 text-violet-800 border-violet-200 text-[10px]"
            >
              {result.count} scenario{result.count !== 1 ? "s" : ""}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 space-y-3">
        {hasEntries ? (
          <>
            <p className="text-xs text-muted-foreground italic">
              Showing your last {result.count} goal-seek scenario
              {result.count !== 1 ? "s" : ""}. Run a new goal seek to add it here.
            </p>

            {result.entries.map((entry, i) => (
              <EntryCard key={entry.id} entry={entry} index={i} />
            ))}
          </>
        ) : (
          <p
            data-testid="history-empty-state"
            className="text-sm text-muted-foreground italic py-2"
          >
            No goal-seek runs yet. Try asking: &ldquo;What inputs would produce revenue of
            $5M?&rdquo; or &ldquo;What do I need to change to achieve 90% accuracy?&rdquo;
          </p>
        )}

        <figcaption className="sr-only">
          Goal seek history:{" "}
          {hasEntries
            ? `${result.count} scenario${result.count !== 1 ? "s" : ""} recorded. ` +
              result.entries
                .map(
                  (e, i) =>
                    `Scenario ${i + 1}: target ${e.target_value_str}, achieved ${e.achieved_value_str}, ${e.achieved ? "goal met" : "best effort"}.`
                )
                .join(" ")
            : "No scenarios recorded yet."}
        </figcaption>
      </CardContent>
    </figure>
  )
}
