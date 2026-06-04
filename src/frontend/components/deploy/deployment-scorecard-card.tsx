"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { DeploymentScorecardResult, DeploymentScorecardEntry } from "@/lib/types"

interface Props {
  result: DeploymentScorecardResult
}

const RANK_MEDALS: Record<number, string> = { 1: "🥇", 2: "🥈", 3: "🥉" }

function ScoreBar({ score, colorClass }: { score: number; colorClass: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${colorClass}`}
          style={{ width: `${score}%` }}
          aria-hidden="true"
        />
      </div>
      <span className="text-xs font-mono font-medium tabular-nums w-7 text-right">{score}</span>
    </div>
  )
}

function SignalPill({
  icon,
  label,
  value,
  available,
}: {
  icon: string
  label: string
  value: string
  available: boolean
}) {
  return (
    <div
      className={`flex items-center gap-1 text-xs rounded px-1.5 py-0.5 ${available ? "bg-muted text-foreground" : "bg-muted/50 text-muted-foreground"}`}
      title={label}
    >
      <span aria-hidden="true">{icon}</span>
      <span>{value}</span>
    </div>
  )
}

function EntryRow({
  entry,
  isWinner,
}: {
  entry: DeploymentScorecardEntry
  isWinner: boolean
}) {
  const medal = RANK_MEDALS[entry.rank]
  const barColor =
    entry.composite_score >= 70
      ? "bg-emerald-500"
      : entry.composite_score >= 45
        ? "bg-amber-500"
        : "bg-rose-500"

  const envBadgeClass =
    entry.environment === "production"
      ? "bg-emerald-100 text-emerald-800 border-emerald-200"
      : "bg-amber-100 text-amber-700 border-amber-200"

  return (
    <div
      data-testid={`scorecard-entry-${entry.rank}`}
      className={`rounded-lg border p-3 space-y-2 ${isWinner ? "border-amber-400/60 bg-amber-50/60" : "border-border bg-background"}`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="text-lg leading-none"
            aria-label={medal ? `Rank ${entry.rank}` : undefined}
            role={medal ? "img" : undefined}
          >
            {medal ?? (
              <span className="text-xs font-bold text-muted-foreground w-5 inline-block text-center">
                #{entry.rank}
              </span>
            )}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-foreground truncate">
              {entry.algorithm_plain}
              <span className="text-muted-foreground font-normal"> → </span>
              {entry.target_column}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {isWinner && (
            <Badge className="bg-amber-100 text-amber-800 border-amber-300 text-xs" variant="outline">
              Top Performer
            </Badge>
          )}
          <Badge className={`text-xs border ${envBadgeClass}`} variant="outline">
            {entry.environment === "production" ? "Production" : "Staging"}
          </Badge>
        </div>
      </div>

      {/* Composite score bar */}
      <div>
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs text-muted-foreground">Composite Score</span>
        </div>
        <ScoreBar score={entry.composite_score} colorClass={barColor} />
      </div>

      {/* Signal pills */}
      <div className="flex flex-wrap gap-1.5">
        <SignalPill
          icon="🔢"
          label="Prediction volume"
          value={`${entry.request_count.toLocaleString()} predictions`}
          available={entry.request_count > 0}
        />
        <SignalPill
          icon="📅"
          label="Model freshness"
          value={`${entry.age_days}d old`}
          available={entry.age_days < 90}
        />
        {entry.accuracy_score !== null ? (
          <SignalPill
            icon="🎯"
            label="Feedback accuracy"
            value={`${entry.accuracy_score}% accurate`}
            available={true}
          />
        ) : (
          <SignalPill icon="🎯" label="No feedback yet" value="No feedback" available={false} />
        )}
        {entry.sla_score !== null && entry.p95_ms !== null ? (
          <SignalPill
            icon="⚡"
            label="SLA p95 latency"
            value={`p95 ${Math.round(entry.p95_ms)}ms`}
            available={entry.sla_score >= 50}
          />
        ) : (
          <SignalPill icon="⚡" label="No SLA data" value="No SLA data" available={false} />
        )}
      </div>
    </div>
  )
}

export function DeploymentScorecardCard({ result }: Props) {
  if (result.total === 0) {
    return (
      <Card
        data-testid="deployment-scorecard-card"
        role="region"
        aria-label="Deployment Comparison Scorecard"
        className="border-slate-300"
      >
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">🏆</span>
            Deployment Comparison Scorecard
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{result.summary}</p>
        </CardContent>
        <figcaption className="sr-only">No active deployments found.</figcaption>
      </Card>
    )
  }

  return (
    <Card
      data-testid="deployment-scorecard-card"
      role="region"
      aria-label="Deployment Comparison Scorecard"
      className="border-amber-300/60"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <span aria-hidden="true">🏆</span>
          Deployment Leaderboard
          <Badge className="bg-amber-100 text-amber-800 border-amber-200 ml-1" variant="outline">
            {result.total} deployment{result.total !== 1 ? "s" : ""}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div data-testid="scorecard-entries" className="space-y-2">
          {result.entries.map((entry) => (
            <EntryRow
              key={entry.deployment_id}
              entry={entry}
              isWinner={entry.deployment_id === result.winner_id}
            />
          ))}
        </div>
        <p className="text-xs text-muted-foreground pt-1 border-t">{result.summary}</p>
      </CardContent>
      <figcaption className="sr-only">
        {result.total} deployments ranked by composite production score. {result.summary}
      </figcaption>
    </Card>
  )
}
