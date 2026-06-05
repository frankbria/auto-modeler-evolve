"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type {
  DriftImportanceFeature,
  DriftImportancePriority,
  DriftImportanceRankingResult,
} from "@/lib/types"

interface Props {
  result: DriftImportanceRankingResult
}

const VERDICT_BORDER: Record<string, string> = {
  action_required: "border-rose-500",
  attention: "border-amber-400",
  monitoring: "border-sky-400",
  clear: "border-emerald-400",
  no_importances: "border-muted",
}

const VERDICT_BADGE: Record<string, { label: string; className: string }> = {
  action_required: { label: "Action Required", className: "bg-rose-100 text-rose-800" },
  attention: { label: "Attention", className: "bg-amber-100 text-amber-800" },
  monitoring: { label: "Monitoring", className: "bg-sky-100 text-sky-800" },
  clear: { label: "Clear", className: "bg-emerald-100 text-emerald-800" },
  no_importances: { label: "No Importances", className: "bg-muted text-muted-foreground" },
}

const PRIORITY_STYLE: Record<
  DriftImportancePriority,
  { badge: string; row: string }
> = {
  critical: { badge: "bg-rose-100 text-rose-800", row: "bg-rose-50/50" },
  high: { badge: "bg-amber-100 text-amber-800", row: "bg-amber-50/50" },
  medium: { badge: "bg-sky-100 text-sky-800", row: "" },
  low: { badge: "bg-slate-100 text-slate-700", row: "" },
  no_drift: { badge: "bg-emerald-50 text-emerald-700", row: "" },
}

const PRIORITY_LABEL: Record<DriftImportancePriority, string> = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
  no_drift: "No Drift",
}

function ImportanceBar({ pct }: { pct: number }) {
  return (
    <div className="w-16 bg-muted rounded-full h-1.5" aria-hidden="true">
      <div
        className="h-1.5 rounded-full bg-indigo-400"
        style={{ width: `${Math.min(pct, 100)}%` }}
      />
    </div>
  )
}

function DriftBar({ pct }: { pct: number }) {
  const color =
    pct >= 20 ? "bg-rose-400" : pct >= 10 ? "bg-amber-400" : pct >= 2 ? "bg-sky-400" : "bg-emerald-400"
  return (
    <div className="w-16 bg-muted rounded-full h-1.5" aria-hidden="true">
      <div
        className={`h-1.5 rounded-full ${color}`}
        style={{ width: `${Math.min(pct, 100)}%` }}
      />
    </div>
  )
}

function FeatureRow({ feature }: { feature: DriftImportanceFeature }) {
  const style = PRIORITY_STYLE[feature.priority]
  return (
    <tr
      className={`border-b last:border-0 text-sm ${style.row}`}
      data-testid={`feature-row-${feature.name}`}
    >
      <td className="py-2 pr-3 font-medium max-w-[140px] truncate" title={feature.name}>
        {feature.name}
      </td>
      <td className="py-2 pr-3">
        <Badge className={style.badge} data-testid={`priority-badge-${feature.name}`}>
          {PRIORITY_LABEL[feature.priority]}
        </Badge>
      </td>
      <td className="py-2 pr-3">
        <div className="flex items-center gap-1.5">
          <span className="tabular-nums text-xs w-9 text-right">
            {feature.importance_pct.toFixed(1)}%
          </span>
          <ImportanceBar pct={feature.importance_pct} />
        </div>
      </td>
      <td className="py-2 pr-3">
        <div className="flex items-center gap-1.5">
          <span className="tabular-nums text-xs w-9 text-right">
            {feature.drift_pct.toFixed(1)}%
          </span>
          <DriftBar pct={feature.drift_pct} />
        </div>
      </td>
      <td className="py-2 text-xs text-muted-foreground hidden md:table-cell">
        {feature.drift_details}
      </td>
    </tr>
  )
}

export function DriftImportanceCard({ result }: Props) {
  const borderColor = VERDICT_BORDER[result.verdict] ?? "border-muted"
  const badge = VERDICT_BADGE[result.verdict] ?? VERDICT_BADGE.no_importances

  return (
    <Card
      className={`border-2 ${borderColor} mt-2`}
      data-testid="drift-importance-card"
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <span aria-hidden="true">🎯</span>
            Drift × Importance Ranking
          </CardTitle>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge className={badge.className} data-testid="verdict-badge">
              {badge.label}
            </Badge>
            {result.has_importances && (
              <Badge variant="outline" data-testid="feature-count-badge">
                {result.n_features} feature{result.n_features !== 1 ? "s" : ""}
              </Badge>
            )}
            {result.n_samples > 0 && (
              <Badge variant="outline" data-testid="sample-count-badge">
                {result.n_samples.toLocaleString()} predictions
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {!result.has_importances ? (
          <p className="text-sm text-muted-foreground" data-testid="no-importances-message">
            {result.summary}
          </p>
        ) : (
          <>
            {(result.critical_count > 0 || result.high_count > 0) && (
              <div
                className="flex items-start gap-2 rounded-lg bg-rose-50 border border-rose-200 px-3 py-2 text-sm"
                role="alert"
                data-testid="priority-alert"
              >
                <span aria-hidden="true" className="mt-0.5">⚠️</span>
                <span>
                  {result.critical_count > 0 && (
                    <strong>{result.critical_count} critical</strong>
                  )}
                  {result.critical_count > 0 && result.high_count > 0 && " and "}
                  {result.high_count > 0 && (
                    <strong>{result.high_count} high-priority</strong>
                  )}{" "}
                  feature{result.critical_count + result.high_count !== 1 ? "s" : ""} need attention
                </span>
              </div>
            )}

            <div className="overflow-x-auto" data-testid="features-table">
              <table className="w-full text-sm" aria-label="Feature drift importance ranking">
                <thead>
                  <tr className="border-b text-xs text-muted-foreground">
                    <th className="pb-1 pr-3 text-left font-medium">Feature</th>
                    <th className="pb-1 pr-3 text-left font-medium">Priority</th>
                    <th className="pb-1 pr-3 text-left font-medium">Importance</th>
                    <th className="pb-1 pr-3 text-left font-medium">Drift</th>
                    <th className="pb-1 text-left font-medium hidden md:table-cell">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {result.ranked_features.map((f) => (
                    <FeatureRow key={f.name} feature={f} />
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        <p className="text-sm text-muted-foreground italic" data-testid="summary">
          {result.summary}
        </p>

        <figcaption className="sr-only">
          Drift importance ranking: {badge.label}.{" "}
          {result.critical_count} critical, {result.high_count} high-priority features.{" "}
          {result.summary}
        </figcaption>
      </CardContent>
    </Card>
  )
}
