"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ModelQualityScoreResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

const COLOR_CLASSES: Record<string, { border: string; badge: string; bar: string }> = {
  emerald: {
    border: "border-emerald-400 dark:border-emerald-600",
    badge: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
    bar: "bg-emerald-500",
  },
  blue: {
    border: "border-blue-400 dark:border-blue-600",
    badge: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    bar: "bg-blue-500",
  },
  amber: {
    border: "border-amber-400 dark:border-amber-600",
    badge: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
    bar: "bg-amber-500",
  },
  rose: {
    border: "border-rose-400 dark:border-rose-600",
    badge: "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-200",
    bar: "bg-rose-500",
  },
}

const LABEL_ICONS: Record<string, string> = {
  "Excellent": "🏆",
  "Good": "✅",
  "Acceptable": "⚠️",
  "Needs Work": "🔧",
}

function getColors(color: string) {
  return COLOR_CLASSES[color] ?? COLOR_CLASSES.rose
}

// ---------------------------------------------------------------------------
// ModelQualityScoreCard
// ---------------------------------------------------------------------------

interface Props {
  result: ModelQualityScoreResult
}

export function ModelQualityScoreCard({ result }: Props) {
  const {
    quality_label,
    quality_score,
    primary_metric,
    primary_metric_name,
    color,
    reasoning,
    recommendation,
    is_stable,
  } = result

  const colors = getColors(color)
  const icon = LABEL_ICONS[quality_label] ?? "📊"
  const pctDisplay =
    primary_metric_name === "R²"
      ? `${Math.round(primary_metric * 100)}%`
      : `${Math.round(primary_metric * 100)}%`

  return (
    <Card
      className={`border-2 ${colors.border} mt-2`}
      role="region"
      aria-label="Model quality score"
    >
      <figcaption className="sr-only">
        Model quality assessment: {quality_label} — {primary_metric_name} {pctDisplay}
      </figcaption>

      <CardHeader className="pb-2">
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
          <span aria-hidden="true">{icon}</span>
          <span>Model Quality Score</span>
          <span
            className={`ml-auto inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${colors.badge}`}
          >
            {quality_label}
          </span>
          {!is_stable && (
            <Badge variant="outline" className="border-amber-400 text-amber-700 text-xs">
              ⚡ Unstable
            </Badge>
          )}
          <span className="text-xs text-muted-foreground font-normal">
            {primary_metric_name}: {pctDisplay}
          </span>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Quality score bar */}
        <div>
          <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
            <span>Quality score</span>
            <span className="font-semibold text-foreground">{quality_score}/100</span>
          </div>
          <div
            className="h-2 w-full rounded-full bg-muted overflow-hidden"
            role="progressbar"
            aria-valuenow={quality_score}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Quality score: ${quality_score} out of 100`}
          >
            <div
              className={`h-full rounded-full transition-all ${colors.bar}`}
              style={{ width: `${quality_score}%` }}
            />
          </div>
        </div>

        {/* Reasoning bullets */}
        {reasoning.length > 0 && (
          <ul className="space-y-1" aria-label="Quality reasoning">
            {reasoning.map((bullet, i) => (
              <li key={i} className="flex gap-2 text-xs text-muted-foreground">
                <span aria-hidden="true" className="mt-0.5 shrink-0 text-foreground/50">
                  •
                </span>
                <span>{bullet}</span>
              </li>
            ))}
          </ul>
        )}

        {/* Recommendation */}
        <div className="rounded-md border border-border bg-muted/40 px-3 py-2">
          <p className="text-xs">
            <span className="font-semibold text-foreground" aria-hidden="true">
              💡 Next step:{" "}
            </span>
            <span className="text-muted-foreground">{recommendation}</span>
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// ModelQualityBadge — inline badge for RunCard in the training panel
// ---------------------------------------------------------------------------

interface BadgeProps {
  metrics: Record<string, unknown>
  problemType: string
}

export function ModelQualityBadge({ metrics, problemType }: BadgeProps) {
  const m = metrics as Record<string, number | undefined>
  const primary =
    problemType === "regression"
      ? (m.r2 ?? 0)
      : (m.accuracy ?? m.f1 ?? 0)
  const cvStd = m.cv_std

  // Determine label & color without calling the server
  let label: string
  let badgeClass: string

  if (problemType === "regression") {
    if (primary >= 0.85) { label = "Excellent"; badgeClass = "border-emerald-400 text-emerald-700 dark:text-emerald-400" }
    else if (primary >= 0.70) { label = "Good"; badgeClass = "border-blue-400 text-blue-700 dark:text-blue-400" }
    else if (primary >= 0.55) { label = "Acceptable"; badgeClass = "border-amber-400 text-amber-700 dark:text-amber-400" }
    else { label = "Needs Work"; badgeClass = "border-rose-400 text-rose-700 dark:text-rose-400" }
  } else {
    if (primary >= 0.90) { label = "Excellent"; badgeClass = "border-emerald-400 text-emerald-700 dark:text-emerald-400" }
    else if (primary >= 0.80) { label = "Good"; badgeClass = "border-blue-400 text-blue-700 dark:text-blue-400" }
    else if (primary >= 0.70) { label = "Acceptable"; badgeClass = "border-amber-400 text-amber-700 dark:text-amber-400" }
    else { label = "Needs Work"; badgeClass = "border-rose-400 text-rose-700 dark:text-rose-400" }
  }

  // Downgrade one step if CV is unstable
  if (cvStd != null && cvStd > 0.10) {
    const order = ["Excellent", "Good", "Acceptable", "Needs Work"]
    const classMap: Record<string, string> = {
      "Excellent": "border-emerald-400 text-emerald-700 dark:text-emerald-400",
      "Good": "border-blue-400 text-blue-700 dark:text-blue-400",
      "Acceptable": "border-amber-400 text-amber-700 dark:text-amber-400",
      "Needs Work": "border-rose-400 text-rose-700 dark:text-rose-400",
    }
    const idx = order.indexOf(label)
    if (idx >= 0 && idx < order.length - 1) {
      label = order[idx + 1]
      badgeClass = classMap[label]
    }
  }

  return (
    <Badge
      variant="outline"
      className={`text-[10px] px-1.5 py-0 h-5 ${badgeClass}`}
      title={`Model quality: ${label}`}
      data-testid="model-quality-badge"
    >
      {label}
    </Badge>
  )
}
