"use client"

import { Badge } from "@/components/ui/badge"
import type { PromotionReadinessGate, PromotionReadinessResult } from "@/lib/types"

interface PromotionReadinessCardProps {
  result: PromotionReadinessResult
  onActionClick?: (msg: string) => void
}

function GateRow({ gate }: { gate: PromotionReadinessGate }) {
  const statusIcon =
    gate.status === "pass" ? "✓" : gate.status === "warn" ? "⚠" : "✗"
  const iconColor =
    gate.status === "pass"
      ? "text-emerald-600"
      : gate.status === "warn"
        ? "text-amber-600"
        : "text-rose-600"
  const rowBg =
    gate.status === "pass"
      ? "bg-emerald-50 dark:bg-emerald-950/10"
      : gate.status === "warn"
        ? "bg-amber-50 dark:bg-amber-950/10"
        : "bg-rose-50 dark:bg-rose-950/10"

  return (
    <li
      className={`flex flex-col gap-1 rounded-md px-3 py-2 text-sm ${rowBg}`}
      data-testid={`gate-row-${gate.gate}`}
    >
      <div className="flex items-start gap-2">
        <span
          className={`font-bold mt-0.5 ${iconColor}`}
          aria-hidden="true"
        >
          {statusIcon}
        </span>
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{gate.label}</span>
            <Badge
              variant="outline"
              className={`text-xs ${
                gate.status === "pass"
                  ? "border-emerald-400 text-emerald-700"
                  : gate.status === "warn"
                    ? "border-amber-400 text-amber-700"
                    : "border-rose-400 text-rose-700"
              }`}
            >
              {gate.status === "pass" ? "Pass" : gate.status === "warn" ? "Warning" : "Fail"}
            </Badge>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mt-0.5">{gate.detail}</p>
          {gate.recommendation && (
            <p className="text-gray-500 dark:text-gray-500 italic mt-1 text-xs">
              → {gate.recommendation}
            </p>
          )}
        </div>
      </div>
    </li>
  )
}

export function PromotionReadinessCard({
  result,
  onActionClick,
}: PromotionReadinessCardProps) {
  const {
    overall_verdict,
    verdict_label,
    passed_count,
    warn_count,
    fail_count,
    total_count,
    gates,
    blocking_issues,
    algorithm,
    problem_type,
    primary_metric,
    primary_metric_name,
    summary,
  } = result

  const borderColor =
    overall_verdict === "ready"
      ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20"
      : overall_verdict === "ready_with_warnings"
        ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20"
        : "border-rose-500 bg-rose-50 dark:bg-rose-950/20"

  const verdictBadgeClass =
    overall_verdict === "ready"
      ? "bg-emerald-600 text-white"
      : overall_verdict === "ready_with_warnings"
        ? "bg-amber-600 text-white"
        : "bg-rose-600 text-white"

  const verdictIcon =
    overall_verdict === "ready" ? "🚀" : overall_verdict === "ready_with_warnings" ? "⚠️" : "🛑"

  const metricDisplay =
    primary_metric_name === "R²"
      ? primary_metric.toFixed(3)
      : `${(primary_metric * 100).toFixed(1)}%`

  return (
    <div
      className={`rounded-lg border-2 p-4 mt-2 ${borderColor}`}
      role="region"
      aria-label="Model promotion readiness assessment"
      data-testid="promotion-readiness-card"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-lg">
          {verdictIcon}
        </span>
        <span className="font-semibold text-sm">Promotion Readiness Check</span>
        <Badge className={`text-xs ${verdictBadgeClass}`} data-testid="verdict-badge">
          {verdict_label}
        </Badge>
        <Badge variant="secondary" className="text-xs">
          {algorithm}
        </Badge>
        <Badge variant="outline" className="text-xs">
          {problem_type}
        </Badge>
        <Badge variant="outline" className="text-xs">
          {primary_metric_name}: {metricDisplay}
        </Badge>
      </div>

      {/* Gate score summary */}
      <div className="flex flex-wrap gap-3 mb-3 text-sm">
        <div className="flex items-center gap-1 text-emerald-700">
          <span className="font-bold">{passed_count}</span>
          <span>passed</span>
        </div>
        {warn_count > 0 && (
          <div className="flex items-center gap-1 text-amber-700">
            <span className="font-bold">{warn_count}</span>
            <span>{warn_count === 1 ? "warning" : "warnings"}</span>
          </div>
        )}
        {fail_count > 0 && (
          <div className="flex items-center gap-1 text-rose-700">
            <span className="font-bold">{fail_count}</span>
            <span>{fail_count === 1 ? "blocker" : "blockers"}</span>
          </div>
        )}
        <span className="text-gray-400">of {total_count} gates</span>
      </div>

      {/* Blocking issues callout */}
      {blocking_issues.length > 0 && (
        <div
          className="mb-3 rounded-md bg-rose-100 dark:bg-rose-950/30 border border-rose-300 px-3 py-2"
          role="alert"
          data-testid="blocking-issues"
        >
          <p className="text-sm font-semibold text-rose-800 dark:text-rose-300 mb-1">
            Blocking issues ({blocking_issues.length})
          </p>
          <ul className="list-disc list-inside text-sm text-rose-700 dark:text-rose-400 space-y-0.5">
            {blocking_issues.map((issue) => (
              <li key={issue}>{issue}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Gate list */}
      <ul
        className="space-y-2 mb-3"
        role="list"
        aria-label="Readiness gates"
        data-testid="gate-list"
      >
        {gates.map((g) => (
          <GateRow key={g.gate} gate={g} />
        ))}
      </ul>

      {/* Summary */}
      <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">{summary}</p>

      {/* Action button when ready */}
      {(overall_verdict === "ready" || overall_verdict === "ready_with_warnings") &&
        onActionClick && (
          <button
            className="text-sm font-medium text-emerald-700 dark:text-emerald-400 underline hover:no-underline"
            onClick={() => onActionClick("deploy my model")}
            data-testid="deploy-action-button"
          >
            Deploy my model →
          </button>
        )}

      <figcaption className="sr-only">
        Model promotion readiness assessment: {verdict_label}. {passed_count} gates passed,{" "}
        {warn_count} warnings, {fail_count} blockers out of {total_count} total gates.
      </figcaption>
    </div>
  )
}
