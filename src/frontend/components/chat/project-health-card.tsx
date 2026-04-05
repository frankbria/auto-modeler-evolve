"use client"

import type { DeploymentHealthItem, ProjectHealthSummary } from "@/lib/types"

interface ProjectHealthCardProps {
  summary: ProjectHealthSummary
  onSwitchTab?: (tab: "deploy" | "models") => void
}

const STATUS_CONFIG = {
  healthy: {
    border: "border-emerald-200",
    bg: "bg-emerald-50",
    icon: "✅",
    label: "Healthy",
    badgeBg: "bg-emerald-100",
    badgeText: "text-emerald-800",
    titleText: "text-emerald-900",
    subtitleText: "text-emerald-700",
  },
  warning: {
    border: "border-amber-200",
    bg: "bg-amber-50",
    icon: "⚠️",
    label: "Needs Attention",
    badgeBg: "bg-amber-100",
    badgeText: "text-amber-800",
    titleText: "text-amber-900",
    subtitleText: "text-amber-700",
  },
  critical: {
    border: "border-red-200",
    bg: "bg-red-50",
    icon: "🚨",
    label: "Action Required",
    badgeBg: "bg-red-100",
    badgeText: "text-red-800",
    titleText: "text-red-900",
    subtitleText: "text-red-700",
  },
}

function HealthBar({ score }: { score: number }) {
  const color =
    score >= 75 ? "bg-emerald-500" : score >= 50 ? "bg-amber-400" : "bg-red-500"
  return (
    <div className="h-1.5 w-full rounded-full bg-gray-200" aria-hidden="true">
      <div
        className={`h-1.5 rounded-full ${color} transition-all`}
        style={{ width: `${score}%` }}
      />
    </div>
  )
}

function AlertRow({
  item,
  onSwitchTab,
}: {
  item: DeploymentHealthItem
  onSwitchTab?: (tab: "deploy" | "models") => void
}) {
  const cfg = STATUS_CONFIG[item.status]
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 text-xs space-y-2">
      {/* Name + environment */}
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold text-gray-800 truncate">{item.name}</span>
        <span
          className={`shrink-0 rounded-full px-2 py-0.5 font-medium text-xs ${cfg.badgeBg} ${cfg.badgeText}`}
        >
          {item.status}
        </span>
      </div>

      {/* Health bar */}
      <div className="space-y-0.5">
        <div className="flex justify-between text-gray-500">
          <span>Health score</span>
          <span className="font-medium text-gray-700">{item.health_score}/100</span>
        </div>
        <HealthBar score={item.health_score} />
      </div>

      {/* Issue + recommendation */}
      {item.top_issue && (
        <p className="text-gray-600">
          <span aria-hidden="true">⚠️ </span>
          {item.top_issue}
        </p>
      )}
      {item.recommendation && (
        <p className="text-gray-500 italic">{item.recommendation}</p>
      )}

      {/* CTAs */}
      <div className="flex gap-2 pt-1">
        {onSwitchTab && (
          <>
            <button
              onClick={() => onSwitchTab("deploy")}
              className="rounded px-2 py-1 bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors font-medium"
            >
              View Deployment
            </button>
            <button
              onClick={() => onSwitchTab("models")}
              className="rounded px-2 py-1 bg-blue-50 text-blue-700 hover:bg-blue-100 transition-colors font-medium"
            >
              Retrain Model
            </button>
          </>
        )}
      </div>
    </div>
  )
}

export function ProjectHealthCard({ summary, onSwitchTab }: ProjectHealthCardProps) {
  const cfg = STATUS_CONFIG[summary.overall_status]

  if (summary.total === 0) {
    return null
  }

  return (
    <figure
      className={`mt-3 rounded-xl border ${cfg.border} ${cfg.bg} p-4 shadow-sm max-w-md`}
      aria-label="Model health summary"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">
          {cfg.icon}
        </span>
        <div>
          <p className={`font-semibold text-sm leading-tight ${cfg.titleText}`}>
            Model Health: {cfg.label}
          </p>
          <p className={`text-xs ${cfg.subtitleText}`}>{summary.summary}</p>
        </div>
      </div>

      {/* Count badges */}
      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-medium">
          {summary.total} deployment{summary.total !== 1 ? "s" : ""}
        </span>
        {summary.healthy > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-medium">
            {summary.healthy} healthy
          </span>
        )}
        {summary.warning > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 font-medium">
            {summary.warning} warning
          </span>
        )}
        {summary.critical > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-red-100 text-red-800 font-medium">
            {summary.critical} critical
          </span>
        )}
      </div>

      {/* Alert rows (warning + critical only) */}
      {summary.alerts.length > 0 && (
        <div className="space-y-2">
          {summary.alerts.map((item) => (
            <AlertRow key={item.deployment_id} item={item} onSwitchTab={onSwitchTab} />
          ))}
        </div>
      )}
    </figure>
  )
}
