/**
 * DeploymentChangelogCard — shows the immutable audit trail of a deployment.
 *
 * Analysts can ask "what changed to my deployment?" or "show my deployment changelog"
 * to see a chronological log of lifecycle events: when the model was first deployed,
 * whether it was retrained, whether API key protection was toggled, etc.
 *
 * This closes the "smart colleague" gap: a colleague who can answer "nothing has changed
 * since last Tuesday" or "your model was updated to version 3 yesterday morning."
 */
"use client"

import { CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { DeploymentChangelogResult, DeploymentChangelogEntry } from "@/lib/types"

// ── change_type → icon/label/color maps ──────────────────────────────────────

const CHANGE_META: Record<
  string,
  { icon: string; label: string; badgeClass: string }
> = {
  deployed: {
    icon: "🚀",
    label: "Deployed",
    badgeClass: "bg-emerald-100 text-emerald-800 border-emerald-200",
  },
  redeployed: {
    icon: "🔄",
    label: "Redeployed",
    badgeClass: "bg-blue-100 text-blue-800 border-blue-200",
  },
  retrained: {
    icon: "🤖",
    label: "Retrained",
    badgeClass: "bg-violet-100 text-violet-800 border-violet-200",
  },
  undeployed: {
    icon: "⏹️",
    label: "Undeployed",
    badgeClass: "bg-slate-100 text-slate-700 border-slate-200",
  },
  api_key_added: {
    icon: "🔑",
    label: "API key added",
    badgeClass: "bg-amber-100 text-amber-800 border-amber-200",
  },
  api_key_removed: {
    icon: "🔓",
    label: "API key removed",
    badgeClass: "bg-orange-100 text-orange-800 border-orange-200",
  },
  rate_limit_set: {
    icon: "⏱️",
    label: "Rate limit set",
    badgeClass: "bg-sky-100 text-sky-800 border-sky-200",
  },
  quota_set: {
    icon: "📊",
    label: "Quota set",
    badgeClass: "bg-sky-100 text-sky-800 border-sky-200",
  },
  alert_rule_added: {
    icon: "🔔",
    label: "Alert rule added",
    badgeClass: "bg-rose-100 text-rose-800 border-rose-200",
  },
  field_config_updated: {
    icon: "⚙️",
    label: "Field config updated",
    badgeClass: "bg-gray-100 text-gray-700 border-gray-200",
  },
  batch_run_complete: {
    icon: "📦",
    label: "Batch run",
    badgeClass: "bg-teal-100 text-teal-800 border-teal-200",
  },
}

function getChangeMeta(changeType: string) {
  return (
    CHANGE_META[changeType] ?? {
      icon: "📝",
      label: changeType.replace(/_/g, " "),
      badgeClass: "bg-gray-100 text-gray-700 border-gray-200",
    }
  )
}

// ── Single timeline entry ────────────────────────────────────────────────────

function ChangeEntry({
  entry,
  isLast,
}: {
  entry: DeploymentChangelogEntry
  isLast: boolean
}) {
  const { icon, label, badgeClass } = getChangeMeta(entry.change_type)

  return (
    <div className="flex gap-3" data-testid={`changelog-entry-${entry.change_type}`}>
      {/* Timeline line */}
      <div className="flex flex-col items-center">
        <div
          className="w-8 h-8 rounded-full border-2 border-muted bg-background flex items-center justify-center text-sm flex-shrink-0"
          aria-hidden="true"
        >
          {icon}
        </div>
        {!isLast && <div className="w-px flex-1 bg-border mt-1" aria-hidden="true" />}
      </div>

      {/* Content */}
      <div className={`pb-4 ${isLast ? "" : ""} min-w-0 flex-1`}>
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <Badge className={`text-[10px] border ${badgeClass}`}>{label}</Badge>
          <span className="text-xs text-muted-foreground">{entry.relative_time}</span>
        </div>
        <p className="text-xs text-foreground leading-relaxed">{entry.description}</p>
      </div>
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="text-center py-6 space-y-2" data-testid="changelog-empty-state">
      <p className="text-2xl" aria-hidden="true">
        📋
      </p>
      <p className="text-sm text-muted-foreground">No changes recorded yet.</p>
      <p className="text-xs text-muted-foreground">
        The changelog will show events like deployments, retraining, and API key changes
        as they happen.
      </p>
    </div>
  )
}

// ── Main card ─────────────────────────────────────────────────────────────────

export function DeploymentChangelogCard({
  result,
}: {
  result: DeploymentChangelogResult
}) {
  const { count, entries } = result

  return (
    <>
      <CardHeader className="pb-3 pt-4 px-4">
        <div className="flex items-center gap-2 flex-wrap">
          <CardTitle
            className="text-sm font-semibold flex items-center gap-1.5"
            data-testid="changelog-card-title"
          >
            <span aria-hidden="true">📋</span> Deployment Changelog
          </CardTitle>
          <Badge
            className="bg-slate-100 text-slate-700 border-slate-200 text-[10px]"
            data-testid="changelog-count-badge"
          >
            {count} {count === 1 ? "event" : "events"}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Audit trail of changes made to this deployment — newest first.
        </p>
      </CardHeader>

      <CardContent className="px-4 pb-4">
        {entries.length === 0 ? (
          <EmptyState />
        ) : (
          <div
            className="space-y-0"
            role="list"
            aria-label="Deployment change history"
          >
            {entries.map((entry, i) => (
              <div key={entry.id} role="listitem">
                <ChangeEntry entry={entry} isLast={i === entries.length - 1} />
              </div>
            ))}
          </div>
        )}

        <figcaption className="sr-only">
          Deployment changelog with {count} event{count !== 1 ? "s" : ""}. Most recent
          change:{" "}
          {entries.length > 0
            ? `${entries[0].change_type.replace(/_/g, " ")} — ${entries[0].relative_time}`
            : "none recorded"}
          .
        </figcaption>
      </CardContent>
    </>
  )
}
