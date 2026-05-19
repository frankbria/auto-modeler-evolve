import type { DashboardMetadataResult } from "@/lib/types"

interface DashboardMetadataCardProps {
  result: DashboardMetadataResult
}

export function DashboardMetadataCard({ result }: DashboardMetadataCardProps) {
  const { action, dashboard_title, dashboard_description, auto_title, summary } = result

  const borderColor =
    action === "cleared"
      ? "border-slate-200 bg-slate-50/50"
      : action === "status"
        ? "border-sky-200 bg-sky-50/50"
        : "border-emerald-200 bg-emerald-50/50"

  const icon =
    action === "cleared"
      ? "🔄"
      : action === "status"
        ? "🏷️"
        : "✏️"

  const heading =
    action === "title_set"
      ? "Dashboard Title Set"
      : action === "description_set"
        ? "Dashboard Description Set"
        : action === "both_set"
          ? "Dashboard Branding Updated"
          : action === "cleared"
            ? "Dashboard Title Cleared"
            : "Dashboard Title Status"

  return (
    <div
      role="region"
      className={`mt-2 rounded-lg border p-3 text-sm ${borderColor}`}
      aria-label="Dashboard metadata card"
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">{icon}</span>
        <span className="font-semibold text-slate-800">{heading}</span>
        {action !== "status" && action !== "cleared" && (
          <span className="ml-auto rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
            Updated
          </span>
        )}
      </div>

      {/* Current title / description */}
      <div className="mb-2 space-y-1">
        <div className="flex items-start gap-2 text-xs">
          <span className="w-20 shrink-0 font-medium text-slate-500">Title</span>
          {dashboard_title ? (
            <span
              className="rounded bg-slate-100 px-2 py-0.5 font-semibold text-slate-800"
              data-testid="dashboard-title-display"
            >
              {dashboard_title}
            </span>
          ) : (
            <span className="italic text-slate-400" data-testid="dashboard-title-auto">
              {auto_title} <span className="text-slate-300">(auto-generated)</span>
            </span>
          )}
        </div>
        {(dashboard_description || action === "status") && (
          <div className="flex items-start gap-2 text-xs">
            <span className="w-20 shrink-0 font-medium text-slate-500">Description</span>
            {dashboard_description ? (
              <span className="text-slate-700" data-testid="dashboard-description-display">
                {dashboard_description}
              </span>
            ) : (
              <span className="italic text-slate-400">not set</span>
            )}
          </div>
        )}
      </div>

      {/* Summary */}
      <p className="text-xs text-slate-600">{summary}</p>

      {/* Footer hint */}
      <p className="mt-1.5 text-xs italic text-slate-400">
        {action === "status"
          ? "To change, say \"set the dashboard title to '…'\" or \"add a dashboard description: '…'\""
          : "Changes are reflected immediately on the shared prediction URL."}
      </p>
    </div>
  )
}
