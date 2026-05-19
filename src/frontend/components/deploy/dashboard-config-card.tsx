"use client"

import type { DashboardConfigResult, DashboardFieldChange } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function colLabel(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function FieldRow({ field }: { field: DashboardFieldChange }) {
  return (
    <div
      className="flex items-center justify-between py-1 text-sm"
      data-testid={`field-row-${field.feature_name}`}
    >
      <span className="font-medium text-foreground">{colLabel(field.feature_name)}</span>
      <div className="flex items-center gap-1.5">
        {field.display_label && (
          <Badge variant="outline" className="border-violet-300 text-violet-700 text-xs">
            {`→ "${field.display_label}"`}
          </Badge>
        )}
        {!field.is_visible && (
          <Badge variant="outline" className="border-slate-300 text-slate-500 text-xs">
            Hidden
          </Badge>
        )}
        {field.is_locked && (
          <Badge variant="outline" className="border-amber-300 text-amber-700 text-xs">
            Locked
            {field.locked_value ? ` = ${field.locked_value}` : ""}
          </Badge>
        )}
        {field.is_visible && !field.is_locked && !field.display_label && (
          <Badge variant="outline" className="border-emerald-300 text-emerald-700 text-xs">
            Visible
          </Badge>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main card
// ---------------------------------------------------------------------------

interface DashboardConfigCardProps {
  config: DashboardConfigResult
}

export function DashboardConfigCard({ config }: DashboardConfigCardProps) {
  const isReset = config.action === "reset"
  const isStatus = config.action === "status"
  const isLabeled = config.action === "labeled"

  const borderClass = isReset
    ? "border-slate-200 bg-slate-50"
    : isStatus
      ? "border-sky-200 bg-sky-50"
      : isLabeled
        ? "border-violet-200 bg-violet-50"
        : "border-emerald-200 bg-emerald-50"

  const icon = isReset ? "🔄" : isStatus ? "🔍" : isLabeled ? "🏷️" : "⚙️"

  const heading = isReset
    ? "Dashboard Reset"
    : isStatus
      ? "Dashboard Config"
      : isLabeled
        ? "Field Labeled"
        : "Dashboard Configured"

  return (
    <Card
      className={`mt-2 ${borderClass}`}
      aria-label="Prediction dashboard field configuration"
      data-testid="dashboard-config-card"
    >
      <CardHeader className="pb-2 pt-3 px-4">
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm font-semibold">
          <span aria-hidden="true">{icon}</span>
          <span>{heading}</span>
          <Badge variant="secondary" className="text-xs">
            {config.visible_count}/{config.total_count} visible
          </Badge>
          {config.locked_count > 0 && (
            <Badge variant="outline" className="border-amber-300 text-amber-700 text-xs">
              {config.locked_count} locked
            </Badge>
          )}
          {(config.labeled_count ?? 0) > 0 && (
            <Badge variant="outline" className="border-violet-300 text-violet-700 text-xs">
              {config.labeled_count} labeled
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="px-4 pb-3 space-y-1">
        <p className="text-xs text-muted-foreground">{config.summary}</p>

        {config.changes.length > 0 && (
          <div className="mt-2 divide-y divide-border rounded-md border bg-background p-1">
            {config.changes.map((field) => (
              <FieldRow key={field.feature_name} field={field} />
            ))}
          </div>
        )}

        <p className="mt-2 text-xs text-muted-foreground border-t pt-2">
          {isReset
            ? "All fields are now visible on the shared prediction URL."
            : isStatus
              ? "Say 'hide X from the dashboard' or 'show all fields' to adjust visibility."
              : isLabeled
                ? "The new label is shown on the shared prediction URL immediately."
                : "Changes are reflected immediately on the shared prediction URL."}
        </p>
      </CardContent>
    </Card>
  )
}
