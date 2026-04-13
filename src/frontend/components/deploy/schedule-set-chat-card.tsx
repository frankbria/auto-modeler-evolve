"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ScheduleSetResult, ScheduleSetScheduleItem } from "@/lib/types"

// ---------------------------------------------------------------------------
// ScheduleSetChatCard — inline chat confirmation for batch schedule creation
// or listing of existing schedules.
// ---------------------------------------------------------------------------

interface ScheduleSetChatCardProps {
  result: ScheduleSetResult
}

function FrequencyBadge({ frequency }: { frequency: "daily" | "weekly" | "monthly" }) {
  const colors: Record<string, string> = {
    daily: "bg-sky-100 text-sky-700 border-sky-200",
    weekly: "bg-violet-100 text-violet-700 border-violet-200",
    monthly: "bg-amber-100 text-amber-700 border-amber-200",
  }
  return (
    <Badge className={`border ${colors[frequency] ?? ""}`}>
      {frequency.charAt(0).toUpperCase() + frequency.slice(1)}
    </Badge>
  )
}

function formatNextRun(nextRun: string | null): string {
  if (!nextRun) return "—"
  const d = new Date(nextRun + "Z")
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  })
}

function ScheduleRow({ s }: { s: ScheduleSetScheduleItem }) {
  return (
    <div className="flex flex-wrap items-center gap-2 py-1 text-sm">
      <FrequencyBadge frequency={s.frequency} />
      <span className="text-foreground">{s.description}</span>
      {s.next_run && (
        <span className="text-muted-foreground">· Next: {formatNextRun(s.next_run)}</span>
      )}
      {s.last_row_count != null && (
        <span className="text-muted-foreground">· Last: {s.last_row_count.toLocaleString()} rows</span>
      )}
    </div>
  )
}

export function ScheduleSetChatCard({ result }: ScheduleSetChatCardProps) {
  const isCreated = result.action === "created"

  return (
    <Card
      className="border-teal-300 bg-teal-50/40"
      role="region"
      aria-label={isCreated ? "Batch schedule created" : "Batch schedules"}
    >
      <CardHeader className="pb-2 pt-3 px-4">
        <CardTitle className="flex flex-wrap items-center gap-2 text-sm font-semibold">
          <span aria-hidden="true">🗓️</span>
          {isCreated ? "Batch Schedule Created" : "Batch Schedules"}
          {isCreated && result.frequency && (
            <FrequencyBadge frequency={result.frequency} />
          )}
          {!isCreated && (
            <Badge variant="outline">{result.count ?? 0} schedule{(result.count ?? 0) !== 1 ? "s" : ""}</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-3 space-y-2">
        {isCreated && result.description && (
          <>
            <p className="text-sm font-medium text-foreground">{result.description}</p>
            {result.next_run && (
              <p className="text-sm text-muted-foreground">
                Next run: <span className="font-medium">{formatNextRun(result.next_run)}</span>
              </p>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              The model will score your current dataset automatically and save results as a downloadable CSV.
              Manage schedules in the Deployment panel.
            </p>
          </>
        )}

        {!isCreated && result.schedules && result.schedules.length > 0 && (
          <div className="divide-y divide-border">
            {result.schedules.map((s) => (
              <ScheduleRow key={s.id} s={s} />
            ))}
          </div>
        )}

        {!isCreated && (!result.schedules || result.schedules.length === 0) && (
          <p className="text-sm text-muted-foreground">
            No batch schedules configured yet. Ask me to schedule daily predictions at 9am to set one up.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
