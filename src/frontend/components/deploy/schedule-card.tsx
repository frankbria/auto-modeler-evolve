"use client"

import { useState, useEffect } from "react"
import { api } from "@/lib/api"
import { BatchSchedule, BatchJobRun } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

interface ScheduleCardProps {
  deploymentId: string
}

const DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

function formatTime(hour: number, minute: number): string {
  const h = hour.toString().padStart(2, "0")
  const m = minute.toString().padStart(2, "0")
  return `${h}:${m} UTC`
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

function formatLastRun(lastRun: string | null): string {
  if (!lastRun) return "Never"
  const d = new Date(lastRun + "Z")
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

function scheduleDescription(s: BatchSchedule): string {
  const time = formatTime(s.run_hour, s.run_minute)
  if (s.frequency === "daily") return `Every day at ${time}`
  if (s.frequency === "weekly") {
    const dayName = s.day_of_week !== null ? DAY_NAMES[s.day_of_week] ?? "Monday" : "Monday"
    return `Every ${dayName} at ${time}`
  }
  const dom = s.day_of_month ?? 1
  return `Monthly on the ${dom}${["st","nd","rd"][dom - 1] ?? "th"} at ${time}`
}

export function ScheduleCard({ deploymentId }: ScheduleCardProps) {
  const [schedules, setSchedules] = useState<BatchSchedule[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [triggering, setTriggering] = useState<string | null>(null)
  const [expandedRuns, setExpandedRuns] = useState<string | null>(null)
  const [runs, setRuns] = useState<BatchJobRun[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Form state
  const [frequency, setFrequency] = useState<"daily" | "weekly" | "monthly">("daily")
  const [runHour, setRunHour] = useState(9)
  const [runMinute, setRunMinute] = useState(0)
  const [dayOfWeek, setDayOfWeek] = useState(0)
  const [dayOfMonth, setDayOfMonth] = useState(1)
  const [showForm, setShowForm] = useState(false)

  useEffect(() => {
    api.deploy.getSchedules(deploymentId)
      .then(setSchedules)
      .catch(() => setError("Failed to load schedules"))
      .finally(() => setLoading(false))
  }, [deploymentId])

  async function handleCreate() {
    setCreating(true)
    setError(null)
    try {
      const schedule = await api.deploy.createSchedule(deploymentId, {
        frequency,
        run_hour: runHour,
        run_minute: runMinute,
        day_of_week: frequency === "weekly" ? dayOfWeek : null,
        day_of_month: frequency === "monthly" ? dayOfMonth : null,
      })
      setSchedules((prev) => [...prev, schedule])
      setShowForm(false)
    } catch {
      setError("Failed to create schedule")
    } finally {
      setCreating(false)
    }
  }

  async function handleDelete(scheduleId: string) {
    try {
      await api.deploy.deleteSchedule(deploymentId, scheduleId)
      setSchedules((prev) => prev.filter((s) => s.id !== scheduleId))
      if (expandedRuns === scheduleId) setExpandedRuns(null)
    } catch {
      setError("Failed to delete schedule")
    }
  }

  async function handleTrigger(scheduleId: string) {
    setTriggering(scheduleId)
    setError(null)
    try {
      await api.deploy.triggerSchedule(deploymentId, scheduleId)
      // Refresh schedule to show updated last_run after a brief delay
      setTimeout(async () => {
        const updated = await api.deploy.getSchedules(deploymentId)
        setSchedules(updated)
        setTriggering(null)
      }, 2000)
    } catch {
      setError("Failed to trigger run")
      setTriggering(null)
    }
  }

  async function handleShowRuns(scheduleId: string) {
    if (expandedRuns === scheduleId) {
      setExpandedRuns(null)
      return
    }
    setExpandedRuns(scheduleId)
    setRunsLoading(true)
    try {
      const data = await api.deploy.getScheduleRuns(deploymentId, scheduleId)
      setRuns(data)
    } catch {
      setError("Failed to load run history")
    } finally {
      setRunsLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
        Loading schedules…
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-foreground">Scheduled Batch Predictions</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Run batch predictions automatically on a recurring schedule. Results are saved as
            downloadable CSVs.
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setShowForm((v) => !v)}
          data-testid="add-schedule-btn"
        >
          {showForm ? "Cancel" : "+ Add Schedule"}
        </Button>
      </div>

      {error && (
        <div className="rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {/* Create form */}
      {showForm && (
        <div
          className="rounded-lg border border-border bg-muted/30 p-4 space-y-3"
          data-testid="schedule-form"
        >
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Frequency</label>
              <select
                className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5 text-sm"
                value={frequency}
                onChange={(e) => setFrequency(e.target.value as "daily" | "weekly" | "monthly")}
                data-testid="frequency-select"
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Time (UTC)</label>
              <div className="mt-1 flex gap-1">
                <select
                  className="flex-1 rounded border border-border bg-background px-2 py-1.5 text-sm"
                  value={runHour}
                  onChange={(e) => setRunHour(Number(e.target.value))}
                  data-testid="hour-select"
                >
                  {Array.from({ length: 24 }, (_, i) => (
                    <option key={i} value={i}>
                      {i.toString().padStart(2, "0")}
                    </option>
                  ))}
                </select>
                <span className="self-center text-muted-foreground">:</span>
                <select
                  className="flex-1 rounded border border-border bg-background px-2 py-1.5 text-sm"
                  value={runMinute}
                  onChange={(e) => setRunMinute(Number(e.target.value))}
                  data-testid="minute-select"
                >
                  {[0, 15, 30, 45].map((m) => (
                    <option key={m} value={m}>
                      {m.toString().padStart(2, "0")}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {frequency === "weekly" && (
            <div>
              <label className="text-xs font-medium text-muted-foreground">Day of Week</label>
              <select
                className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5 text-sm"
                value={dayOfWeek}
                onChange={(e) => setDayOfWeek(Number(e.target.value))}
                data-testid="day-of-week-select"
              >
                {DAY_NAMES.map((d, i) => (
                  <option key={i} value={i}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          )}

          {frequency === "monthly" && (
            <div>
              <label className="text-xs font-medium text-muted-foreground">
                Day of Month (1–28)
              </label>
              <input
                type="number"
                min={1}
                max={28}
                className="mt-1 w-full rounded border border-border bg-background px-2 py-1.5 text-sm"
                value={dayOfMonth}
                onChange={(e) => setDayOfMonth(Number(e.target.value))}
                data-testid="day-of-month-input"
              />
            </div>
          )}

          <Button
            size="sm"
            onClick={handleCreate}
            disabled={creating}
            data-testid="create-schedule-btn"
          >
            {creating ? "Creating…" : "Create Schedule"}
          </Button>
        </div>
      )}

      {/* Schedule list */}
      {schedules.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
          No schedules yet. Add one above to automate batch predictions.
        </div>
      ) : (
        <div className="space-y-3">
          {schedules.map((schedule) => (
            <div
              key={schedule.id}
              className="rounded-lg border border-border bg-card p-4"
              data-testid={`schedule-item-${schedule.id}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm text-foreground">
                      {scheduleDescription(schedule)}
                    </span>
                    <Badge
                      variant="outline"
                      className={
                        schedule.is_active
                          ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700"
                          : "border-muted text-muted-foreground"
                      }
                    >
                      {schedule.is_active ? "Active" : "Inactive"}
                    </Badge>
                  </div>
                  <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
                    <span>
                      Next run:{" "}
                      <span className="text-foreground">{formatNextRun(schedule.next_run)}</span>
                    </span>
                    <span>
                      Last run:{" "}
                      <span className="text-foreground">{formatLastRun(schedule.last_run)}</span>
                    </span>
                    {schedule.last_row_count !== null && (
                      <span>
                        Last output:{" "}
                        <span className="text-foreground">
                          {schedule.last_row_count.toLocaleString()} rows
                        </span>
                      </span>
                    )}
                    {schedule.last_error && (
                      <span className="col-span-2 text-destructive">
                        Error: {schedule.last_error}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={triggering === schedule.id}
                    onClick={() => handleTrigger(schedule.id)}
                    data-testid={`run-now-btn-${schedule.id}`}
                  >
                    {triggering === schedule.id ? "Running…" : "Run Now"}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => handleShowRuns(schedule.id)}
                    data-testid={`history-btn-${schedule.id}`}
                  >
                    {expandedRuns === schedule.id ? "Hide History" : "History"}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-destructive hover:text-destructive"
                    onClick={() => handleDelete(schedule.id)}
                    data-testid={`delete-schedule-btn-${schedule.id}`}
                  >
                    Remove
                  </Button>
                </div>
              </div>

              {/* Run history */}
              {expandedRuns === schedule.id && (
                <div className="mt-3 border-t border-border pt-3">
                  <h4 className="text-xs font-medium text-muted-foreground mb-2">Recent Runs</h4>
                  {runsLoading ? (
                    <p className="text-xs text-muted-foreground">Loading…</p>
                  ) : runs.length === 0 ? (
                    <p className="text-xs text-muted-foreground">No runs yet.</p>
                  ) : (
                    <div className="space-y-1">
                      {runs.map((run) => (
                        <div
                          key={run.id}
                          className="flex items-center justify-between gap-2 text-xs"
                          data-testid={`job-run-${run.id}`}
                        >
                          <span className="text-muted-foreground">
                            {new Date(run.started_at + "Z").toLocaleString(undefined, {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </span>
                          <Badge
                            variant="outline"
                            className={
                              run.status === "success"
                                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700"
                                : run.status === "failed"
                                ? "border-destructive/40 bg-destructive/10 text-destructive"
                                : "border-amber-500/40 bg-amber-500/10 text-amber-700"
                            }
                          >
                            {run.status}
                          </Badge>
                          {run.row_count !== null && (
                            <span className="text-muted-foreground">
                              {run.row_count.toLocaleString()} rows
                            </span>
                          )}
                          {run.download_url ? (
                            <a
                              href={`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}${run.download_url}`}
                              download
                              className="text-primary underline"
                              data-testid={`download-run-${run.id}`}
                            >
                              Download
                            </a>
                          ) : run.error ? (
                            <span className="text-destructive truncate max-w-[120px]" title={run.error}>
                              {run.error}
                            </span>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        Batch jobs run against your training dataset and produce a CSV with predictions added. All
        times are UTC. Jobs run within ~1 minute of the scheduled time.
      </p>
    </div>
  )
}
