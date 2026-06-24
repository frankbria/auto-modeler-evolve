"use client"

import type { MonitoringNote } from "@/lib/types"

interface MonitoringNoteCardProps {
  note: MonitoringNote
}

const KIND_ICON: Record<MonitoringNote["kind"], string> = {
  readiness: "✅",
  drift: "📈",
  health: "❤️",
  alerts: "🔔",
  error: "⛔",
}

const TONE_BORDER: Record<NonNullable<MonitoringNote["tone"]>, string> = {
  good: "border-emerald-300",
  neutral: "border-slate-200",
  warning: "border-amber-300",
  critical: "border-rose-300",
}

/**
 * Compact card shared by the `readiness`, `drift`, `health`, and `alerts` SSE
 * events (#17). Each handler maps its backend payload to a `MonitoringNote`.
 */
export function MonitoringNoteCard({ note }: MonitoringNoteCardProps) {
  const border = TONE_BORDER[note.tone ?? "neutral"]
  return (
    <figure
      className={`rounded-xl border-2 ${border} bg-card p-4 shadow-sm`}
      aria-label={`${note.title}: ${note.summary}`}
      data-testid="monitoring-note-card"
      data-kind={note.kind}
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true" className="text-2xl leading-none">
          {KIND_ICON[note.kind]}
        </span>
        <h3 className="text-sm font-bold text-foreground" data-testid="monitoring-note-title">
          {note.title}
        </h3>
      </div>

      <p className="text-sm text-muted-foreground" data-testid="monitoring-note-summary">
        {note.summary}
      </p>

      {note.items && note.items.length > 0 && (
        <ul className="mt-3 space-y-1" data-testid="monitoring-note-items">
          {note.items.map((item, i) => (
            <li
              key={i}
              className="rounded-md border border-slate-100 bg-slate-50/60 px-3 py-1.5 text-sm text-foreground"
            >
              {item}
            </li>
          ))}
        </ul>
      )}
    </figure>
  )
}
