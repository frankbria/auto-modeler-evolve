"use client"

/**
 * ReadinessCheckCard — displays a data readiness assessment produced by
 * core/readiness.py.
 *
 * Layout:
 *   • Score gauge: large number + letter grade + status badge
 *   • 5-component breakdown: name, progress bar, status icon, detail line
 *   • Top recommendations list (capped at 5, with action hints)
 *
 * Appears inline in the chat when the user asks "is my data ready?" and also
 * as a standalone card in the Data tab via the "Check Readiness" button.
 */

import { useState } from "react"
import { api } from "@/lib/api"
import type { DataReadinessResult, ReadinessComponent } from "@/lib/types"

interface Props {
  /** Pre-computed result (from SSE event). If absent, shows a "Check" button. */
  result?: DataReadinessResult
  /** Dataset ID — needed for the manual fetch path. */
  datasetId?: string
}

function statusColor(status: string): string {
  switch (status) {
    case "good":
      return "text-green-600"
    case "warning":
      return "text-yellow-600"
    case "critical":
      return "text-red-600"
    default:
      return "text-gray-500"
  }
}

function statusIcon(status: string): string {
  switch (status) {
    case "good":
      return "✓"
    case "warning":
      return "⚠"
    case "critical":
      return "✗"
    default:
      return "·"
  }
}

function gradeColor(grade: string): string {
  switch (grade) {
    case "A":
      return "text-green-600"
    case "B":
      return "text-blue-600"
    case "C":
      return "text-yellow-600"
    case "D":
      return "text-orange-600"
    default:
      return "text-red-600"
  }
}

function statusBadge(status: string): { label: string; className: string } {
  switch (status) {
    case "ready":
      return {
        label: "Ready to Train",
        className:
          "inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700",
      }
    case "needs_attention":
      return {
        label: "Needs Attention",
        className:
          "inline-flex items-center rounded-full bg-yellow-100 px-2 py-0.5 text-xs font-medium text-yellow-700",
      }
    default:
      return {
        label: "Not Ready",
        className:
          "inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700",
      }
  }
}

function ComponentRow({ component }: { component: ReadinessComponent }) {
  const pct = component.max_score > 0
    ? Math.round((component.score / component.max_score) * 100)
    : 0
  const isAdvisory = component.advisory === true

  return (
    <div className="flex items-start gap-2 py-1.5" data-testid="readiness-component">
      <span
        className={`mt-0.5 w-4 shrink-0 text-center text-xs font-bold ${statusColor(component.status)}`}
        data-testid={`component-icon-${component.status}`}
      >
        {statusIcon(component.status)}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium text-gray-800">
            {component.name}
            {isAdvisory && (
              <span className="ml-1 text-xs font-normal text-gray-400">(advisory)</span>
            )}
          </span>
          {!isAdvisory && (
            <span className="shrink-0 text-xs text-gray-500">
              {component.score}/{component.max_score}
            </span>
          )}
        </div>
        {!isAdvisory && (
          <div
            className="mt-0.5 h-1.5 w-full overflow-hidden rounded-full bg-gray-100"
            data-testid="component-progress-bar"
          >
            <div
              className={`h-full rounded-full ${
                component.status === "good"
                  ? "bg-green-400"
                  : component.status === "warning"
                    ? "bg-yellow-400"
                    : "bg-red-400"
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
        )}
        <p className="mt-0.5 text-xs text-gray-500 leading-relaxed">{component.detail}</p>
      </div>
    </div>
  )
}

export function ReadinessCheckCard({ result: initialResult, datasetId }: Props) {
  const [result, setResult] = useState<DataReadinessResult | null>(
    initialResult ?? null
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleCheck() {
    if (!datasetId) return
    setLoading(true)
    setError(null)
    try {
      const data = await api.data.getReadinessCheck(datasetId)
      setResult(data)
    } catch {
      setError("Could not compute readiness check.")
    } finally {
      setLoading(false)
    }
  }

  // If no result yet, show a "Check" button
  if (!result) {
    return (
      <div
        className="rounded-lg border bg-white p-3 shadow-sm"
        data-testid="readiness-check-card"
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-gray-900">Data Readiness Check</p>
            <p className="text-xs text-gray-500">
              Assess your dataset before training
            </p>
          </div>
          <button
            onClick={handleCheck}
            disabled={loading || !datasetId}
            className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            data-testid="check-readiness-btn"
          >
            {loading ? "Checking…" : "Check Readiness"}
          </button>
        </div>
        {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
      </div>
    )
  }

  const badge = statusBadge(result.status)
  // Core components (non-advisory)
  const coreComponents = result.components.filter((c) => !c.advisory)
  const advisoryComponents = result.components.filter((c) => c.advisory)

  return (
    <div
      className="rounded-lg border bg-white p-4 shadow-sm"
      data-testid="readiness-check-card"
    >
      {/* Header — score gauge */}
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-900">Data Readiness</p>
          <p className="mt-0.5 text-xs text-gray-500 leading-relaxed">
            {result.summary}
          </p>
          <div className="mt-1.5">
            <span className={badge.className} data-testid="readiness-status-badge">
              {badge.label}
            </span>
          </div>
        </div>
        <div className="shrink-0 text-center">
          <div
            className={`text-3xl font-bold tabular-nums ${gradeColor(result.grade)}`}
            data-testid="readiness-grade"
          >
            {result.grade}
          </div>
          <div className="text-sm font-semibold text-gray-600" data-testid="readiness-score">
            {result.score}/100
          </div>
        </div>
      </div>

      {/* Component breakdown */}
      <div className="border-t pt-2">
        <p className="mb-1 text-xs font-medium text-gray-400 uppercase tracking-wide">
          Components
        </p>
        <div className="divide-y divide-gray-50">
          {coreComponents.map((comp) => (
            <ComponentRow key={comp.name} component={comp} />
          ))}
        </div>
        {advisoryComponents.length > 0 && (
          <div className="mt-1 border-t pt-1">
            {advisoryComponents.map((comp) => (
              <ComponentRow key={comp.name} component={comp} />
            ))}
          </div>
        )}
      </div>

      {/* Recommendations */}
      {result.recommendations.length > 0 && (
        <div className="mt-3 border-t pt-2">
          <p className="mb-1.5 text-xs font-medium text-gray-400 uppercase tracking-wide">
            Recommendations
          </p>
          <ul className="space-y-1" data-testid="readiness-recommendations">
            {result.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs text-gray-600">
                <span className="mt-0.5 shrink-0 text-primary">→</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
