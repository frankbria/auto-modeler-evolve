"use client"

import { useState } from "react"
import type { AutoRetrainResult } from "@/lib/types"

interface AutoRetrainCardProps {
  result: AutoRetrainResult
  onToggle?: (enabled: boolean) => void
}

function AlgoLabel({ algorithm }: { algorithm: string }) {
  return (
    <span className="font-mono text-xs bg-teal-100 text-teal-800 px-2 py-0.5 rounded">
      {algorithm.replace(/_/g, " ")}
    </span>
  )
}

export function AutoRetrainCard({ result, onToggle }: AutoRetrainCardProps) {
  const [enabled, setEnabled] = useState(result.enabled)
  const [loading, setLoading] = useState(false)

  async function handleToggle() {
    setLoading(true)
    try {
      const res = await fetch(
        `/api/projects/${result.project_id}/auto-retrain`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !enabled }),
        }
      )
      if (res.ok) {
        setEnabled(!enabled)
        onToggle?.(!enabled)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mt-3 rounded-xl border border-teal-200 bg-teal-50 p-4 shadow-sm max-w-md">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">🔄</span>
        <div>
          <p className="font-semibold text-teal-900 text-sm leading-tight">
            Auto-Retrain
          </p>
          <p className="text-xs text-teal-600">
            Automatically retrain on new data uploads
          </p>
        </div>
        <div className="ml-auto">
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
              enabled
                ? "bg-teal-200 text-teal-800"
                : "bg-slate-200 text-slate-600"
            }`}
          >
            {enabled ? "Enabled" : "Disabled"}
          </span>
        </div>
      </div>

      {/* Algorithm info */}
      {enabled && result.has_selected_model && result.selected_algorithm && (
        <div className="mb-3 flex items-center gap-2 text-xs text-teal-700">
          <span>Will retrain using</span>
          <AlgoLabel algorithm={result.selected_algorithm} />
        </div>
      )}

      {enabled && !result.has_selected_model && (
        <p className="mb-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
          No selected model found. Train and select a model first for
          auto-retrain to work.
        </p>
      )}

      {!enabled && (
        <p className="mb-3 text-xs text-slate-600">
          When enabled, uploading new data will automatically kick off a
          background retrain using your selected model&apos;s algorithm.
        </p>
      )}

      {/* Toggle button */}
      <button
        onClick={handleToggle}
        disabled={loading}
        className={`w-full text-sm font-medium py-1.5 px-4 rounded-lg transition-colors disabled:opacity-50 ${
          enabled
            ? "bg-slate-200 text-slate-700 hover:bg-slate-300"
            : "bg-teal-600 text-white hover:bg-teal-700"
        }`}
      >
        {loading
          ? "Updating…"
          : enabled
            ? "Disable Auto-Retrain"
            : "Enable Auto-Retrain"}
      </button>
    </div>
  )
}
