"use client"

import type { ModelCardExportInfo } from "@/lib/types"

interface ModelCardExportCardProps {
  info: ModelCardExportInfo
}

export function ModelCardExportCard({ info }: ModelCardExportCardProps) {
  const backendBase =
    typeof window !== "undefined"
      ? window.location.origin.replace(":3000", ":8000")
      : "http://localhost:8000"
  const downloadUrl = `${backendBase}${info.download_url}`

  return (
    <figure
      className="mt-3 rounded-xl border border-indigo-200 bg-indigo-50 p-4 shadow-sm max-w-md"
      aria-label="Model card export"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">📋</span>
        <div>
          <p className="font-semibold text-indigo-900 text-sm leading-tight">
            Model Card Ready
          </p>
          <p className="text-xs text-indigo-600">
            Standardized model documentation for compliance &amp; governance
          </p>
        </div>
      </div>

      {/* Badges */}
      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-800 font-medium">
          {info.algorithm_plain}
        </span>
        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-purple-100 text-purple-800 font-medium">
          {info.problem_type}
        </span>
        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-medium">
          target: {info.target_column}
        </span>
      </div>

      {/* Metrics row */}
      <div className="mb-3 flex flex-wrap gap-3 text-xs text-indigo-800">
        <span>
          <span className="font-semibold">{info.metric_name.toUpperCase()}:</span>{" "}
          {info.metric_display}
        </span>
        <span>
          <span className="font-semibold">Features:</span> {info.feature_count}
        </span>
        <span>
          <span className="font-semibold">Rows:</span> {info.row_count.toLocaleString()}
        </span>
      </div>

      {info.trained_at && (
        <p className="text-xs text-indigo-600 mb-3">
          Trained {new Date(info.trained_at).toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric",
          })}
        </p>
      )}

      {/* Download button */}
      <a
        href={downloadUrl}
        download
        className="block w-full text-center text-sm font-medium py-1.5 px-4 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
        aria-label="Download model card as HTML file"
      >
        Download HTML Model Card
      </a>
    </figure>
  )
}
