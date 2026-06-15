"use client"

import type { ModelStatusReportInfo } from "@/lib/types"

interface ModelStatusReportCardProps {
  info: ModelStatusReportInfo
}

export function ModelStatusReportCard({ info }: ModelStatusReportCardProps) {
  const backendBase =
    typeof window !== "undefined"
      ? window.location.origin.replace(":3000", ":8000")
      : "http://localhost:8000"
  const downloadUrl = `${backendBase}${info.download_url}`

  const healthColor =
    info.health_score >= 75
      ? "text-emerald-700"
      : info.health_score >= 50
        ? "text-amber-700"
        : "text-rose-700"

  const healthBg =
    info.health_score >= 75
      ? "bg-emerald-50 border-emerald-200"
      : info.health_score >= 50
        ? "bg-amber-50 border-amber-200"
        : "bg-rose-50 border-rose-200"

  const slaLabel = info.sla_ok
    ? { text: "Within SLA", cls: "bg-emerald-100 text-emerald-800" }
    : { text: "SLA Alert", cls: "bg-rose-100 text-rose-800" }

  return (
    <figure
      className={`mt-3 rounded-xl border p-4 shadow-sm max-w-md ${healthBg}`}
      aria-label="Model status report"
      data-testid="model-status-report-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">📊</span>
        <div>
          <p className="font-semibold text-sm leading-tight text-gray-900">
            Model Status Report
          </p>
          <p className="text-xs text-gray-500">
            Production monitoring summary for stakeholders
          </p>
        </div>
      </div>

      {/* Badges */}
      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-medium">
          {info.algorithm_plain}
        </span>
        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-medium">
          {info.problem_type}
        </span>
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded-full font-medium ${slaLabel.cls}`}
          data-testid="sla-badge"
        >
          {slaLabel.text}
        </span>
      </div>

      {/* Health score */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-medium text-gray-600">Model Health</span>
          <span className={`text-sm font-bold ${healthColor}`} data-testid="health-score">
            {info.health_score}/100 — {info.health_status}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div
            className={`h-2 rounded-full transition-all ${info.health_score >= 75 ? "bg-emerald-500" : info.health_score >= 50 ? "bg-amber-500" : "bg-rose-500"}`}
            style={{ width: `${Math.max(0, Math.min(100, info.health_score))}%` }}
            role="progressbar"
            aria-valuenow={info.health_score}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Health score: ${info.health_score} out of 100`}
          />
        </div>
      </div>

      {/* Volume grid */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="bg-white rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-blue-700" data-testid="predictions-today">
            {info.predictions_today.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">Today</div>
        </div>
        <div className="bg-white rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-blue-700" data-testid="predictions-7d">
            {info.predictions_7d.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">Last 7 days</div>
        </div>
        <div className="bg-white rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-gray-700" data-testid="predictions-30d">
            {info.predictions_30d.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">Last 30 days</div>
        </div>
        <div className="bg-white rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-gray-700" data-testid="predictions-total">
            {info.predictions_total.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">All time</div>
        </div>
      </div>

      {/* SLA latency */}
      {info.p95_ms != null && (
        <p className="text-xs text-gray-600 mb-3" data-testid="sla-latency">
          <span className="font-medium">p95 latency:</span>{" "}
          <span className={info.sla_ok ? "text-emerald-700" : "text-rose-700"}>
            {Math.round(info.p95_ms)} ms
          </span>
          {" "}(target: &lt;500 ms)
        </p>
      )}

      {/* Summary */}
      <p className="text-xs text-gray-600 mb-3 italic" data-testid="report-summary">
        {info.summary}
      </p>

      {/* Download */}
      <a
        href={downloadUrl}
        download
        className="block w-full text-center text-sm font-medium py-1.5 px-4 rounded-lg bg-gray-800 text-white hover:bg-gray-700 transition-colors"
        aria-label="Download full monitoring status report as HTML"
        data-testid="download-report-link"
      >
        Download Full Report
      </a>

      <figcaption className="sr-only">
        Model status report for {info.project_name}: health score {info.health_score} out of 100
        ({info.health_status}), {info.predictions_7d} predictions in the last 7 days,
        SLA {info.sla_ok ? "within" : "exceeding"} target.
        Click to download the full HTML report for stakeholders.
      </figcaption>
    </figure>
  )
}
