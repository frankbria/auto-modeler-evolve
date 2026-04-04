"use client"

import type { ConversationExportInfo } from "@/lib/types"

interface ConversationExportCardProps {
  info: ConversationExportInfo
}

export function ConversationExportCard({ info }: ConversationExportCardProps) {
  const backendBase =
    typeof window !== "undefined"
      ? window.location.origin.replace(":3000", ":8000")
      : "http://localhost:8000"
  const downloadUrl = `${backendBase}${info.download_url}`

  return (
    <figure
      className="mt-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4 shadow-sm max-w-md"
      aria-label="Conversation export report"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">📄</span>
        <div>
          <p className="font-semibold text-emerald-900 text-sm leading-tight">
            Analysis Report Ready
          </p>
          <p className="text-xs text-emerald-600">
            Download the full conversation as a shareable HTML report
          </p>
        </div>
      </div>

      {/* Metadata */}
      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-medium">
          {info.message_count} AI response{info.message_count !== 1 ? "s" : ""}
        </span>
        {info.dataset_name && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-medium">
            {info.dataset_name}
          </span>
        )}
      </div>

      <p className="text-xs text-emerald-700 mb-3">
        The report includes the full conversation transcript, dataset info, and
        model results — ready to share with your team or VP.
      </p>

      {/* Download button */}
      <a
        href={downloadUrl}
        download
        className="block w-full text-center text-sm font-medium py-1.5 px-4 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition-colors"
        aria-label="Download analysis report as HTML file"
      >
        Download HTML Report
      </a>
    </figure>
  )
}
