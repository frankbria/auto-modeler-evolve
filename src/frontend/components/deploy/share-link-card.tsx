"use client"

import { useState } from "react"
import type { ShareLinkResult } from "@/lib/types"

interface ShareLinkCardProps {
  result: ShareLinkResult
}

export function ShareLinkCard({ result }: ShareLinkCardProps) {
  const [copied, setCopied] = useState(false)

  const { prefilled_url, dashboard_url, title, feature_values, feature_count, summary } = result
  const origin = typeof window !== "undefined" ? window.location.origin : "https://your-app.example.com"
  const fullUrl = `${origin}${prefilled_url}`

  const handleCopy = () => {
    navigator.clipboard.writeText(fullUrl).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const featureEntries = Object.entries(feature_values)

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-orange-200 bg-orange-50/50 p-3 text-sm"
      aria-label="Share link card"
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🔗</span>
        <span className="font-semibold text-slate-800" data-testid="share-link-heading">
          Pre-filled Scenario Link
        </span>
        {feature_count > 0 ? (
          <span className="ml-auto rounded bg-orange-100 px-2 py-0.5 text-xs font-medium text-orange-700" data-testid="feature-count-badge">
            {feature_count} value{feature_count !== 1 ? "s" : ""} pre-loaded
          </span>
        ) : (
          <span className="ml-auto rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600" data-testid="feature-count-badge">
            Opens at defaults
          </span>
        )}
      </div>

      {/* Dashboard title */}
      <p className="mb-2 text-xs text-slate-600">
        Dashboard:{" "}
        <span className="font-medium text-slate-800" data-testid="share-link-title">
          {title}
        </span>
      </p>

      {/* Pre-filled values */}
      {featureEntries.length > 0 && (
        <div className="mb-2 rounded border border-orange-100 bg-white p-2" data-testid="feature-values-list">
          <p className="mb-1 text-xs font-medium text-slate-600">Pre-filled values:</p>
          <div className="flex flex-wrap gap-1">
            {featureEntries.map(([key, value]) => (
              <span
                key={key}
                className="inline-flex items-center gap-0.5 rounded bg-orange-100 px-1.5 py-0.5 text-xs font-mono text-orange-800"
                data-testid={`feature-chip-${key}`}
              >
                <span className="font-semibold">{key}</span>
                <span className="text-orange-500">=</span>
                <span>{value}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* URL display */}
      <div className="relative mb-2 rounded border border-slate-200 bg-white">
        <div
          className="overflow-x-auto p-3 pr-16 text-xs leading-relaxed text-slate-700 font-mono break-all"
          data-testid="share-link-url"
        >
          {fullUrl}
        </div>
        <button
          onClick={handleCopy}
          className="absolute right-2 top-2 rounded bg-orange-100 px-2 py-1 text-xs font-medium text-orange-700 transition-colors hover:bg-orange-200"
          aria-label="Copy share link to clipboard"
          data-testid="copy-share-link-button"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>

      {/* Summary */}
      {summary && (
        <p className="mb-1.5 text-xs text-slate-600" data-testid="share-link-summary">
          {summary}
        </p>
      )}

      {/* Instructions */}
      <div className="mt-1.5 rounded bg-orange-100/60 px-2.5 py-2 text-xs text-orange-800" data-testid="share-link-instructions">
        <span className="font-medium">How to use:</span> Share this URL with your VP or team. When
        they open it, the prediction form will be pre-filled with the values above — ready to run.
      </div>

      {/* Footer */}
      <p className="mt-1.5 text-xs italic text-slate-400">
        Tip: Bookmark this link to quickly return to this exact scenario.{" "}
        <a
          href={`${origin}${dashboard_url}`}
          target="_blank"
          rel="noreferrer"
          className="text-orange-500 underline hover:text-orange-700"
          data-testid="open-dashboard-link"
        >
          Open dashboard →
        </a>
      </p>
    </div>
  )
}
