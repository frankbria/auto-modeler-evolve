"use client"

import { useState } from "react"
import type { EmbedCodeResult } from "@/lib/types"

interface EmbedCodeCardProps {
  result: EmbedCodeResult
}

type SizePreset = "full" | "fixed" | "compact"

const SIZE_PRESETS: Record<SizePreset, { label: string; width: string; height: string }> = {
  full: { label: "Full Width", width: "100%", height: "700" },
  fixed: { label: "Fixed (960×700)", width: "960", height: "700" },
  compact: { label: "Compact (600×500)", width: "600", height: "500" },
}

function buildIframeHtml(src: string, width: string, height: string, title: string): string {
  const h = `${height}px`
  const w = width.endsWith("%") ? width : `${width}px`
  return (
    `<iframe\n` +
    `  src="${src}"\n` +
    `  title="${title}"\n` +
    `  width="${w}"\n` +
    `  height="${h}"\n` +
    `  frameborder="0"\n` +
    `  allow="clipboard-write"\n` +
    `  style="border:none; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.1);"\n` +
    `></iframe>`
  )
}

export function EmbedCodeCard({ result }: EmbedCodeCardProps) {
  const [preset, setPreset] = useState<SizePreset>("full")
  const [copied, setCopied] = useState(false)

  const { dashboard_url, title, summary } = result
  const { width, height } = SIZE_PRESETS[preset]

  const origin = typeof window !== "undefined" ? window.location.origin : "https://your-app.example.com"
  const src = `${origin}${dashboard_url}`
  const iframeHtml = buildIframeHtml(src, width, height, title)

  const handleCopy = () => {
    navigator.clipboard.writeText(iframeHtml).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-indigo-200 bg-indigo-50/50 p-3 text-sm"
      aria-label="Embed code card"
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🖼️</span>
        <span className="font-semibold text-slate-800" data-testid="embed-code-heading">
          Embed Prediction Dashboard
        </span>
        <span className="ml-auto rounded bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
          Ready to embed
        </span>
      </div>

      {/* Dashboard title */}
      <p className="mb-2 text-xs text-slate-600">
        Dashboard:{" "}
        <span className="font-medium text-slate-800" data-testid="embed-dashboard-title">
          {title}
        </span>
      </p>

      {/* Size presets */}
      <div className="mb-2 flex items-center gap-1.5" data-testid="size-presets">
        <span className="text-xs text-slate-500">Size:</span>
        {(Object.keys(SIZE_PRESETS) as SizePreset[]).map((key) => (
          <button
            key={key}
            onClick={() => setPreset(key)}
            className={`rounded px-2 py-0.5 text-xs transition-colors ${
              preset === key
                ? "bg-indigo-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
            data-testid={`preset-${key}`}
            aria-pressed={preset === key}
          >
            {SIZE_PRESETS[key].label}
          </button>
        ))}
      </div>

      {/* Code block */}
      <div className="relative mb-2 rounded border border-slate-200 bg-white">
        <pre
          className="overflow-x-auto p-3 text-xs leading-relaxed text-slate-700"
          data-testid="embed-code-block"
        >
          <code>{iframeHtml}</code>
        </pre>
        <button
          onClick={handleCopy}
          className="absolute right-2 top-2 rounded bg-indigo-100 px-2 py-1 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-200"
          aria-label="Copy embed code to clipboard"
          data-testid="copy-button"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>

      {/* Summary */}
      {summary && (
        <p className="mb-1.5 text-xs text-slate-600" data-testid="embed-summary">
          {summary}
        </p>
      )}

      {/* Where to use */}
      <div className="mt-1.5 rounded bg-indigo-100/60 px-2.5 py-2 text-xs text-indigo-800">
        <span className="font-medium">Where to paste this:</span> SharePoint (Embed web part) ·
        Notion (Embed block) · Confluence (HTML macro) · any internal HTML page.
      </div>

      {/* Footer */}
      <p className="mt-1.5 text-xs italic text-slate-400">
        The prediction form works fully inside the iframe — VPs can submit predictions without
        leaving the portal.
      </p>
    </div>
  )
}
