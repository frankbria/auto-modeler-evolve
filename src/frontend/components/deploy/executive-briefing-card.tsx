"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ExecutiveBriefingResult, BriefingSection } from "@/lib/types"

interface ExecutiveBriefingCardProps {
  briefing: ExecutiveBriefingResult
}

export function ExecutiveBriefingCard({ briefing }: ExecutiveBriefingCardProps) {
  const [copied, setCopied] = useState(false)

  const buildPlainText = () => {
    const lines: string[] = []
    lines.push(`Executive Briefing — ${briefing.project_name}`)
    lines.push("=".repeat(50))
    lines.push("")
    lines.push(briefing.summary)
    lines.push("")
    for (const section of briefing.sections) {
      lines.push(section.heading.toUpperCase())
      lines.push(section.body.replace(/\*\*/g, ""))
      lines.push("")
    }
    if (briefing.action_items.length > 0) {
      lines.push("RECOMMENDED ACTIONS")
      for (const item of briefing.action_items) {
        lines.push(`• ${item}`)
      }
      lines.push("")
    }
    if (briefing.prediction_url) {
      lines.push(`Prediction Dashboard: ${briefing.prediction_url}`)
    }
    return lines.join("\n")
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(buildPlainText()).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const metricBadgeColor =
    briefing.metric_value !== null && briefing.metric_value !== undefined
      ? briefing.problem_type === "regression"
        ? briefing.metric_value >= 0.85
          ? "bg-emerald-100 text-emerald-800 border-emerald-200"
          : briefing.metric_value >= 0.70
          ? "bg-blue-100 text-blue-800 border-blue-200"
          : "bg-amber-100 text-amber-800 border-amber-200"
        : briefing.metric_value >= 0.90
        ? "bg-emerald-100 text-emerald-800 border-emerald-200"
        : briefing.metric_value >= 0.80
        ? "bg-blue-100 text-blue-800 border-blue-200"
        : "bg-amber-100 text-amber-800 border-amber-200"
      : "bg-muted text-muted-foreground"

  return (
    <div
      role="region"
      aria-label="Executive briefing"
      className="mt-2 rounded-lg border-2 border-emerald-200 bg-card p-4 text-sm"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-lg">📋</span>
          <h3 className="font-semibold text-foreground">Executive Briefing</h3>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {briefing.algorithm && (
            <Badge variant="outline" className="text-xs">
              {briefing.algorithm}
            </Badge>
          )}
          {briefing.metric_label && (
            <span
              className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${metricBadgeColor}`}
            >
              {briefing.metric_label}
            </span>
          )}
        </div>
      </div>

      {/* One-line summary */}
      <p className="mt-2 text-muted-foreground italic text-xs">{briefing.summary}</p>

      {/* Sections */}
      <div className="mt-3 space-y-3">
        {briefing.sections.map((section: BriefingSection, i: number) => (
          <div key={i}>
            <h4 className="font-medium text-foreground text-xs uppercase tracking-wide text-muted-foreground mb-0.5">
              {section.heading}
            </h4>
            <p className="text-foreground leading-relaxed whitespace-pre-line">
              {section.body.replace(/\*\*/g, "")}
            </p>
          </div>
        ))}
      </div>

      {/* Recommended actions */}
      {briefing.action_items.length > 0 && (
        <div className="mt-3">
          <h4 className="font-medium text-xs uppercase tracking-wide text-muted-foreground mb-1">
            Recommended Actions
          </h4>
          <ul className="space-y-1">
            {briefing.action_items.map((item, i) => (
              <li key={i} className="flex gap-2 text-foreground">
                <span className="mt-0.5 text-emerald-600 shrink-0">→</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Footer */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t pt-2">
        {briefing.prediction_url ? (
          <a
            href={briefing.prediction_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-emerald-700 underline underline-offset-2 hover:text-emerald-900"
          >
            Open prediction dashboard →
          </a>
        ) : (
          <span className="text-xs text-muted-foreground">
            Deploy the model to get a shareable prediction dashboard
          </span>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopy}
          className="text-xs"
          aria-label="Copy briefing to clipboard"
        >
          {copied ? "Copied!" : "Copy to clipboard"}
        </Button>
      </div>
    </div>
  )
}
