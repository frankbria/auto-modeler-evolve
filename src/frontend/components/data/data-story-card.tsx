"use client"

import type { DataStory, DataStorySection } from "@/lib/types"

interface DataStoryCardProps {
  result: DataStory
}

function gradeBadgeClass(grade: string): string {
  if (grade === "A") return "bg-green-100 text-green-800"
  if (grade === "B") return "bg-blue-100 text-blue-800"
  if (grade === "C") return "bg-yellow-100 text-yellow-800"
  if (grade === "D") return "bg-orange-100 text-orange-800"
  return "bg-red-100 text-red-800"
}

function scoreBarClass(score: number): string {
  if (score >= 75) return "bg-green-500"
  if (score >= 50) return "bg-yellow-500"
  return "bg-red-500"
}

function SectionIcon({ type }: { type: DataStorySection["type"] }) {
  if (type === "readiness") return <span>📊</span>
  if (type === "group_by") return <span>📈</span>
  if (type === "correlations") return <span>🔗</span>
  if (type === "anomalies") return <span>⚠️</span>
  return <span>•</span>
}

export function DataStoryCard({ result }: DataStoryCardProps) {
  return (
    <div className="mt-2 rounded-lg border bg-card text-card-foreground overflow-hidden">
      {/* Header */}
      <div className="border-b bg-muted/30 px-3 py-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-base">📋</span>
          <span className="text-xs font-semibold text-foreground">Data Story — {result.filename}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{result.row_count.toLocaleString()} rows</span>
          <span>·</span>
          <span>{result.col_count} cols</span>
          <span
            className={`ml-1 rounded px-1.5 py-0.5 text-xs font-bold ${gradeBadgeClass(result.readiness_grade)}`}
          >
            Grade {result.readiness_grade}
          </span>
        </div>
      </div>

      {/* Readiness score bar */}
      <div className="px-3 pt-2 pb-1">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-20 shrink-0">Data quality</span>
          <div className="flex-1 h-2 rounded bg-gray-100 overflow-hidden">
            <div
              className={`h-full rounded ${scoreBarClass(result.readiness_score)}`}
              style={{ width: `${result.readiness_score}%` }}
            />
          </div>
          <span className="text-xs font-medium text-foreground w-8 text-right">
            {result.readiness_score}/100
          </span>
        </div>
      </div>

      {/* Sections */}
      {result.sections.length > 0 && (
        <div className="divide-y">
          {result.sections.map((section, i) => (
            <div key={i} className="px-3 py-2">
              <div className="flex items-center gap-1.5 mb-0.5">
                <SectionIcon type={section.type} />
                <span className="text-xs font-semibold text-foreground">{section.title}</span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">{section.insight}</p>
            </div>
          ))}
        </div>
      )}

      {/* Recommended next step */}
      <div className="border-t bg-primary/5 px-3 py-2">
        <div className="flex items-start gap-1.5">
          <span className="text-xs">→</span>
          <p className="text-xs text-primary font-medium leading-relaxed">
            {result.recommended_next_step}
          </p>
        </div>
      </div>
    </div>
  )
}
