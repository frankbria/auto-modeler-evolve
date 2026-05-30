"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import type { ClassFeatureImportanceResult, ClassImportanceGroup } from "@/lib/types"

interface ClassFeatureImportanceCardProps {
  result: ClassFeatureImportanceResult
}

const CLASS_COLORS = [
  { border: "border-indigo-200", bg: "bg-indigo-50", badge: "bg-indigo-100 text-indigo-800 border-indigo-200", bar: "bg-indigo-400" },
  { border: "border-emerald-200", bg: "bg-emerald-50", badge: "bg-emerald-100 text-emerald-800 border-emerald-200", bar: "bg-emerald-400" },
  { border: "border-amber-200", bg: "bg-amber-50", badge: "bg-amber-100 text-amber-800 border-amber-200", bar: "bg-amber-400" },
  { border: "border-violet-200", bg: "bg-violet-50", badge: "bg-violet-100 text-violet-800 border-violet-200", bar: "bg-violet-400" },
  { border: "border-sky-200", bg: "bg-sky-50", badge: "bg-sky-100 text-sky-800 border-sky-200", bar: "bg-sky-400" },
]

function getClassStyle(idx: number) {
  return CLASS_COLORS[idx % CLASS_COLORS.length]
}

function directionLabel(direction: string, deviationPct: number): string {
  const abs = Math.abs(deviationPct).toFixed(0)
  if (direction === "above_average") return `↑ ${abs}% above avg`
  if (direction === "below_average") return `↓ ${abs}% below avg`
  return "≈ similar to avg"
}

function directionColor(direction: string): string {
  if (direction === "above_average") return "text-emerald-700"
  if (direction === "below_average") return "text-rose-700"
  return "text-gray-500"
}

interface ClassPanelProps {
  group: ClassImportanceGroup
  colorStyle: (typeof CLASS_COLORS)[0]
  isOpen: boolean
  onToggle: () => void
}

function ClassPanel({ group, colorStyle, isOpen, onToggle }: ClassPanelProps) {
  const maxWeighted = Math.max(...group.features.map((f) => f.weighted_deviation), 0.001)

  return (
    <div
      className={`rounded-md border ${colorStyle.border} ${colorStyle.bg} p-3`}
      data-testid={`class-panel-${group.class_label}`}
    >
      <button
        className="w-full flex items-center justify-between gap-2 text-left"
        onClick={onToggle}
        aria-expanded={isOpen}
        aria-controls={`class-features-${group.class_label}`}
      >
        <div className="flex items-center gap-2 flex-wrap">
          <Badge
            variant="outline"
            className={`text-xs ${colorStyle.badge}`}
            data-testid={`class-label-${group.class_label}`}
          >
            {group.class_label}
          </Badge>
          <span className="text-xs text-gray-600">
            {group.sample_count.toLocaleString()} predictions ({group.sample_pct.toFixed(0)}%)
          </span>
          {group.top_feature && (
            <span className="text-xs text-gray-500 italic">
              Top: {group.top_feature}
            </span>
          )}
        </div>
        <span className="text-xs text-gray-400" aria-hidden="true">
          {isOpen ? "▲" : "▼"}
        </span>
      </button>

      {isOpen && (
        <div
          id={`class-features-${group.class_label}`}
          className="mt-3 space-y-2"
          data-testid={`class-features-${group.class_label}`}
        >
          {group.features.length === 0 ? (
            <p className="text-xs text-gray-500 italic">No feature data available.</p>
          ) : (
            group.features.map((feat) => {
              const barWidth = maxWeighted > 0
                ? Math.round((feat.weighted_deviation / maxWeighted) * 100)
                : 0
              return (
                <div
                  key={feat.feature}
                  className="flex items-center gap-2"
                  data-testid={`feat-row-${feat.feature}`}
                >
                  <span className="text-xs text-gray-700 w-28 truncate shrink-0" title={feat.feature}>
                    {feat.feature}
                  </span>
                  <div className="flex-1 bg-white/60 rounded-full h-2">
                    <div
                      className={`${colorStyle.bar} h-2 rounded-full`}
                      style={{ width: `${barWidth}%` }}
                      aria-label={`${feat.feature}: ${barWidth}% relative importance for class ${group.class_label}`}
                    />
                  </div>
                  <span
                    className={`text-xs whitespace-nowrap ${directionColor(feat.direction)}`}
                    data-testid={`feat-direction-${feat.feature}`}
                  >
                    {directionLabel(feat.direction, feat.deviation_pct)}
                  </span>
                </div>
              )
            })
          )}
          <p className="text-xs text-gray-500 italic pt-1">{group.summary}</p>
        </div>
      )}
    </div>
  )
}

export function ClassFeatureImportanceCard({ result }: ClassFeatureImportanceCardProps) {
  const [openClass, setOpenClass] = useState<string | null>(
    result?.classes?.length > 0 ? result.classes[0].class_label : null
  )

  if (!result) return null

  const toggleClass = (label: string) => {
    setOpenClass((prev) => (prev === label ? null : label))
  }

  return (
    <figure
      className="rounded-lg border border-indigo-300 bg-indigo-50/40 p-4 mt-2"
      data-testid="class-feature-importance-card"
      aria-label={`Class-conditional feature importance for ${result.target_col}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-lg">🎯</span>
        <span className="font-semibold text-sm">Class-Conditional Feature Importance</span>
        <Badge variant="outline" className="bg-indigo-100 text-indigo-800 border-indigo-200 text-xs">
          {result.n_classes} classes
        </Badge>
        <Badge variant="outline" className="bg-gray-100 text-gray-700 border-gray-200 text-xs">
          {result.algorithm}
        </Badge>
        <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200 text-xs">
          {result.target_col}
        </Badge>
        <Badge variant="outline" className="bg-slate-100 text-slate-700 border-slate-200 text-xs">
          {result.n_samples.toLocaleString()} records
        </Badge>
      </div>

      {/* Explainer */}
      <p className="text-xs text-gray-600 mb-3 italic">
        For each predicted class, shows which features deviate most from the training average —
        answering &ldquo;what makes the model predict each outcome?&rdquo;
      </p>

      {/* Per-class panels */}
      <div className="space-y-2">
        {result.classes.map((group, idx) => (
          <ClassPanel
            key={group.class_label}
            group={group}
            colorStyle={getClassStyle(idx)}
            isOpen={openClass === group.class_label}
            onToggle={() => toggleClass(group.class_label)}
          />
        ))}
      </div>

      {/* Overall summary */}
      <p
        className="text-xs text-gray-500 italic mt-3"
        data-testid="class-feature-importance-summary"
      >
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Class-conditional feature importance for {result.target_col} model ({result.algorithm}).
        {result.classes.map((c) => ` ${c.class_label}: ${c.sample_count} predictions, top feature ${c.top_feature}.`).join("")}
      </figcaption>
    </figure>
  )
}
