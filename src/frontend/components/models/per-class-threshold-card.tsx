"use client"

import { useState } from "react"
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { PerClassThresholdEntry, PerClassThresholdResult } from "@/lib/types"

interface PerClassThresholdCardProps {
  result: PerClassThresholdResult
}

function ClassRow({ entry }: { entry: PerClassThresholdEntry }) {
  const [expanded, setExpanded] = useState(false)

  const directionLabel =
    entry.direction === "raise"
      ? `↑ Raise to ${Math.round(entry.optimal_threshold * 100)}%`
      : entry.direction === "lower"
        ? `↓ Lower to ${Math.round(entry.optimal_threshold * 100)}%`
        : "✓ Default OK"

  const directionColor =
    entry.direction === "raise"
      ? "bg-sky-100 text-sky-800"
      : entry.direction === "lower"
        ? "bg-amber-100 text-amber-800"
        : "bg-emerald-100 text-emerald-800"

  const f1GainPct = Math.round(entry.f1_gain * 100)

  const chartData = entry.sweep.map((pt) => ({
    threshold: `${Math.round(pt.threshold * 100)}%`,
    Precision: Math.round(pt.precision * 100),
    Recall: Math.round(pt.recall * 100),
    F1: Math.round(pt.f1 * 100),
  }))

  const optimalLabel = `${Math.round(entry.optimal_threshold * 100)}%`

  return (
    <div
      className={`rounded-lg border p-3 ${entry.is_actionable ? "border-violet-200 bg-violet-50" : "border-gray-200 bg-gray-50"}`}
      data-testid="per-class-row"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold text-sm text-gray-900" data-testid="class-name">
          {entry.class_name}
        </span>
        <span
          className={`rounded px-2 py-0.5 text-xs font-semibold ${directionColor}`}
          data-testid="direction-badge"
        >
          {directionLabel}
        </span>
        {entry.is_actionable && f1GainPct > 0 && (
          <span className="rounded px-2 py-0.5 text-xs font-medium bg-emerald-100 text-emerald-800" data-testid="f1-gain-badge">
            +{f1GainPct}pp F1
          </span>
        )}
        <span className="ml-auto text-xs text-gray-500">
          {Math.round(entry.prevalence * 100)}% prevalence · {entry.n_positive} positives
        </span>
        {entry.is_actionable && (
          <button
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
            className="text-xs text-violet-600 hover:underline focus-visible:ring-2 focus-visible:ring-violet-400 rounded"
          >
            {expanded ? "Hide chart" : "Show sweep"}
          </button>
        )}
      </div>

      <p className="mt-1.5 text-xs text-gray-600 leading-relaxed" data-testid="class-recommendation">
        {entry.recommendation}
      </p>

      {expanded && entry.is_actionable && (
        <div className="mt-3" data-testid="class-sweep-chart">
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="threshold" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
              <Tooltip formatter={(v) => (typeof v === "number" ? `${v}%` : v)} />
              <ReferenceLine
                x={optimalLabel}
                stroke="#7c3aed"
                strokeDasharray="4 2"
                label={{ value: "Optimal", position: "top", fontSize: 9, fill: "#7c3aed" }}
              />
              <Line type="monotone" dataKey="Precision" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="Recall" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="F1" stroke="#7c3aed" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-1 justify-center text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-blue-500" />Precision
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-amber-500" />Recall
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-violet-600" />F1
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export function PerClassThresholdCard({ result }: PerClassThresholdCardProps) {
  const actionableCount = result.n_actionable
  const borderColor = actionableCount > 0 ? "border-violet-300 bg-violet-50" : "border-gray-200 bg-gray-50"

  return (
    <figure
      className={`rounded-xl border p-4 space-y-4 ${borderColor}`}
      role="region"
      aria-label="Per-class threshold analysis"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-lg" aria-hidden="true">🎯</span>
        <span className="font-semibold text-gray-800">Per-Class Threshold Tuning</span>
        <span className="rounded-full px-2.5 py-0.5 text-xs font-medium bg-violet-200 text-violet-800" data-testid="n-classes-badge">
          {result.n_classes} classes
        </span>
        {actionableCount > 0 && (
          <span className="rounded-full px-2.5 py-0.5 text-xs font-medium bg-amber-200 text-amber-800" data-testid="actionable-badge">
            {actionableCount} actionable
          </span>
        )}
        <span className="rounded-full px-2.5 py-0.5 text-xs font-medium bg-gray-200 text-gray-700">
          Target: {result.target_col}
        </span>
      </div>

      {/* Summary */}
      <p className="text-sm text-gray-700 leading-relaxed" data-testid="summary">
        {result.summary}
      </p>

      {/* Per-class rows */}
      <div className="space-y-2">
        {result.classes.map((cls) => (
          <ClassRow key={cls.class_index} entry={cls} />
        ))}
      </div>

      {/* Footer guide */}
      <div className="rounded-lg bg-white/60 border border-violet-200 p-3 text-xs text-gray-600 leading-relaxed">
        <strong>How to use:</strong> Lower thresholds catch more cases of that class (higher recall, more false alarms).
        Raise thresholds to reduce false alarms (higher precision, may miss some cases).
        Use this when different classes have different costs for errors — e.g., missing a fraud case is worse than a false flag.
      </div>

      <figcaption className="sr-only">
        Per-class threshold analysis for {result.target_col} ({result.n_classes} classes, {result.n_samples} samples).
        {actionableCount > 0
          ? ` ${actionableCount} classes benefit from threshold adjustment.`
          : " All classes are optimal at the default 50% threshold."}
      </figcaption>
    </figure>
  )
}
