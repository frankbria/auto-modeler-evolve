"use client"

import { PopulationCounterfactualResult, PopulationCFFeatureSummary } from "@/lib/types"

interface PopulationCounterfactualCardProps {
  data: PopulationCounterfactualResult
}

function FeatureSummaryRow({ row }: { row: PopulationCFFeatureSummary }) {
  const isIncrease = row.direction === "increase"
  const arrowLabel = isIncrease ? "▲" : "▼"
  const arrowClass = isIncrease ? "text-emerald-600" : "text-rose-600"
  const badgeClass = isIncrease
    ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
    : "bg-rose-50 text-rose-700 border border-rose-200"

  return (
    <tr
      className="border-b border-gray-100 last:border-0"
      data-testid="population-cf-feature-row"
    >
      <td className="py-2 pr-3 text-sm font-medium text-gray-700">
        {row.feature.replace(/_/g, " ")}
      </td>
      <td className="py-2 pr-3 text-sm tabular-nums font-semibold text-gray-800">
        {row.flip_count}
        <span className="text-xs text-gray-400 font-normal ml-1">
          ({row.flip_pct.toFixed(0)}%)
        </span>
      </td>
      <td className="py-2 pr-3">
        <span className={`${arrowClass} text-sm font-bold`} aria-hidden="true">
          {arrowLabel}
        </span>
      </td>
      <td className="py-2">
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeClass}`}
          aria-label={`${row.avg_change_pct.toFixed(1)} percent ${row.direction} on average`}
        >
          ~{row.avg_change_pct.toFixed(1)}% {row.direction}
        </span>
      </td>
    </tr>
  )
}

export function PopulationCounterfactualCard({ data }: PopulationCounterfactualCardProps) {
  const hasFlips = data.flipped_count > 0
  const flipPct = data.total_rows > 0
    ? Math.round((data.flipped_count / data.total_rows) * 100)
    : 0

  const borderClass = hasFlips
    ? "border-amber-300 bg-amber-50"
    : "border-rose-300 bg-rose-50"

  const headerBadge = hasFlips ? (
    <span
      className="text-xs px-2 py-0.5 rounded-full font-medium bg-amber-100 text-amber-700 border border-amber-300"
      data-testid="flip-badge"
    >
      {data.flipped_count} of {data.total_rows} flippable
    </span>
  ) : (
    <span
      className="text-xs px-2 py-0.5 rounded-full font-medium bg-rose-100 text-rose-700 border border-rose-300"
      data-testid="no-flip-badge"
    >
      No flips found
    </span>
  )

  return (
    <figure
      className={`rounded-lg border-2 p-4 my-2 ${borderClass}`}
      aria-label={`Population counterfactual: ${data.flipped_count} of ${data.total_rows} records could be flipped`}
      data-testid="population-counterfactual-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className="text-base" aria-hidden="true">🎯</span>
        <span className="font-semibold text-gray-800 text-sm">
          Population Counterfactual
        </span>
        {headerBadge}
        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 border border-gray-200">
          {data.target_column.replace(/_/g, " ")}
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200">
          {data.total_rows} records
        </span>
      </div>

      {/* Flip rate summary */}
      <div
        className="rounded-md bg-white border border-gray-200 p-3 mb-4"
        role="region"
        aria-label="Flip rate summary"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-500 font-medium">Records that can be flipped</span>
          <span
            className="text-sm font-bold text-gray-800 tabular-nums"
            data-testid="flip-rate"
          >
            {data.flipped_count} / {data.total_rows} ({flipPct}%)
          </span>
        </div>
        <div
          className="w-full bg-gray-100 rounded-full h-2"
          role="progressbar"
          aria-valuenow={flipPct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${flipPct} percent of records can be flipped`}
        >
          <div
            className={`h-2 rounded-full ${hasFlips ? "bg-amber-400" : "bg-rose-300"}`}
            style={{ width: `${flipPct}%` }}
          />
        </div>
      </div>

      {/* Dominant intervention highlight */}
      {data.dominant_feature && hasFlips && (
        <div
          className="rounded-md bg-amber-100 border border-amber-200 p-3 mb-4"
          data-testid="dominant-intervention"
          role="region"
          aria-label={`Dominant intervention: ${data.dominant_direction} ${data.dominant_feature}`}
        >
          <p className="text-xs font-semibold text-amber-800 uppercase tracking-wide mb-1">
            Most Impactful Single Lever
          </p>
          <p className="text-sm text-amber-900 font-medium">
            <span
              className={`inline-block mr-1 font-bold ${
                data.dominant_direction === "increase" ? "text-emerald-700" : "text-rose-700"
              }`}
              aria-hidden="true"
            >
              {data.dominant_direction === "increase" ? "▲" : "▼"}
            </span>
            {data.dominant_direction === "increase" ? "Increase" : "Decrease"}{" "}
            <strong>&ldquo;{data.dominant_feature.replace(/_/g, " ")}&rdquo;</strong>
            {" "}by ~{data.dominant_avg_change_pct.toFixed(1)}%
          </p>
          <p className="text-xs text-amber-700 mt-1">
            Flips{" "}
            <span data-testid="dominant-flip-count" className="font-semibold">
              {data.dominant_flip_count}
            </span>{" "}
            of {data.total_rows} records (
            {Math.round((data.dominant_flip_count / data.total_rows) * 100)}%)
          </p>
        </div>
      )}

      {/* Feature summary table */}
      {data.feature_summary.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2">
            Feature Breakdown
          </p>
          <div className="overflow-x-auto">
            <table
              className="w-full text-left"
              aria-label="Feature intervention breakdown"
            >
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="pb-1 pr-3 text-xs text-gray-500 font-medium">Feature</th>
                  <th className="pb-1 pr-3 text-xs text-gray-500 font-medium">Rows flipped</th>
                  <th className="pb-1 pr-3 text-xs text-gray-500 font-medium">Dir.</th>
                  <th className="pb-1 text-xs text-gray-500 font-medium">Avg change</th>
                </tr>
              </thead>
              <tbody>
                {data.feature_summary.map((row) => (
                  <FeatureSummaryRow key={`${row.feature}-${row.direction}`} row={row} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* No flips state */}
      {!hasFlips && (
        <p className="text-sm text-rose-700 mb-3">
          None of the {data.total_rows} analysed records could be flipped within the
          step budget. This cohort appears firmly in its current prediction category.
        </p>
      )}

      {/* Summary */}
      <p
        className="text-sm text-gray-700 mt-2"
        data-testid="population-cf-summary"
      >
        {data.summary}
      </p>

      <figcaption className="sr-only">
        Population counterfactual analysis across {data.total_rows} records.
        {hasFlips
          ? ` ${data.flipped_count} records (${flipPct}%) can be flipped.`
          : ` No records could be flipped.`}
        {data.dominant_feature
          ? ` Dominant lever: ${data.dominant_direction} "${data.dominant_feature}" by ~${data.dominant_avg_change_pct.toFixed(1)}%.`
          : ""}
      </figcaption>
    </figure>
  )
}
