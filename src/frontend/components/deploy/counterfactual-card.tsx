"use client"

import { CounterfactualResult, CounterfactualChangedFeature } from "@/lib/types"

interface CounterfactualCardProps {
  data: CounterfactualResult
}

function formatValue(v: number): string {
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return v.toFixed(2)
}

function FeatureChangeRow({ feature }: { feature: CounterfactualChangedFeature }) {
  const isIncrease = feature.direction === "increase"
  const arrowLabel = isIncrease ? "▲" : "▼"
  const arrowClass = isIncrease ? "text-emerald-600" : "text-rose-600"
  const badgeClass = isIncrease
    ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
    : "bg-rose-50 text-rose-700 border border-rose-200"

  return (
    <tr className="border-b border-gray-100 last:border-0">
      <td className="py-2 pr-3 text-sm font-medium text-gray-700">
        {feature.name.replace(/_/g, " ")}
      </td>
      <td className="py-2 pr-3 text-sm text-gray-600 tabular-nums">
        {formatValue(feature.original_value)}
      </td>
      <td className="py-2 pr-3">
        <span className={`${arrowClass} text-sm font-bold`} aria-hidden="true">
          {arrowLabel}
        </span>
      </td>
      <td className="py-2 pr-3 text-sm font-semibold text-gray-900 tabular-nums">
        {formatValue(feature.counterfactual_value)}
      </td>
      <td className="py-2">
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeClass}`}
          aria-label={`${Math.abs(feature.change_pct).toFixed(1)} percent ${feature.direction}`}
        >
          {isIncrease ? "+" : ""}{feature.change_pct.toFixed(1)}%
        </span>
      </td>
    </tr>
  )
}

export function CounterfactualCard({ data }: CounterfactualCardProps) {
  const borderClass = data.flipped
    ? "border-amber-300 bg-amber-50"
    : "border-rose-300 bg-rose-50"

  const headerBadge = data.flipped ? (
    <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-amber-100 text-amber-700 border border-amber-300">
      Flip found ✓
    </span>
  ) : (
    <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-rose-100 text-rose-700 border border-rose-300">
      No flip found
    </span>
  )

  return (
    <figure
      className={`rounded-lg border-2 p-4 my-2 ${borderClass}`}
      aria-label={`Counterfactual explanation for row ${data.row_index}`}
      data-testid="counterfactual-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className="text-base" aria-hidden="true">🔀</span>
        <span className="font-semibold text-gray-800 text-sm">
          Counterfactual Explanation — Row {data.row_index}
        </span>
        {headerBadge}
        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 border border-gray-200">
          {data.target_column.replace(/_/g, " ")}
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200">
          Classification
        </span>
      </div>

      {/* Original → Counterfactual prediction boxes */}
      <div
        className="grid grid-cols-2 gap-3 mb-4"
        role="group"
        aria-label="Original and counterfactual predictions"
      >
        <div className="rounded-md bg-white border border-gray-200 p-3 text-center">
          <p className="text-xs text-gray-500 mb-1">Original prediction</p>
          <p
            className="text-lg font-bold text-gray-800"
            data-testid="original-prediction"
          >
            {data.original_class}
          </p>
          <p
            className="text-xs text-gray-500 mt-1"
            data-testid="original-confidence"
          >
            {(data.original_confidence * 100).toFixed(0)}% confidence
          </p>
        </div>
        <div
          className={`rounded-md border p-3 text-center ${
            data.flipped
              ? "bg-amber-50 border-amber-200"
              : "bg-gray-50 border-gray-200"
          }`}
        >
          <p className="text-xs text-gray-500 mb-1">
            {data.flipped ? "Would become" : "Target class"}
          </p>
          <p
            className={`text-lg font-bold ${
              data.flipped ? "text-amber-700" : "text-gray-400"
            }`}
            data-testid="counterfactual-prediction"
          >
            {data.flipped ? data.counterfactual_prediction : data.target_class}
          </p>
          {data.counterfactual_confidence !== null && data.flipped && (
            <p className="text-xs text-gray-500 mt-1">
              {(data.counterfactual_confidence * 100).toFixed(0)}% confidence
            </p>
          )}
          {!data.flipped && (
            <p className="text-xs text-gray-400 mt-1">Not reachable in {data.n_steps} steps</p>
          )}
        </div>
      </div>

      {/* Changed features table */}
      {data.changed_features.length > 0 ? (
        <div className="mb-3">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2">
            Changes needed
          </p>
          <div className="overflow-x-auto">
            <table
              className="w-full text-left"
              aria-label="Feature changes needed for counterfactual"
            >
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="pb-1 pr-3 text-xs text-gray-500 font-medium">Feature</th>
                  <th className="pb-1 pr-3 text-xs text-gray-500 font-medium">Current</th>
                  <th className="pb-1 pr-3 text-xs text-gray-500 font-medium"></th>
                  <th className="pb-1 pr-3 text-xs text-gray-500 font-medium">Needed</th>
                  <th className="pb-1 text-xs text-gray-500 font-medium">Change</th>
                </tr>
              </thead>
              <tbody>
                {data.changed_features.map((feature) => (
                  <FeatureChangeRow key={feature.name} feature={feature} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        data.flipped && (
          <p className="text-sm text-gray-500 mb-3 italic">
            No feature changes were needed — prediction is already the target class.
          </p>
        )
      )}

      {!data.flipped && data.changed_features.length === 0 && (
        <p className="text-sm text-rose-700 mb-3">
          No numeric features could be adjusted to reach class &lsquo;{data.target_class}&rsquo;.
          The model places this record firmly in class &lsquo;{data.original_class}&rsquo;.
        </p>
      )}

      {/* Summary */}
      <p
        className="text-sm text-gray-700 mt-2"
        data-testid="counterfactual-summary"
      >
        {data.summary}
      </p>

      <figcaption className="sr-only">
        Counterfactual explanation for row {data.row_index}.
        Original prediction: {data.original_class} with {(data.original_confidence * 100).toFixed(0)}% confidence.
        Target class: {data.target_class}.
        {data.flipped
          ? ` Prediction can be flipped by changing ${data.changed_features.length} feature(s).`
          : ` Prediction could not be flipped within ${data.n_steps} steps.`}
      </figcaption>
    </figure>
  )
}
