"use client"

import type { CostSensitiveThresholdResult } from "@/lib/types"

interface CostSensitiveThresholdCardProps {
  result: CostSensitiveThresholdResult
}

function fmt(n: number) {
  return n.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })
}

function pct(n: number) {
  return `${Math.round(n * 100)}%`
}

interface MetricsRowProps {
  label: string
  threshold: number
  cost: number
  precision: number
  recall: number
  f1: number
  tp: number
  fp: number
  fn: number
  isOptimal: boolean
}

function MetricsRow({
  label,
  threshold,
  cost,
  precision,
  recall,
  f1,
  tp,
  fp,
  fn,
  isOptimal,
}: MetricsRowProps) {
  return (
    <div
      className={`rounded-lg border p-3 ${
        isOptimal ? "border-emerald-300 bg-emerald-50" : "border-gray-200 bg-gray-50"
      }`}
      data-testid={isOptimal ? "optimal-metrics" : "default-metrics"}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-gray-800">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">threshold</span>
          <span
            className={`rounded px-2 py-0.5 text-xs font-bold ${
              isOptimal
                ? "bg-emerald-200 text-emerald-800"
                : "bg-gray-200 text-gray-700"
            }`}
          >
            {pct(threshold)}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div>
          <div className="text-gray-500">Expected Cost</div>
          <div className="font-semibold text-gray-900">${fmt(cost)}</div>
        </div>
        <div>
          <div className="text-gray-500">Precision</div>
          <div className="font-semibold text-gray-900">{pct(precision)}</div>
        </div>
        <div>
          <div className="text-gray-500">Recall</div>
          <div className="font-semibold text-gray-900">{pct(recall)}</div>
        </div>
      </div>
      <div className="mt-2 flex justify-center gap-4 text-xs text-gray-500">
        <span>
          TP: <span className="font-semibold text-emerald-700">{tp}</span>
        </span>
        <span>
          FP: <span className="font-semibold text-amber-700">{fp}</span>
        </span>
        <span>
          FN: <span className="font-semibold text-rose-700">{fn}</span>
        </span>
        <span>
          F1: <span className="font-semibold text-gray-700">{f1.toFixed(2)}</span>
        </span>
      </div>
    </div>
  )
}

export function CostSensitiveThresholdCard({ result }: CostSensitiveThresholdCardProps) {
  const isChange = result.verdict === "threshold_change_recommended"
  const savingsPositive = result.cost_savings_pct > 0
  const thrPct = Math.round(result.optimal_threshold * 100)

  return (
    <div
      className="mt-3 rounded-xl border border-violet-200 bg-white p-4 shadow-sm"
      data-testid="cost-sensitive-threshold-card"
    >
      {/* Header */}
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-gray-900">
            Cost-Sensitive Threshold Analysis
          </p>
          <p className="text-xs text-gray-500 mt-0.5">
            {result.algorithm} · target: <span className="font-medium">{result.target_col}</span>
            {" · "}positive class: <span className="font-medium">{result.positive_label}</span>
          </p>
        </div>
        <span
          className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-semibold ${
            isChange
              ? "bg-violet-100 text-violet-800"
              : "bg-emerald-100 text-emerald-800"
          }`}
          data-testid="verdict-badge"
        >
          {isChange ? "Threshold Change Recommended" : "Default Near-Optimal"}
        </span>
      </div>

      {/* Cost inputs */}
      <div className="mb-3 grid grid-cols-3 gap-2 text-center">
        <div className="rounded-lg bg-amber-50 border border-amber-200 p-2">
          <div className="text-xs text-amber-600">False Positive Cost</div>
          <div className="text-sm font-bold text-amber-800">${fmt(result.fp_cost)}</div>
        </div>
        <div className="rounded-lg bg-rose-50 border border-rose-200 p-2">
          <div className="text-xs text-rose-600">False Negative Cost</div>
          <div className="text-sm font-bold text-rose-800">${fmt(result.fn_cost)}</div>
        </div>
        <div className="rounded-lg bg-violet-50 border border-violet-200 p-2">
          <div className="text-xs text-violet-600">Optimal Threshold</div>
          <div className="text-sm font-bold text-violet-800">{thrPct}%</div>
          <div className="text-xs text-violet-500">
            = ${fmt(result.fp_cost)} ÷ ${fmt(result.fp_cost + result.fn_cost)}
          </div>
        </div>
      </div>

      {/* Comparison */}
      <div className="space-y-2 mb-3">
        <MetricsRow
          label="Default (50%)"
          threshold={result.default_metrics.threshold}
          cost={result.default_cost}
          precision={result.default_metrics.precision}
          recall={result.default_metrics.recall}
          f1={result.default_metrics.f1}
          tp={result.default_metrics.tp}
          fp={result.default_metrics.fp}
          fn={result.default_metrics.fn}
          isOptimal={false}
        />
        <MetricsRow
          label={`Cost-Optimal (${thrPct}%)`}
          threshold={result.optimal_metrics.threshold}
          cost={result.optimal_cost}
          precision={result.optimal_metrics.precision}
          recall={result.optimal_metrics.recall}
          f1={result.optimal_metrics.f1}
          tp={result.optimal_metrics.tp}
          fp={result.optimal_metrics.fp}
          fn={result.optimal_metrics.fn}
          isOptimal
        />
      </div>

      {/* Savings banner */}
      {savingsPositive && (
        <div
          className="mb-3 rounded-lg bg-emerald-50 border border-emerald-200 px-3 py-2 text-center"
          data-testid="savings-banner"
        >
          <span className="text-sm font-semibold text-emerald-800">
            {result.cost_savings_pct.toFixed(1)}% cost reduction
          </span>
          <span className="text-xs text-emerald-600 ml-2">
            (${fmt(result.default_cost)} → ${fmt(result.optimal_cost)} per{" "}
            {result.n_samples.toLocaleString()} predictions)
          </span>
        </div>
      )}

      {/* Retrain hint */}
      <div className="rounded-lg bg-sky-50 border border-sky-200 px-3 py-2">
        <p className="text-xs text-sky-700">
          <span className="font-semibold">Retrain hint:</span> To bake this bias into training,
          use{" "}
          <span className="font-mono bg-sky-100 px-1 rounded">
            class_weight = {`{positive: ${result.suggested_class_weight}×}`}
          </span>{" "}
          (FN/FP cost ratio). This is equivalent to upweighting missed{" "}
          <span className="font-semibold">{result.positive_label}</span> cases during training.
        </p>
      </div>
    </div>
  )
}
