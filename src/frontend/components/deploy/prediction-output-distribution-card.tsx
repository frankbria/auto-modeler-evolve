"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { PredictionOutputDistributionShiftResult, PredictionDistHistogramBin } from "@/lib/types"

interface Props {
  result: PredictionOutputDistributionShiftResult
}

function verdictColor(verdict: string): string {
  if (verdict === "significant_shift") return "rose"
  if (verdict === "moderate_shift") return "amber"
  if (verdict === "stable") return "emerald"
  return "gray"
}

function StatCell({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-sm font-mono font-medium text-gray-800">{value}</span>
    </div>
  )
}

function HistogramChart({
  trainingBins,
  productionBins,
}: {
  trainingBins: PredictionDistHistogramBin[]
  productionBins: PredictionDistHistogramBin[]
}) {
  const data = trainingBins.map((bin, i) => ({
    label: bin.label,
    training: bin.count,
    production: productionBins[i]?.count ?? 0,
  }))

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 9 }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis tick={{ fontSize: 10 }} tickLine={false} width={28} />
        <Tooltip
          contentStyle={{ fontSize: 11 }}
          formatter={(value, name) => [
            value,
            name === "training" ? "Training" : "Production",
          ]}
        />
        <Legend
          iconSize={10}
          formatter={(value) => (value === "training" ? "Training" : "Production")}
          wrapperStyle={{ fontSize: 11 }}
        />
        <Bar dataKey="training" fill="#94a3b8" name="training" />
        <Bar dataKey="production" fill="#6366f1" name="production" />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function PredictionOutputDistributionCard({ result }: Props) {
  const color = verdictColor(result.verdict)
  const borderClass =
    color === "rose"
      ? "border-rose-300 bg-rose-50"
      : color === "amber"
        ? "border-amber-300 bg-amber-50"
        : color === "emerald"
          ? "border-emerald-300 bg-emerald-50"
          : "border-gray-200 bg-gray-50"
  const badgeClass =
    color === "rose"
      ? "bg-rose-100 text-rose-800"
      : color === "amber"
        ? "bg-amber-100 text-amber-800"
        : color === "emerald"
          ? "bg-emerald-100 text-emerald-800"
          : "bg-gray-100 text-gray-700"

  const isNoData = result.verdict === "no_data"

  const shiftSign =
    result.mean_shift_pct !== undefined && result.mean_shift_pct > 0 ? "+" : ""
  const shiftLabel =
    result.mean_shift_pct !== undefined
      ? `${shiftSign}${result.mean_shift_pct.toFixed(1)}%`
      : "N/A"

  return (
    <figure
      className={`rounded-lg border-2 p-4 text-sm ${borderClass}`}
      aria-label="Prediction output distribution shift analysis"
      data-testid="prediction-output-distribution-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className="text-base" role="img" aria-label="distribution shift">
          📊
        </span>
        <span className="font-semibold text-gray-800">Output Distribution Shift</span>
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeClass}`}
          data-testid="verdict-badge"
        >
          {result.verdict_label}
        </span>
        {result.problem_type && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 font-medium">
            {result.problem_type}
          </span>
        )}
        {!isNoData && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 font-medium">
            {result.n_training} training / {result.n_production} production predictions
          </span>
        )}
      </div>

      {isNoData ? (
        <p className="text-gray-600 text-sm">{result.summary}</p>
      ) : (
        <>
          {/* KS test + shift summary row */}
          <div className="grid grid-cols-3 gap-3 mb-4 p-3 rounded-md bg-white border border-gray-200">
            <StatCell
              label="KS Statistic"
              value={result.ks_statistic?.toFixed(3) ?? "N/A"}
            />
            <StatCell
              label="p-value"
              value={result.ks_p_value?.toFixed(4) ?? "N/A"}
            />
            <StatCell label="Mean Shift" value={shiftLabel} />
          </div>

          {/* Distribution stats comparison */}
          {result.training_stats && result.production_stats && (
            <div className="mb-4">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
                Distribution Comparison
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse" role="table">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="text-left py-1.5 px-2 font-medium text-gray-600 border border-gray-200">
                        Metric
                      </th>
                      <th className="text-right py-1.5 px-2 font-medium text-slate-600 border border-gray-200">
                        Training
                      </th>
                      <th className="text-right py-1.5 px-2 font-medium text-indigo-600 border border-gray-200">
                        Production
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {(
                      [
                        ["Mean", "mean"],
                        ["Std Dev", "std"],
                        ["Min", "min"],
                        ["P25", "p25"],
                        ["Median", "median"],
                        ["P75", "p75"],
                        ["P95", "p95"],
                        ["Max", "max"],
                      ] as [string, keyof typeof result.training_stats][]
                    ).map(([label, key]) => (
                      <tr key={key} className="even:bg-gray-50">
                        <td className="py-1 px-2 text-gray-600 border border-gray-200">
                          {label}
                        </td>
                        <td className="py-1 px-2 text-right font-mono text-slate-700 border border-gray-200">
                          {result.training_stats![key].toFixed(3)}
                        </td>
                        <td className="py-1 px-2 text-right font-mono text-indigo-700 border border-gray-200">
                          {result.production_stats![key].toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Histogram overlay */}
          {result.training_histogram && result.production_histogram && (
            <div className="mb-4">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
                Prediction Value Distribution (Training vs Production)
              </p>
              <HistogramChart
                trainingBins={result.training_histogram}
                productionBins={result.production_histogram}
              />
            </div>
          )}

          {/* Summary */}
          <p className="text-gray-700 text-sm mt-2">{result.summary}</p>
        </>
      )}

      <figcaption className="sr-only">
        {`Prediction output distribution shift: ${result.verdict_label}. ${result.summary}`}
      </figcaption>
    </figure>
  )
}
