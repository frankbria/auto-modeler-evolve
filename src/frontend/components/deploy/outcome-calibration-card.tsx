"use client"

import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import {
  OutcomeCalibrationBucket,
  OutcomeCalibrationResult,
} from "@/lib/types"

interface OutcomeCalibrationCardProps {
  result: OutcomeCalibrationResult
}

type Verdict = OutcomeCalibrationResult["verdict"]

function verdictColors(v: Verdict): { border: string; badge: string } {
  switch (v) {
    case "well_calibrated":
    case "unbiased":
      return { border: "border-emerald-500", badge: "bg-emerald-100 text-emerald-800" }
    case "slightly_overconfident":
    case "slightly_underconfident":
    case "low_data":
      return { border: "border-amber-500", badge: "bg-amber-100 text-amber-800" }
    case "overconfident":
    case "underconfident":
    case "positive_bias":
    case "negative_bias":
      return { border: "border-rose-500", badge: "bg-rose-100 text-rose-800" }
    case "no_data":
    case "not_applicable":
    default:
      return { border: "border-gray-300", badge: "bg-gray-100 text-gray-600" }
  }
}

function verdictLabel(v: Verdict): string {
  switch (v) {
    case "well_calibrated": return "Well Calibrated"
    case "slightly_overconfident": return "Slightly Overconfident"
    case "slightly_underconfident": return "Slightly Underconfident"
    case "overconfident": return "Overconfident"
    case "underconfident": return "Underconfident"
    case "unbiased": return "Unbiased"
    case "positive_bias": return "Positive Bias"
    case "negative_bias": return "Negative Bias"
    case "low_data": return "Low Data"
    case "no_data": return "No Data"
    case "not_applicable": return "N/A"
    default: return v
  }
}

function ClassificationChart({ buckets }: { buckets: OutcomeCalibrationBucket[] }) {
  const filled = buckets.filter((b) => b.n > 0)
  if (filled.length === 0) return null

  const data = filled.map((b) => ({
    label: `${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%`,
    confidence: b.mean_confidence != null ? Math.round(b.mean_confidence * 1000) / 10 : null,
    accuracy: b.actual_accuracy != null ? Math.round(b.actual_accuracy * 1000) / 10 : null,
    n: b.n,
  }))

  // Perfect calibration reference line data points
  const perfect = filled.map((b) => ({
    label: `${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%`,
    perfect: b.mean_confidence != null ? Math.round(b.mean_confidence * 1000) / 10 : null,
  }))

  const combined = data.map((d, i) => ({ ...d, perfect: perfect[i].perfect }))

  return (
    <div className="mt-4">
      <p className="text-xs text-gray-500 mb-2">
        Confidence Bucket → Actual Accuracy (diagonal = perfect calibration)
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={combined} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10 }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 10 }}
            tickLine={false}
            unit="%"
          />
          <Tooltip
            formatter={(value, name) => [
              `${Number(value ?? 0).toFixed(1)}%`,
              String(name ?? "") === "accuracy" ? "Actual Accuracy" : String(name ?? "") === "confidence" ? "Mean Confidence" : "Perfect",
            ]}
          />
          <Line
            type="monotone"
            dataKey="perfect"
            stroke="#d1d5db"
            strokeDasharray="4 2"
            dot={false}
            name="Perfect"
          />
          <Line
            type="monotone"
            dataKey="confidence"
            stroke="#6366f1"
            strokeWidth={1.5}
            dot={{ r: 3 }}
            name="Confidence"
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ r: 4, fill: "#10b981" }}
            name="Accuracy"
          />
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="sr-only">
        Reliability diagram showing predicted confidence vs actual accuracy per bucket.
        The dashed gray line is perfect calibration.
      </figcaption>
    </div>
  )
}

function RegressionChart({ bins, mean_error }: { bins: { lo: number; hi: number; count: number }[]; mean_error: number }) {
  const data = bins.map((b) => ({
    label: `${b.lo.toFixed(2)}`,
    count: b.count,
  }))

  return (
    <div className="mt-4">
      <p className="text-xs text-gray-500 mb-2">
        Prediction Error Distribution (actual − predicted)
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="label" tick={{ fontSize: 9 }} tickLine={false} />
          <YAxis tick={{ fontSize: 10 }} tickLine={false} />
          <Tooltip formatter={(v) => [Number(v ?? 0), "Count"]} />
          <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
          <ReferenceLine x={mean_error.toFixed(2)} stroke="#ef4444" strokeDasharray="4 2" />
        </BarChart>
      </ResponsiveContainer>
      <figcaption className="sr-only">
        Error histogram showing distribution of actual minus predicted values.
        The red dashed line marks the mean error (bias).
      </figcaption>
    </div>
  )
}

export default function OutcomeCalibrationCard({ result }: OutcomeCalibrationCardProps) {
  const { border, badge } = verdictColors(result.verdict)
  const label = verdictLabel(result.verdict)
  const isClassification = result.problem_type === "classification"

  return (
    <div className={`rounded-lg border-l-4 ${border} bg-white shadow-sm p-4 my-2 max-w-xl`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-base" aria-hidden="true">📐</span>
        <span className="font-semibold text-sm text-gray-800">
          Prediction Outcome Calibration
        </span>
        <span className={`ml-auto text-xs font-medium px-2 py-0.5 rounded-full ${badge}`}>
          {label}
        </span>
      </div>

      {/* Header badges */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
          {isClassification ? "Classification" : "Regression"}
        </span>
        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
          {result.n_samples} outcome{result.n_samples !== 1 ? "s" : ""}
        </span>
        {result.algorithm && (
          <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
            {result.algorithm}
          </span>
        )}
        {result.target_col && (
          <span className="text-xs bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full">
            {result.target_col}
          </span>
        )}
      </div>

      {/* Summary */}
      <p className="text-sm text-gray-700 mb-3">{result.summary}</p>

      {/* Classification stats */}
      {isClassification && result.ece != null && result.verdict !== "no_data" && result.verdict !== "not_applicable" && (
        <>
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className="bg-gray-50 rounded-md p-2 text-center">
              <div className="text-lg font-bold text-gray-900">{(result.ece * 100).toFixed(1)}%</div>
              <div className="text-xs text-gray-500">ECE (lower is better)</div>
            </div>
            {result.overall_accuracy != null && (
              <div className="bg-gray-50 rounded-md p-2 text-center">
                <div className="text-lg font-bold text-gray-900">
                  {(result.overall_accuracy * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500">Overall Accuracy</div>
              </div>
            )}
          </div>
          {result.buckets && (
            <ClassificationChart buckets={result.buckets} />
          )}
          {result.n_filled_buckets != null && (
            <p className="text-xs text-gray-400 mt-1">
              {result.n_filled_buckets} of 10 confidence buckets have data
            </p>
          )}
        </>
      )}

      {/* Regression stats */}
      {!isClassification && result.verdict !== "no_data" && result.verdict !== "not_applicable" && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
            {result.mae != null && (
              <div className="bg-gray-50 rounded-md p-2 text-center">
                <div className="text-base font-bold text-gray-900">{result.mae.toFixed(3)}</div>
                <div className="text-xs text-gray-500">MAE</div>
              </div>
            )}
            {result.rmse != null && (
              <div className="bg-gray-50 rounded-md p-2 text-center">
                <div className="text-base font-bold text-gray-900">{result.rmse.toFixed(3)}</div>
                <div className="text-xs text-gray-500">RMSE</div>
              </div>
            )}
            {result.mean_error != null && (
              <div className="bg-gray-50 rounded-md p-2 text-center">
                <div className="text-base font-bold text-gray-900">
                  {result.mean_error >= 0 ? "+" : ""}{result.mean_error.toFixed(3)}
                </div>
                <div className="text-xs text-gray-500">Bias (mean error)</div>
              </div>
            )}
            {result.std_error != null && (
              <div className="bg-gray-50 rounded-md p-2 text-center">
                <div className="text-base font-bold text-gray-900">{result.std_error.toFixed(3)}</div>
                <div className="text-xs text-gray-500">Std Dev</div>
              </div>
            )}
          </div>
          {result.bins && result.mean_error != null && (
            <RegressionChart bins={result.bins} mean_error={result.mean_error} />
          )}
        </>
      )}

      {/* Footer info */}
      <div className="mt-3 pt-2 border-t border-gray-100">
        <p className="text-xs text-gray-400">
          {isClassification
            ? "Built from production feedback outcomes. Distinct from training-time calibration check."
            : "Error = actual − predicted. Positive mean error = model under-predicts on average."}
        </p>
      </div>
    </div>
  )
}
