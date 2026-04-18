"use client"

import { Badge } from "@/components/ui/badge"
import type { ConfusionMatrixChatResult } from "@/lib/types"

interface Props {
  result: ConfusionMatrixChatResult
}

function accuracyColor(accuracy: number): string {
  if (accuracy >= 0.85) return "border-emerald-300 bg-emerald-50"
  if (accuracy >= 0.7) return "border-amber-300 bg-amber-50"
  return "border-rose-300 bg-rose-50"
}

function cellBg(row: number, col: number, value: number, rowTotal: number): string {
  if (row === col) {
    const ratio = rowTotal > 0 ? value / rowTotal : 0
    if (ratio >= 0.8) return "bg-emerald-100 text-emerald-800 font-semibold"
    if (ratio >= 0.5) return "bg-amber-100 text-amber-800"
    return "bg-rose-100 text-rose-800"
  }
  if (value === 0) return "bg-gray-50 text-gray-400"
  return "bg-rose-50 text-rose-700"
}

export function ConfusionMatrixChatCard({ result }: Props) {
  const { matrix, labels, total, correct, accuracy, per_class_metrics, most_confused_pair, summary, algorithm_plain, target_col } = result

  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0))

  return (
    <figure
      className={`rounded-lg border p-4 my-2 ${accuracyColor(accuracy)}`}
      aria-label="Confusion matrix"
    >
      <div className="flex items-center gap-2 mb-3">
        <span role="img" aria-label="target" className="text-xl">🎯</span>
        <span className="font-semibold text-gray-800">Confusion Matrix</span>
        <Badge className="bg-blue-100 text-blue-800 text-xs">{algorithm_plain}</Badge>
        <Badge className="bg-purple-100 text-purple-800 text-xs">Target: {target_col}</Badge>
        <Badge
          className={`text-xs ${accuracy >= 0.85 ? "bg-emerald-100 text-emerald-800" : accuracy >= 0.7 ? "bg-amber-100 text-amber-800" : "bg-rose-100 text-rose-800"}`}
        >
          {(accuracy * 100).toFixed(1)}% accurate
        </Badge>
      </div>

      <p className="text-sm text-gray-600 mb-3">
        {correct} of {total} predictions correct
      </p>

      {/* Matrix grid */}
      <div className="overflow-x-auto mb-4">
        <table className="text-xs border-collapse" aria-label="Confusion matrix grid">
          <thead>
            <tr>
              <th className="p-1 text-gray-500 font-normal text-right pr-2" colSpan={2} />
              <th className="p-1 text-gray-600 font-semibold text-center" colSpan={labels.length}>
                Predicted
              </th>
            </tr>
            <tr>
              <th className="p-1" colSpan={2} />
              {labels.map((lbl) => (
                <th key={lbl} className="p-1 text-center text-gray-600 font-medium min-w-[3rem]">
                  {lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                {ri === 0 && (
                  <td
                    className="p-1 text-gray-600 font-semibold text-xs writing-mode-vertical align-middle pr-1"
                    rowSpan={labels.length}
                    style={{ writingMode: "vertical-lr", transform: "rotate(180deg)" }}
                  >
                    Actual
                  </td>
                )}
                <td className="p-1 text-center text-gray-600 font-medium min-w-[3rem]">
                  {labels[ri]}
                </td>
                {row.map((val, ci) => (
                  <td
                    key={ci}
                    className={`p-1 text-center rounded min-w-[3rem] ${cellBg(ri, ci, val, rowTotals[ri])}`}
                  >
                    {val}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-class metrics table */}
      {per_class_metrics.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-1">
            Per-Class Metrics
          </p>
          <table className="text-xs w-full border-collapse" aria-label="Per-class precision recall F1">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left p-1 text-gray-500 font-medium">Class</th>
                <th className="text-center p-1 text-gray-500 font-medium">Precision</th>
                <th className="text-center p-1 text-gray-500 font-medium">Recall</th>
                <th className="text-center p-1 text-gray-500 font-medium">F1</th>
                <th className="text-center p-1 text-gray-500 font-medium">Support</th>
              </tr>
            </thead>
            <tbody>
              {per_class_metrics.map((m) => (
                <tr key={m.label} className="border-b border-gray-100">
                  <td className="p-1 font-medium text-gray-700">{m.label}</td>
                  <td className="p-1 text-center text-gray-700">{(m.precision * 100).toFixed(0)}%</td>
                  <td className="p-1 text-center text-gray-700">{(m.recall * 100).toFixed(0)}%</td>
                  <td className="p-1 text-center text-gray-700">{(m.f1 * 100).toFixed(0)}%</td>
                  <td className="p-1 text-center text-gray-500">{m.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Most confused pair callout */}
      {most_confused_pair && (
        <div className="bg-rose-50 border border-rose-200 rounded p-2 text-xs text-rose-700 mb-2">
          <span className="font-semibold">Most common mistake:</span> Actual &ldquo;{most_confused_pair.actual}&rdquo; predicted as &ldquo;{most_confused_pair.predicted}&rdquo; ({most_confused_pair.count} times)
        </div>
      )}

      <figcaption className="text-xs text-gray-500 mt-1">{summary}</figcaption>
    </figure>
  )
}
