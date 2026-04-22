"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { FairnessCheckResult } from "@/lib/types"

interface FairnessCheckCardProps {
  result: FairnessCheckResult
}

const STATUS_COLORS: Record<string, string> = {
  fair: "border-emerald-500/30",
  warning: "border-amber-500/30",
  biased: "border-rose-500/30",
  insufficient_data: "border-slate-300",
}

const STATUS_BADGE_CLASSES: Record<string, string> = {
  fair: "bg-emerald-100 text-emerald-800",
  warning: "bg-amber-100 text-amber-800",
  biased: "bg-rose-100 text-rose-800",
  insufficient_data: "bg-slate-100 text-slate-600",
}

const STATUS_LABELS: Record<string, string> = {
  fair: "Fair",
  warning: "Minor Disparity",
  biased: "Bias Detected",
  insufficient_data: "Insufficient Data",
}

const SPD_COLORS: Record<string, string> = {
  fair: "text-emerald-700",
  "slight disparity": "text-amber-700",
  "moderate disparity": "text-orange-700",
  "significant disparity": "text-rose-700",
}

const DIR_COLORS: Record<string, string> = {
  "passes 4/5ths rule": "text-emerald-700",
  borderline: "text-amber-700",
  "fails 4/5ths rule": "text-rose-700",
}

export function FairnessCheckCard({ result }: FairnessCheckCardProps) {
  const {
    overall_status,
    sensitive_col,
    target_col,
    algorithm,
    problem_type,
    per_group_metrics,
    spd,
    spd_label,
    dir: dir_val,
    dir_label,
    mae_disparity,
    summary,
  } = result

  const borderClass = STATUS_COLORS[overall_status] ?? "border-slate-300"
  const badgeClass =
    STATUS_BADGE_CLASSES[overall_status] ?? "bg-slate-100 text-slate-600"
  const statusLabel = STATUS_LABELS[overall_status] ?? overall_status

  const isClassification = problem_type === "classification"

  return (
    <figure aria-label={`Fairness analysis for ${sensitive_col ?? "sensitive column"}`} className="not-prose">
      <Card className={`${borderClass}`}>
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <span aria-hidden="true">⚖️</span> Fairness Check
            </CardTitle>
            <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${badgeClass}`}>
              {statusLabel}
            </span>
            {algorithm && (
              <Badge variant="outline" className="text-xs">
                {algorithm}
              </Badge>
            )}
            {problem_type && (
              <Badge variant="outline" className="text-xs capitalize">
                {problem_type}
              </Badge>
            )}
            {sensitive_col && (
              <Badge variant="outline" className="text-xs">
                by {sensitive_col}
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="pt-0 space-y-3">
          {overall_status === "insufficient_data" ? (
            <p className="text-sm text-muted-foreground">{summary}</p>
          ) : (
            <>
              {/* Fairness metrics row */}
              {isClassification && spd !== undefined && dir_val !== undefined && (
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-md border p-2">
                    <p className="text-xs text-muted-foreground mb-0.5">Statistical Parity Diff</p>
                    <p className={`font-mono text-base font-bold ${SPD_COLORS[spd_label ?? ""] ?? ""}`}>
                      {spd.toFixed(3)}
                    </p>
                    {spd_label && (
                      <p className="text-xs text-muted-foreground capitalize">{spd_label}</p>
                    )}
                  </div>
                  <div className="rounded-md border p-2">
                    <p className="text-xs text-muted-foreground mb-0.5">Disparate Impact Ratio</p>
                    <p className={`font-mono text-base font-bold ${DIR_COLORS[dir_label ?? ""] ?? ""}`}>
                      {dir_val.toFixed(3)}
                    </p>
                    {dir_label && (
                      <p className="text-xs text-muted-foreground">{dir_label}</p>
                    )}
                  </div>
                </div>
              )}

              {!isClassification && mae_disparity !== undefined && (
                <div className="rounded-md border p-2 text-sm">
                  <p className="text-xs text-muted-foreground mb-0.5">MAE Disparity Ratio</p>
                  <p className={`font-mono text-base font-bold ${
                    mae_disparity < 1.25
                      ? "text-emerald-700"
                      : mae_disparity < 1.5
                      ? "text-amber-700"
                      : "text-rose-700"
                  }`}>
                    {mae_disparity.toFixed(2)}×
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {mae_disparity < 1.25
                      ? "Error rates are consistent across groups"
                      : mae_disparity < 1.5
                      ? "Moderate error gap — monitor as data grows"
                      : "Large error gap — consider re-balancing training data"}
                  </p>
                </div>
              )}

              {/* Per-group table */}
              {per_group_metrics && per_group_metrics.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                    Per-Group Metrics
                  </p>
                  <table className="w-full text-xs border-collapse" aria-label="Per-group fairness metrics">
                    <thead>
                      <tr className="border-b text-muted-foreground">
                        <th className="text-left py-1 pr-2 font-medium">Group</th>
                        <th className="text-right py-1 pr-2 font-medium">Count</th>
                        {isClassification ? (
                          <>
                            <th className="text-right py-1 pr-2 font-medium">Pos. Rate</th>
                            <th className="text-right py-1 font-medium">Accuracy</th>
                          </>
                        ) : (
                          <th className="text-right py-1 font-medium">MAE</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {per_group_metrics.map((g, i) => (
                        <tr key={i} className="border-b last:border-0 hover:bg-muted/30">
                          <td className="py-1 pr-2 font-mono text-xs">{g.group}</td>
                          <td className="text-right py-1 pr-2 tabular-nums">{g.count}</td>
                          {isClassification ? (
                            <>
                              <td className="text-right py-1 pr-2 tabular-nums">
                                {g.positive_rate !== undefined
                                  ? `${(g.positive_rate * 100).toFixed(1)}%`
                                  : "—"}
                              </td>
                              <td className="text-right py-1 tabular-nums">
                                {g.accuracy !== undefined
                                  ? `${(g.accuracy * 100).toFixed(1)}%`
                                  : "—"}
                              </td>
                            </>
                          ) : (
                            <td className="text-right py-1 tabular-nums">
                              {g.mae !== undefined ? g.mae.toFixed(4) : "—"}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Plain-English guidance */}
              {overall_status !== "fair" && (
                <div
                  role="alert"
                  className={`text-xs rounded-md p-2 ${
                    overall_status === "biased"
                      ? "bg-rose-50 border border-rose-200 text-rose-800"
                      : "bg-amber-50 border border-amber-200 text-amber-800"
                  }`}
                >
                  {overall_status === "biased"
                    ? "Consider collecting more balanced training data or applying re-weighting techniques to reduce disparity."
                    : "Minor disparity detected. Monitor as more data accumulates."}
                </div>
              )}

              {/* Summary */}
              <p className="text-xs text-muted-foreground leading-relaxed">{summary}</p>

              {target_col && (
                <p className="text-xs text-muted-foreground">
                  Predicting:{" "}
                  <span className="font-mono bg-muted px-1 rounded">{target_col}</span>
                </p>
              )}
            </>
          )}
          <figcaption className="sr-only">
            Fairness analysis showing {overall_status} status
            {sensitive_col ? ` across groups in column ${sensitive_col}` : ""}.
          </figcaption>
        </CardContent>
      </Card>
    </figure>
  )
}
