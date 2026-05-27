"use client"

import { Badge } from "@/components/ui/badge"
import type { FeatureEngineeringImpactResult } from "@/lib/types"

interface FeatureEngineeringImpactCardProps {
  result: FeatureEngineeringImpactResult
}

export function FeatureEngineeringImpactCard({ result }: FeatureEngineeringImpactCardProps) {
  if (!result) return null

  const totalImp = result.groups.reduce((sum, g) => sum + g.total_importance, 0)
  const maxFeatureImp = Math.max(
    ...result.groups.flatMap((g) => g.features.map((f) => f.importance)),
    0.001
  )

  const groupColors: Record<string, { bar: string; badge: string; border: string }> = {
    Original: {
      bar: "bg-blue-400",
      badge: "bg-blue-100 text-blue-800 border-blue-200",
      border: "border-blue-200",
    },
    Engineered: {
      bar: "bg-violet-400",
      badge: "bg-violet-100 text-violet-800 border-violet-200",
      border: "border-violet-200",
    },
  }

  const isPositive = result.has_engineering && result.groups[1]?.total_importance > 0

  return (
    <div
      className="rounded-lg border border-violet-300 bg-violet-50 p-4 mt-2"
      data-testid="feature-engineering-impact-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-lg">⚗️</span>
        <span className="font-semibold text-sm">Feature Engineering Impact</span>
        <Badge variant="outline" className="bg-violet-100 text-violet-800 border-violet-200 text-xs">
          {result.algorithm}
        </Badge>
        {result.has_engineering ? (
          <Badge
            variant="outline"
            className={`text-xs ${isPositive ? "bg-emerald-100 text-emerald-700 border-emerald-200" : "bg-amber-100 text-amber-700 border-amber-200"}`}
          >
            {result.n_engineered} engineered feature{result.n_engineered !== 1 ? "s" : ""}
          </Badge>
        ) : (
          <Badge variant="outline" className="bg-gray-100 text-gray-600 border-gray-200 text-xs">
            No engineering applied
          </Badge>
        )}
      </div>

      {/* Verdict */}
      <p className="text-xs text-gray-700 mb-3 leading-relaxed">{result.verdict}</p>

      {/* Group summary bars */}
      {result.has_engineering && totalImp > 0 && (
        <div className="mb-4">
          <p className="text-xs font-medium text-gray-600 mb-2">Total importance by group</p>
          <div className="space-y-2">
            {result.groups.map((group) => {
              const pct = totalImp > 0 ? Math.round((group.total_importance / totalImp) * 100) : 0
              const colors = groupColors[group.name] ?? groupColors.Original
              return (
                <div key={group.name} className="flex items-center gap-2">
                  <span className={`text-xs font-medium w-20 ${colors.badge.split(" ")[1]}`}>
                    {group.name}
                  </span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div
                      className={`${colors.bar} h-2 rounded-full transition-all`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 w-10 text-right">{pct}%</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Per-group feature lists */}
      {result.groups.map((group) => {
        if (group.features.length === 0) return null
        const colors = groupColors[group.name] ?? groupColors.Original
        return (
          <div key={group.name} className={`rounded border ${colors.border} bg-white/60 p-3 mb-2`}>
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="outline" className={`${colors.badge} text-xs`}>
                {group.name}
              </Badge>
              <span className="text-xs text-gray-500">
                {group.features.length} feature{group.features.length !== 1 ? "s" : ""} ·{" "}
                {Math.round(group.total_importance * 100)}% of model importance
              </span>
            </div>
            <div className="space-y-1">
              {group.features.slice(0, 8).map((feat) => {
                const barPct = maxFeatureImp > 0 ? (feat.importance / maxFeatureImp) * 100 : 0
                return (
                  <div key={feat.name} className="flex items-center gap-2">
                    <span className="text-xs text-gray-600 w-40 truncate" title={feat.name}>
                      {feat.name}
                      {feat.source && (
                        <span className="text-gray-400 ml-1">← {feat.source}</span>
                      )}
                    </span>
                    <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                      <div
                        className={`${colors.bar} h-1.5 rounded-full`}
                        style={{ width: `${barPct}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-12 text-right">
                      {(feat.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
              {group.features.length > 8 && (
                <p className="text-xs text-gray-400 mt-1">
                  +{group.features.length - 8} more features
                </p>
              )}
            </div>
          </div>
        )
      })}

      <p className="text-xs text-gray-400 mt-2">
        Target: <span className="font-medium text-gray-600">{result.target_column}</span> ·{" "}
        {result.n_features} total features
      </p>
    </div>
  )
}
