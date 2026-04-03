"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { FeatureSelectionResult } from "@/lib/types"

interface FeatureSelectionCardProps {
  result?: FeatureSelectionResult
  projectId?: string
  // ModelTrainingPanel variant — shows checkboxes for retrain
  data?: FeatureSelectionResult
  excludedFeatures?: string[]
  onExcludedFeaturesChange?: (features: string[]) => void
}

/**
 * Renders feature importance results as a ranked list.
 *
 * Two modes:
 * 1. Chat card (result prop) — read-only, amber border
 * 2. Panel card (data prop) — interactive checkboxes to exclude features before retraining
 */
export function FeatureSelectionCard({
  result,
  data,
  excludedFeatures = [],
  onExcludedFeaturesChange,
}: FeatureSelectionCardProps) {
  const fs = data ?? result
  if (!fs) return null

  const isPanel = !!data

  const maxImp = Math.max(
    ...fs.feature_importances
      .map((f) => f.importance ?? 0)
      .filter((v) => v > 0),
    0.001
  )

  const toggleExclude = (name: string) => {
    if (!onExcludedFeaturesChange) return
    if (excludedFeatures.includes(name)) {
      onExcludedFeaturesChange(excludedFeatures.filter((n) => n !== name))
    } else {
      onExcludedFeaturesChange([...excludedFeatures, name])
    }
  }

  const methodLabel =
    fs.method === "feature_importances"
      ? "feature importance"
      : fs.method === "coefficients"
      ? "coefficient magnitude"
      : "importance"

  return (
    <div
      className={`rounded-lg border p-4 mt-2 ${
        isPanel
          ? "border-amber-300 bg-amber-50"
          : "border-amber-300 bg-amber-50"
      }`}
      data-testid="feature-selection-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span aria-hidden="true" className="text-lg">🎯</span>
        <span className="font-semibold text-sm">Feature Importance</span>
        <Badge variant="outline" className="bg-amber-100 text-amber-800 border-amber-200 text-xs">
          {fs.algorithm}
        </Badge>
        {fs.n_weak > 0 && (
          <Badge variant="outline" className="bg-rose-100 text-rose-700 border-rose-200 text-xs">
            {fs.n_weak} weak {fs.n_weak === 1 ? "feature" : "features"}
          </Badge>
        )}
        {fs.n_weak === 0 && fs.has_importances && (
          <Badge variant="outline" className="bg-emerald-100 text-emerald-700 border-emerald-200 text-xs">
            All features contributing
          </Badge>
        )}
      </div>

      <p className="text-xs text-foreground/80 mb-3">{fs.explanation}</p>

      {/* Feature importance bars */}
      {fs.has_importances && (
        <div className="space-y-1.5 mb-3">
          {fs.feature_importances.map((feat) => {
            const imp = feat.importance ?? 0
            const barWidth = maxImp > 0 ? Math.round((imp / maxImp) * 100) : 0
            const isWeak = feat.is_weak
            const isExcluded = excludedFeatures.includes(feat.name)

            return (
              <div key={feat.name} className="flex items-center gap-2">
                {isPanel && (
                  <input
                    type="checkbox"
                    id={`exclude-${feat.name}`}
                    checked={isExcluded}
                    onChange={() => toggleExclude(feat.name)}
                    aria-label={`Exclude ${feat.name} from retraining`}
                    className="h-3 w-3 cursor-pointer accent-amber-600"
                  />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span
                      className={`text-xs font-mono truncate ${
                        isWeak ? "text-rose-700" : "text-foreground"
                      } ${isExcluded ? "line-through opacity-50" : ""}`}
                    >
                      {feat.name}
                    </span>
                    {isWeak && (
                      <span
                        aria-label="weak feature"
                        className="text-[10px] text-rose-600 font-medium"
                      >
                        ↓ weak
                      </span>
                    )}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      #{feat.rank}
                    </span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        isWeak ? "bg-rose-400" : "bg-amber-500"
                      } ${isExcluded ? "opacity-30" : ""}`}
                      style={{ width: `${barWidth}%` }}
                      aria-label={`${feat.name} importance: ${(imp * 100).toFixed(1)}%`}
                    />
                  </div>
                </div>
                <span className="text-[10px] text-muted-foreground w-10 text-right shrink-0">
                  {(imp * 100).toFixed(1)}%
                </span>
              </div>
            )
          })}
        </div>
      )}

      {/* Method note */}
      <p className="text-[10px] text-muted-foreground mb-2">
        Based on {methodLabel} from {fs.algorithm}.
        {fs.n_weak > 0 &&
          ` Weak features are below the 20th percentile threshold (${((fs.threshold ?? 0) * 100).toFixed(2)}%).`}
      </p>

      {/* Panel-mode: exclude controls */}
      {isPanel && onExcludedFeaturesChange && (
        <div className="pt-2 border-t border-amber-200">
          {excludedFeatures.length > 0 ? (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-amber-800 font-medium">
                {excludedFeatures.length} feature{excludedFeatures.length !== 1 ? "s" : ""} will be excluded on next retrain
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs h-5 px-2 text-muted-foreground hover:text-foreground"
                onClick={() => onExcludedFeaturesChange([])}
              >
                Clear
              </Button>
            </div>
          ) : fs.n_weak > 0 ? (
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7 border-amber-300 text-amber-800 hover:bg-amber-100"
              onClick={() => onExcludedFeaturesChange(fs.weak_features)}
            >
              Exclude {fs.n_weak} weak {fs.n_weak === 1 ? "feature" : "features"} on retrain
            </Button>
          ) : (
            <p className="text-xs text-muted-foreground">
              Check boxes above to exclude features from the next training run.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
