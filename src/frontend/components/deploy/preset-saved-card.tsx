"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { PresetSavedInfo } from "@/lib/types"

interface PresetSavedCardProps {
  preset: PresetSavedInfo
}

export function PresetSavedCard({ preset }: PresetSavedCardProps) {
  return (
    <Card
      data-testid="preset-saved-card"
      aria-label={`Prediction preset saved: ${preset.name}`}
      className="border-emerald-300"
    >
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-base">🎯</span>
          <CardTitle className="text-sm">Prediction Preset Saved</CardTitle>
          <Badge className="ml-auto bg-emerald-100 text-emerald-800 border-emerald-200 text-xs">
            {preset.feature_count} {preset.feature_count === 1 ? "feature" : "features"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <p className="font-medium text-foreground">&ldquo;{preset.name}&rdquo;</p>
        <div className="flex flex-wrap gap-1">
          {Object.entries(preset.feature_values).map(([key, val]) => (
            <Badge
              key={key}
              variant="outline"
              className="text-xs font-mono"
              data-testid="preset-feature-badge"
            >
              {key}={String(val)}
            </Badge>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          This preset now appears as a quick-fill button on the shared prediction dashboard.
          VPs and colleagues can click it to instantly load these values.
        </p>
      </CardContent>
    </Card>
  )
}
