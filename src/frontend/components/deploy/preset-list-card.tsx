"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { PresetListInfo } from "@/lib/types"

interface PresetListCardProps {
  preset_list: PresetListInfo
  /** Called when user clicks Load on a preset — fills feature values in the chat or elsewhere */
  onLoadPreset?: (featureValues: Record<string, string | number>) => void
}

export function PresetListCard({ preset_list, onLoadPreset }: PresetListCardProps) {
  return (
    <Card
      data-testid="preset-list-card"
      aria-label={`${preset_list.count} saved prediction presets`}
      className="border-indigo-300"
    >
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-base">📋</span>
          <CardTitle className="text-sm">Saved Prediction Presets</CardTitle>
          <Badge className="ml-auto bg-indigo-100 text-indigo-800 border-indigo-200 text-xs">
            {preset_list.count} {preset_list.count === 1 ? "preset" : "presets"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {preset_list.presets.length === 0 ? (
          <p className="text-muted-foreground text-xs">
            No presets saved yet. Say &ldquo;add a preset called [Name] with [feature=value, ...]&rdquo; to create one.
          </p>
        ) : (
          <div className="space-y-2">
            {preset_list.presets.map((preset) => (
              <div
                key={preset.id}
                className="flex items-start justify-between gap-2 rounded-md border px-3 py-2"
                data-testid="preset-list-row"
              >
                <div className="space-y-1 min-w-0">
                  <p className="font-medium text-foreground truncate">{preset.name}</p>
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(preset.feature_values).slice(0, 4).map(([key, val]) => (
                      <Badge
                        key={key}
                        variant="outline"
                        className="text-xs font-mono"
                      >
                        {key}={String(val)}
                      </Badge>
                    ))}
                    {Object.keys(preset.feature_values).length > 4 && (
                      <span className="text-xs text-muted-foreground self-center">
                        +{Object.keys(preset.feature_values).length - 4} more
                      </span>
                    )}
                  </div>
                </div>
                {onLoadPreset && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="shrink-0 h-7 text-xs"
                    onClick={() => onLoadPreset(preset.feature_values)}
                    data-testid="preset-load-button"
                  >
                    Load
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
        <p className="text-xs text-muted-foreground">
          Each preset appears as a quick-fill button on the shared prediction dashboard.
        </p>
      </CardContent>
    </Card>
  )
}
