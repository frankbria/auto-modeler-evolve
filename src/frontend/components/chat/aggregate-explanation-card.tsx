"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { AggregateExplanationResult, AggregateExplanationFeature } from "@/lib/types"

interface AggregateExplanationCardProps {
  result: AggregateExplanationResult
}

function DirectionBadge({ label }: { label: string }) {
  if (label === "mostly positive") {
    return (
      <Badge className="text-xs bg-sky-100 text-sky-800 border-sky-300">
        ↑ mostly positive
      </Badge>
    )
  }
  if (label === "mostly negative") {
    return (
      <Badge className="text-xs bg-rose-100 text-rose-800 border-rose-300">
        ↓ mostly negative
      </Badge>
    )
  }
  return (
    <Badge className="text-xs bg-gray-100 text-gray-700 border-gray-300">
      ↔ mixed
    </Badge>
  )
}

function FeatureRow({
  feat,
  maxContrib,
}: {
  feat: AggregateExplanationFeature
  maxContrib: number
}) {
  const barWidth =
    maxContrib > 0 ? Math.round((feat.avg_abs_contribution / maxContrib) * 100) : 0
  const isPositive = feat.direction_label === "mostly positive"
  const isNegative = feat.direction_label === "mostly negative"
  const barColor = isPositive
    ? "bg-sky-400"
    : isNegative
      ? "bg-rose-400"
      : "bg-violet-400"

  return (
    <li
      role="listitem"
      aria-label={`${feat.feature}: ${feat.direction_label}, top driver in ${feat.top_driver_pct}% of predictions`}
      className="flex flex-col gap-1 py-2 border-b border-gray-100 last:border-0"
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-800">{feat.feature}</span>
        <div className="flex items-center gap-1">
          <DirectionBadge label={feat.direction_label} />
          {feat.top_driver_pct >= 30 && (
            <Badge className="text-xs bg-amber-100 text-amber-800 border-amber-300">
              top driver {feat.top_driver_pct}%
            </Badge>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 bg-gray-100 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${barColor}`}
            style={{ width: `${barWidth}%` }}
            aria-hidden="true"
          />
        </div>
        <span className="text-xs text-gray-500 w-16 text-right">
          {feat.positive_pct}% pos
        </span>
      </div>
    </li>
  )
}

export function AggregateExplanationCard({ result }: AggregateExplanationCardProps) {
  const { features, sample_count, summary } = result

  const maxContrib =
    features.length > 0
      ? Math.max(...features.map((f) => f.avg_abs_contribution))
      : 1

  return (
    <figure
      role="region"
      aria-label="Aggregate production explanation"
      className="my-2"
    >
      <Card className="border-violet-400/40 bg-violet-50/30">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <span aria-hidden="true">📊</span>
            Production Feature Influence
            <Badge variant="outline" className="ml-auto text-xs">
              {sample_count} predictions
            </Badge>
            <Badge variant="outline" className="text-xs">
              {features.length} features
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {features.length === 0 ? (
            <p className="text-sm text-gray-500 italic">
              No feature contributions found.
            </p>
          ) : (
            <>
              <ul
                role="list"
                aria-label="Feature influence in production predictions"
                className="divide-y divide-gray-100"
              >
                {features.slice(0, 10).map((feat) => (
                  <FeatureRow key={feat.feature} feat={feat} maxContrib={maxContrib} />
                ))}
              </ul>
              <p className="text-xs text-gray-500 italic mt-3">{summary}</p>
              <p className="text-xs text-gray-400 mt-1">
                Bar width = average absolute influence. Color = typical direction
                (↑ blue = pushes prediction up, ↓ rose = pushes down, ↔ violet = mixed).
              </p>
            </>
          )}
          <figcaption className="sr-only">
            Aggregate feature influence across {sample_count} production predictions.{" "}
            {summary}
          </figcaption>
        </CardContent>
      </Card>
    </figure>
  )
}
