"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type {
  ProductionInputDistributionResult,
  ProdInputFeature,
  ProdInputNumericFeature,
  ProdInputCategoricalFeature,
} from "@/lib/types"

interface ProductionInputDistributionCardProps {
  result: ProductionInputDistributionResult
}

function NumericFeatureRow({ feat }: { feat: ProdInputNumericFeature }) {
  const hasTrainingRange = feat.train_min !== null && feat.train_max !== null
  const hasOOR = feat.out_of_range_count > 0

  return (
    <div
      className={`rounded-md border p-3 ${hasOOR ? "border-amber-200 bg-amber-50" : "border-gray-200 bg-gray-50"}`}
      aria-label={`Feature ${feat.feature}: numeric`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">{feat.feature}</span>
        <div className="flex gap-1 items-center">
          <Badge variant="outline" className="text-xs text-gray-500">
            numeric
          </Badge>
          {hasOOR && (
            <Badge className="text-xs bg-amber-100 text-amber-800 border-amber-300">
              {feat.out_of_range_pct}% out of range
            </Badge>
          )}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs text-gray-600">
        <div>
          <span className="text-gray-400">min</span>
          <div className="font-mono font-medium">{feat.min.toLocaleString()}</div>
        </div>
        <div>
          <span className="text-gray-400">avg</span>
          <div className="font-mono font-medium">{feat.mean.toLocaleString()}</div>
        </div>
        <div>
          <span className="text-gray-400">max</span>
          <div className="font-mono font-medium">{feat.max.toLocaleString()}</div>
        </div>
      </div>
      {hasTrainingRange && (
        <div className="mt-2 text-xs text-gray-400">
          Training range: {feat.train_min?.toLocaleString()} – {feat.train_max?.toLocaleString()}
          {feat.out_of_range_count > 0 && (
            <span className="ml-1 text-amber-600">
              ({feat.out_of_range_count} outside)
            </span>
          )}
        </div>
      )}
    </div>
  )
}

function CategoricalFeatureRow({ feat }: { feat: ProdInputCategoricalFeature }) {
  const hasUnseen = feat.unseen_count > 0
  const maxPct = Math.max(...feat.top_categories.map((c) => c.pct), 1)

  return (
    <div
      className={`rounded-md border p-3 ${hasUnseen ? "border-rose-200 bg-rose-50" : "border-gray-200 bg-gray-50"}`}
      aria-label={`Feature ${feat.feature}: categorical`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">{feat.feature}</span>
        <div className="flex gap-1 items-center">
          <Badge variant="outline" className="text-xs text-gray-500">
            categorical · {feat.n_unique} unique
          </Badge>
          {hasUnseen && (
            <Badge className="text-xs bg-rose-100 text-rose-800 border-rose-300">
              {feat.unseen_pct}% unseen
            </Badge>
          )}
        </div>
      </div>
      <div className="space-y-1">
        {feat.top_categories.map((cat) => (
          <div key={cat.value} className="flex items-center gap-2 text-xs">
            <span className="w-24 truncate text-gray-600 font-mono">{cat.value}</span>
            <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-2 rounded-full bg-sky-400"
                style={{ width: `${(cat.pct / maxPct) * 100}%` }}
                aria-label={`${cat.value}: ${cat.pct}%`}
              />
            </div>
            <span className="w-10 text-right text-gray-500">{cat.pct}%</span>
          </div>
        ))}
      </div>
      {hasUnseen && (
        <div className="mt-2 text-xs text-rose-600">
          {feat.unseen_count} value{feat.unseen_count !== 1 ? "s" : ""} not seen during training
        </div>
      )}
    </div>
  )
}

function FeatureRow({ feat }: { feat: ProdInputFeature }) {
  if (feat.feature_type === "numeric") {
    return <NumericFeatureRow feat={feat as ProdInputNumericFeature} />
  }
  return <CategoricalFeatureRow feat={feat as ProdInputCategoricalFeature} />
}

export function ProductionInputDistributionCard({
  result,
}: ProductionInputDistributionCardProps) {
  const { sample_count, features, summary } = result
  const totalOOR = features.reduce((acc, f) => {
    if (f.feature_type === "numeric") return acc + (f as ProdInputNumericFeature).out_of_range_count
    return acc + (f as ProdInputCategoricalFeature).unseen_count
  }, 0)

  return (
    <Card
      className="border-sky-300 bg-sky-50"
      aria-label="Production input distribution"
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <span>📊</span>
          <span>Production Input Distribution</span>
        </CardTitle>
        <div className="flex flex-wrap gap-2 mt-1">
          <Badge className="bg-sky-100 text-sky-800 border-sky-300">
            {sample_count} prediction{sample_count !== 1 ? "s" : ""} analyzed
          </Badge>
          <Badge className="bg-gray-100 text-gray-700 border-gray-300">
            {features.length} feature{features.length !== 1 ? "s" : ""}
          </Badge>
          {totalOOR > 0 ? (
            <Badge className="bg-amber-100 text-amber-800 border-amber-300">
              {totalOOR} out-of-range values
            </Badge>
          ) : (
            features.length > 0 && (
              <Badge className="bg-emerald-100 text-emerald-800 border-emerald-300">
                All inputs in range
              </Badge>
            )
          )}
        </div>
      </CardHeader>
      <CardContent>
        {sample_count === 0 ? (
          <p className="text-sm text-gray-500 italic">{summary}</p>
        ) : (
          <>
            <p className="text-sm text-gray-600 mb-3">{summary}</p>
            {features.length === 0 ? (
              <p className="text-sm text-gray-400 italic">No feature data available.</p>
            ) : (
              <div className="space-y-2">
                {features.map((feat) => (
                  <FeatureRow key={feat.feature} feat={feat} />
                ))}
              </div>
            )}
          </>
        )}
        <figcaption className="text-xs text-gray-400 mt-3">
          Amber = numeric values outside training range · Rose = unseen category values
        </figcaption>
      </CardContent>
    </Card>
  )
}
