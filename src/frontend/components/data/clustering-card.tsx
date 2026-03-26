"use client"

import type { ClusteringResult, ClusterProfile } from "@/lib/types"

interface ClusteringCardProps {
  result: ClusteringResult
}

const CLUSTER_COLORS = [
  "bg-violet-500",
  "bg-blue-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-cyan-500",
  "bg-orange-500",
  "bg-pink-500",
]

const CLUSTER_TEXT_COLORS = [
  "text-violet-700",
  "text-blue-700",
  "text-emerald-700",
  "text-amber-700",
  "text-rose-700",
  "text-cyan-700",
  "text-orange-700",
  "text-pink-700",
]

const CLUSTER_BG_COLORS = [
  "bg-violet-50 border-violet-200",
  "bg-blue-50 border-blue-200",
  "bg-emerald-50 border-emerald-200",
  "bg-amber-50 border-amber-200",
  "bg-rose-50 border-rose-200",
  "bg-cyan-50 border-cyan-200",
  "bg-orange-50 border-orange-200",
  "bg-pink-50 border-pink-200",
]

function SizeBar({ pct, colorClass }: { pct: number; colorClass: string }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
      <div
        className={`h-full rounded-full transition-all ${colorClass}`}
        style={{ width: `${Math.max(pct, 2)}%` }}
      />
    </div>
  )
}

function ClusterRow({ cluster, index }: { cluster: ClusterProfile; index: number }) {
  const barColor = CLUSTER_COLORS[index % CLUSTER_COLORS.length]
  const textColor = CLUSTER_TEXT_COLORS[index % CLUSTER_TEXT_COLORS.length]
  const bgColor = CLUSTER_BG_COLORS[index % CLUSTER_BG_COLORS.length]

  return (
    <div className={`rounded-md border p-3 ${bgColor}`}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className={`text-sm font-semibold ${textColor}`}>
          Group {cluster.cluster_id + 1}
        </span>
        <span className="text-xs text-muted-foreground">
          {cluster.size} rows ({cluster.size_pct}%)
        </span>
      </div>
      <SizeBar pct={cluster.size_pct} colorClass={barColor} />
      {cluster.distinguishing.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {cluster.distinguishing.slice(0, 3).map((d) => (
            <span
              key={d.feature}
              className="inline-flex items-center gap-0.5 rounded border bg-background/70 px-1.5 py-0.5 text-[10px] font-medium"
            >
              {d.direction === "above" ? "↑" : "↓"}{" "}
              {d.feature.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}
      <p className="mt-2 text-xs text-muted-foreground">{cluster.description}</p>
    </div>
  )
}

export function ClusteringCard({ result }: ClusteringCardProps) {
  const { n_clusters, features_used, auto_k, rows_clustered, clusters, summary } = result

  return (
    <div className="my-2 w-full max-w-2xl rounded-lg border border-violet-300 bg-violet-50/40 p-4">
      {/* Header */}
      <div className="mb-3 flex items-center gap-2">
        <span className="text-base font-semibold text-violet-800">
          Customer Segmentation
        </span>
        <span className="rounded border border-violet-200 bg-violet-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-violet-700">
          {n_clusters} groups {auto_k ? "· auto" : "· manual"}
        </span>
      </div>

      {/* Summary */}
      <p className="mb-3 text-sm text-muted-foreground">{summary}</p>

      {/* Features used */}
      <div className="mb-3 flex flex-wrap gap-1">
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Clustered on:
        </span>
        {features_used.map((f) => (
          <span
            key={f}
            className="rounded border bg-background/80 px-1.5 py-0.5 text-[10px] font-medium text-foreground"
          >
            {f.replace(/_/g, " ")}
          </span>
        ))}
      </div>

      {/* Cluster rows */}
      <div className="grid gap-2 sm:grid-cols-2">
        {clusters.map((cluster, i) => (
          <ClusterRow key={cluster.cluster_id} cluster={cluster} index={i} />
        ))}
      </div>

      {/* Footer */}
      <p className="mt-3 text-[10px] text-muted-foreground">
        {rows_clustered} rows clustered · K-means with{" "}
        {auto_k ? "auto-selected" : "specified"} k={n_clusters}
      </p>
    </div>
  )
}
