"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { DataDictionary, ColumnDescription, ColumnSemanticType } from "@/lib/types"

// ---------------------------------------------------------------------------
// Column type badge
// ---------------------------------------------------------------------------

const TYPE_CONFIG: Record<
  ColumnSemanticType,
  { label: string; className: string }
> = {
  metric:    { label: "Metric",    className: "bg-blue-100 text-blue-800 border-blue-200" },
  dimension: { label: "Dimension", className: "bg-purple-100 text-purple-800 border-purple-200" },
  date:      { label: "Date",      className: "bg-green-100 text-green-800 border-green-200" },
  id:        { label: "ID",        className: "bg-gray-100 text-gray-600 border-gray-200" },
  flag:      { label: "Flag",      className: "bg-yellow-100 text-yellow-800 border-yellow-200" },
  text:      { label: "Text",      className: "bg-orange-100 text-orange-800 border-orange-200" },
  unknown:   { label: "?",         className: "bg-gray-100 text-gray-500 border-gray-200" },
}

function ColTypeBadge({ type }: { type: ColumnSemanticType }) {
  const cfg = TYPE_CONFIG[type] ?? TYPE_CONFIG.unknown
  return (
    <Badge className={`${cfg.className} text-xs font-medium shrink-0`}>
      {cfg.label}
    </Badge>
  )
}

// ---------------------------------------------------------------------------
// Single column row
// ---------------------------------------------------------------------------

function ColumnRow({ col }: { col: ColumnDescription }) {
  return (
    <div className="flex items-start gap-3 py-2 border-b last:border-0">
      <div className="w-32 shrink-0">
        <p className="text-xs font-mono font-semibold text-gray-800 truncate" title={col.name}>
          {col.name}
        </p>
        <p className="text-xs text-gray-400">{col.dtype}</p>
      </div>
      <ColTypeBadge type={col.col_type} />
      <p className="text-xs text-gray-600 flex-1 leading-relaxed">
        {col.description}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// DictionaryCard props + main component
// ---------------------------------------------------------------------------

interface DictionaryCardProps {
  datasetId: string
  initialData?: DataDictionary | null
}

export function DictionaryCard({ datasetId, initialData }: DictionaryCardProps) {
  const [data, setData] = useState<DataDictionary | null>(initialData ?? null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)

  const PREVIEW_LIMIT = 8

  async function loadDictionary() {
    setLoading(true)
    setError(null)
    try {
      const result = await api.data.getDictionary(datasetId)
      setData(result)
    } catch {
      setError("Failed to load data dictionary.")
    } finally {
      setLoading(false)
    }
  }

  async function generateDescriptions() {
    setLoading(true)
    setError(null)
    try {
      const result = await api.data.generateDictionary(datasetId)
      setData(result)
    } catch {
      setError("Failed to generate descriptions.")
    } finally {
      setLoading(false)
    }
  }

  const columns = data?.columns ?? []
  const displayed = showAll ? columns : columns.slice(0, PREVIEW_LIMIT)
  const hasMore = columns.length > PREVIEW_LIMIT

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Data Dictionary</CardTitle>
          <div className="flex items-center gap-2">
            {data && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs px-2"
                onClick={generateDescriptions}
                disabled={loading}
              >
                {loading ? "Generating…" : "Regenerate"}
              </Button>
            )}
          </div>
        </div>
        {data && (
          <p className="text-xs text-gray-500 mt-0.5">
            {columns.length} column{columns.length !== 1 ? "s" : ""} · {data.filename}
            {data.generated && (
              <span className="ml-1 text-blue-600">· AI descriptions</span>
            )}
          </p>
        )}
      </CardHeader>

      <CardContent className="pt-0">
        {error && (
          <p className="text-xs text-red-600 mb-2">{error}</p>
        )}

        {!data && !loading && (
          <div className="flex flex-col items-center gap-2 py-4">
            <p className="text-xs text-gray-500 text-center">
              Generate plain-English descriptions for each column — great for understanding
              inherited data or sharing with stakeholders.
            </p>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" className="text-xs" onClick={loadDictionary}>
                Quick summary
              </Button>
              <Button size="sm" className="text-xs" onClick={generateDescriptions}>
                AI descriptions
              </Button>
            </div>
          </div>
        )}

        {loading && (
          <p className="text-xs text-gray-500 py-4 text-center">Generating descriptions…</p>
        )}

        {data && !loading && (
          <>
            {/* Legend */}
            <div className="flex flex-wrap gap-1.5 mb-3">
              {(Object.keys(TYPE_CONFIG) as ColumnSemanticType[])
                .filter((t) => t !== "unknown" && columns.some((c) => c.col_type === t))
                .map((t) => (
                  <ColTypeBadge key={t} type={t} />
                ))}
            </div>

            {/* Column rows */}
            <div id="dictionary-columns-list">
              {displayed.map((col) => (
                <ColumnRow key={col.name} col={col} />
              ))}
            </div>

            {hasMore && (
              <button
                className="mt-2 text-xs text-blue-600 hover:underline"
                onClick={() => setShowAll((v) => !v)}
                aria-expanded={showAll}
                aria-controls="dictionary-columns-list"
              >
                {showAll
                  ? "Show less"
                  : `Show ${columns.length - PREVIEW_LIMIT} more column${columns.length - PREVIEW_LIMIT !== 1 ? "s" : ""}`}
              </button>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
