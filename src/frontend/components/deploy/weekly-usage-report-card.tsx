"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import type {
  WeeklyUsageReportResult,
  TopInputPattern,
  WeeklyUsageDayEntry,
} from "@/lib/types"

interface WeeklyUsageReportCardProps {
  result: WeeklyUsageReportResult
}

function TrendBadge({ trend, changePct }: { trend: string; changePct: number | null }) {
  if (changePct === null) {
    return (
      <Badge variant="outline" className="text-xs text-gray-500">
        No prior data
      </Badge>
    )
  }
  if (trend === "up") {
    return (
      <Badge className="text-xs bg-emerald-100 text-emerald-800 border-emerald-300">
        ↑ {changePct}% vs last week
      </Badge>
    )
  }
  if (trend === "down") {
    return (
      <Badge className="text-xs bg-rose-100 text-rose-800 border-rose-300">
        ↓ {Math.abs(changePct)}% vs last week
      </Badge>
    )
  }
  return (
    <Badge variant="outline" className="text-xs text-sky-700 border-sky-300">
      → Stable
    </Badge>
  )
}

function DayBar({ entry, maxCount }: { entry: WeeklyUsageDayEntry; maxCount: number }) {
  const [, m, d] = entry.date.split("-")
  const label = `${m}/${d}`
  const pct = maxCount > 0 ? Math.round((entry.count / maxCount) * 100) : 0
  return (
    <div className="flex flex-col items-center gap-1 flex-1">
      <span className="text-[10px] font-medium text-sky-700">{entry.count > 0 ? entry.count : ""}</span>
      <div className="w-full bg-gray-100 rounded-sm" style={{ height: "48px" }}>
        <div
          className="w-full bg-sky-400 rounded-sm transition-all"
          style={{ height: `${pct}%`, marginTop: `${100 - pct}%` }}
          aria-label={`${label}: ${entry.count} predictions`}
        />
      </div>
      <span className="text-[9px] text-gray-500">{label}</span>
    </div>
  )
}

function PatternRow({ pattern }: { pattern: TopInputPattern }) {
  return (
    <div className="flex flex-wrap items-start gap-1.5 py-1 border-b border-gray-100 last:border-0">
      <span className="text-xs font-medium text-gray-700 min-w-[80px]">
        {pattern.feature}
      </span>
      <div className="flex flex-wrap gap-1">
        {pattern.top_values.map((tv) => (
          <span
            key={tv.value}
            className="rounded bg-sky-50 border border-sky-200 px-1.5 py-0.5 text-[10px] text-sky-700"
          >
            {tv.value}{" "}
            <span className="text-sky-400">
              ({tv.count}×)
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}

export function WeeklyUsageReportCard({ result }: WeeklyUsageReportCardProps) {
  const {
    this_week_count,
    last_week_count,
    change_pct,
    trend,
    by_day,
    top_input_patterns,
    summary,
  } = result

  const maxCount = Math.max(...by_day.map((d) => d.count), 1)

  return (
    <figure aria-label="Weekly usage report" className="not-prose">
      <Card className="border-sky-400/40" data-testid="weekly-usage-report-card">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2 flex-wrap">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <span aria-hidden="true">📈</span> Weekly Usage Report
            </CardTitle>
            <TrendBadge trend={trend} changePct={change_pct} />
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* Summary */}
          <p className="text-xs text-muted-foreground" data-testid="weekly-summary">
            {summary}
          </p>

          {/* Week-over-week stats */}
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded bg-sky-50 border border-sky-200 px-3 py-2 text-xs text-center">
              <p className="text-muted-foreground">This week</p>
              <p
                className="font-semibold text-sky-800 text-lg"
                data-testid="this-week-count"
              >
                {this_week_count.toLocaleString()}
              </p>
            </div>
            <div className="rounded bg-gray-50 border border-gray-200 px-3 py-2 text-xs text-center">
              <p className="text-muted-foreground">Last week</p>
              <p
                className="font-semibold text-gray-700 text-lg"
                data-testid="last-week-count"
              >
                {last_week_count.toLocaleString()}
              </p>
            </div>
          </div>

          {/* 7-day bar chart */}
          <div>
            <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Day-by-day (this week)
            </p>
            <div
              className="flex items-end gap-1"
              role="img"
              aria-label="Bar chart: predictions per day this week"
            >
              {by_day.map((entry) => (
                <DayBar key={entry.date} entry={entry} maxCount={maxCount} />
              ))}
            </div>
          </div>

          {/* Top input patterns */}
          {top_input_patterns.length > 0 && (
            <div>
              <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-2">
                Top input patterns
              </p>
              <div className="space-y-0.5" data-testid="top-input-patterns">
                {top_input_patterns.map((pattern) => (
                  <PatternRow key={pattern.feature} pattern={pattern} />
                ))}
              </div>
            </div>
          )}

          {top_input_patterns.length === 0 && (
            <p className="text-xs text-gray-400 italic">
              No categorical input patterns found in this week's predictions.
            </p>
          )}
        </CardContent>
      </Card>
      <figcaption className="sr-only">{summary}</figcaption>
    </figure>
  )
}
