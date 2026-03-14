"use client"

import { useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  Pie,
  PieChart,
  Cell,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts"
import type { ChartSpec } from "@/lib/types"

const COLORS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b",
  "#10b981", "#3b82f6", "#ef4444", "#f97316",
]

interface ChartMessageProps {
  spec: ChartSpec
}

export function ChartMessage({ spec }: ChartMessageProps) {
  const { chart_type, title, data, x_key, y_keys, x_label, y_label } = spec

  // Heatmap uses its own layout (not Recharts)
  if (chart_type === "heatmap") {
    return (
      <div className="mt-2 rounded-lg border bg-card p-3">
        {title && (
          <p className="mb-2 text-xs font-semibold text-muted-foreground">{title}</p>
        )}
        <HeatmapChart data={data} columns={y_keys} />
      </div>
    )
  }

  // Scatter uses an interactive component with click-to-highlight
  if (chart_type === "scatter") {
    return (
      <div className="mt-2 rounded-lg border bg-card p-3">
        {title && (
          <p className="mb-2 text-xs font-semibold text-muted-foreground">{title}</p>
        )}
        <InteractiveScatterChart data={data} xLabel={x_label} yLabel={y_label} />
      </div>
    )
  }

  return (
    <div className="mt-2 rounded-lg border bg-card p-3">
      {title && (
        <p className="mb-2 text-xs font-semibold text-muted-foreground">
          {title}
        </p>
      )}
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {renderChart(chart_type, data, x_key, y_keys, x_label, y_label)}
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Heatmap renderer (correlation matrix)
// ---------------------------------------------------------------------------

function corrColor(value: number | null | undefined): string {
  if (value === null || value === undefined) return "hsl(var(--muted))"
  // Map [-1, 1] to color: negative=red, zero=white/gray, positive=blue
  const clamped = Math.max(-1, Math.min(1, value))
  if (clamped >= 0) {
    // 0 → near-white, 1 → blue
    const intensity = Math.round(clamped * 80)
    return `hsl(220 80% ${100 - intensity}%)`
  } else {
    // 0 → near-white, -1 → red
    const intensity = Math.round(-clamped * 80)
    return `hsl(0 80% ${100 - intensity}%)`
  }
}

function textColor(value: number | null | undefined): string {
  if (value === null || value === undefined) return "hsl(var(--muted-foreground))"
  return Math.abs(value ?? 0) > 0.5 ? "white" : "hsl(var(--foreground))"
}

function HeatmapChart({
  data,
  columns,
}: {
  data: Record<string, unknown>[]
  columns: string[]
}) {
  const cellSize = Math.max(28, Math.min(48, Math.floor(240 / (columns.length + 1))))
  const labelWidth = 64

  return (
    <div className="overflow-x-auto">
      <div style={{ display: "inline-block", minWidth: labelWidth + columns.length * cellSize }}>
        {/* Column header row */}
        <div style={{ display: "flex", marginLeft: labelWidth }}>
          {columns.map((col) => (
            <div
              key={col}
              style={{ width: cellSize, fontSize: 9, textAlign: "center", overflow: "hidden" }}
              className="text-muted-foreground font-medium px-0.5 truncate"
              title={col}
            >
              {col.length > 6 ? col.slice(0, 6) + "…" : col}
            </div>
          ))}
        </div>
        {/* Data rows */}
        {data.map((row) => {
          const rowLabel = String(row.row ?? "")
          return (
            <div key={rowLabel} style={{ display: "flex", alignItems: "center" }}>
              {/* Row label */}
              <div
                style={{ width: labelWidth, fontSize: 9, textAlign: "right", paddingRight: 4, flexShrink: 0 }}
                className="text-muted-foreground font-medium truncate"
                title={rowLabel}
              >
                {rowLabel.length > 8 ? rowLabel.slice(0, 8) + "…" : rowLabel}
              </div>
              {/* Cells */}
              {columns.map((col) => {
                const val = row[col] as number | null | undefined
                return (
                  <div
                    key={col}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: corrColor(val),
                      color: textColor(val),
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 9,
                      fontWeight: 500,
                      border: "1px solid hsl(var(--border))",
                      cursor: "default",
                    }}
                    title={`${rowLabel} × ${col}: ${val != null ? val.toFixed(3) : "N/A"}`}
                  >
                    {val != null ? val.toFixed(2) : ""}
                  </div>
                )
              })}
            </div>
          )
        })}
        {/* Legend */}
        <div className="mt-1 flex items-center gap-1" style={{ marginLeft: labelWidth }}>
          <span className="text-[8px] text-muted-foreground">−1</span>
          <div style={{ background: "linear-gradient(to right, hsl(0 80% 20%), hsl(0 80% 100%), hsl(220 80% 20%))", height: 6, flex: 1, borderRadius: 3 }} />
          <span className="text-[8px] text-muted-foreground">+1</span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Interactive Scatter Chart — click to highlight a point
// ---------------------------------------------------------------------------

interface ScatterPoint {
  x: number
  y: number
  label?: string
  [key: string]: unknown
}

interface ActivePoint {
  x: number
  y: number
  label?: string
}

function InteractiveScatterChart({
  data,
  xLabel,
  yLabel,
}: {
  data: Record<string, unknown>[]
  xLabel: string
  yLabel: string
}) {
  const [activePoint, setActivePoint] = useState<ActivePoint | null>(null)
  const axisStyle = { fontSize: 10, fill: "hsl(var(--muted-foreground))" }
  const tooltipStyle = {
    contentStyle: {
      fontSize: 11,
      backgroundColor: "hsl(var(--card))",
      border: "1px solid hsl(var(--border))",
      borderRadius: 6,
    },
  }

  const points = data as ScatterPoint[]

  function handleClick(point: ScatterPoint | null) {
    if (!point) return
    if (activePoint && activePoint.x === point.x && activePoint.y === point.y) {
      setActivePoint(null)  // deselect on second click
    } else {
      setActivePoint({ x: point.x, y: point.y, label: point.label })
    }
  }

  // Split data into highlighted and non-highlighted for rendering
  const highlighted = activePoint
    ? points.filter((p) => p.x === activePoint.x && p.y === activePoint.y)
    : []
  const normal = activePoint
    ? points.filter((p) => !(p.x === activePoint.x && p.y === activePoint.y))
    : points

  return (
    <div className="h-52 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="x"
            type="number"
            name={xLabel}
            tick={axisStyle}
            label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, style: axisStyle } : undefined}
          />
          <YAxis
            dataKey="y"
            type="number"
            name={yLabel}
            tick={axisStyle}
            label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", style: axisStyle } : undefined}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            {...tooltipStyle}
          />
          {/* Reference lines at selected point */}
          {activePoint && (
            <>
              <ReferenceLine
                x={activePoint.x}
                stroke={COLORS[2]}
                strokeDasharray="4 4"
                strokeOpacity={0.7}
              />
              <ReferenceLine
                y={activePoint.y}
                stroke={COLORS[2]}
                strokeDasharray="4 4"
                strokeOpacity={0.7}
              />
            </>
          )}
          {/* Regular points */}
          <Scatter
            data={normal}
            fill={COLORS[0]}
            fillOpacity={activePoint ? 0.35 : 0.8}
            onClick={(point) => handleClick(point as unknown as ScatterPoint)}
            style={{ cursor: "pointer" }}
          />
          {/* Highlighted point */}
          {highlighted.length > 0 && (
            <Scatter
              data={highlighted}
              fill={COLORS[2]}
              fillOpacity={1}
              onClick={() => setActivePoint(null)}
              style={{ cursor: "pointer" }}
              r={6}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
      {activePoint && (
        <p className="mt-0.5 text-center text-[10px] text-muted-foreground">
          Selected: ({xLabel || "x"} = {activePoint.x}, {yLabel || "y"} = {activePoint.y})
          {activePoint.label && ` — ${activePoint.label}`}
          <button
            onClick={() => setActivePoint(null)}
            className="ml-2 text-primary hover:underline"
          >
            Clear
          </button>
        </p>
      )}
    </div>
  )
}


function renderChart(
  chartType: ChartSpec["chart_type"],
  data: Record<string, unknown>[],
  xKey: string,
  yKeys: string[],
  xLabel: string,
  yLabel: string,
): React.ReactElement {
  const axisStyle = { fontSize: 10, fill: "hsl(var(--muted-foreground))" }
  const tooltipStyle = {
    contentStyle: {
      fontSize: 11,
      backgroundColor: "hsl(var(--card))",
      border: "1px solid hsl(var(--border))",
      borderRadius: 6,
    },
  }

  switch (chartType) {
    case "bar":
    case "histogram":
      return (
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey={xKey}
            tick={axisStyle}
            label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, style: axisStyle } : undefined}
          />
          <YAxis
            tick={axisStyle}
            label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", style: axisStyle } : undefined}
          />
          <Tooltip {...tooltipStyle} />
          {yKeys.map((key, i) => (
            <Bar key={key} dataKey={key} fill={COLORS[i % COLORS.length]} radius={[2, 2, 0, 0]} />
          ))}
        </BarChart>
      )

    case "line":
      return (
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey={xKey} tick={axisStyle} />
          <YAxis tick={axisStyle} />
          <Tooltip {...tooltipStyle} />
          {yKeys.length > 1 && <Legend wrapperStyle={{ fontSize: 10 }} />}
          {yKeys.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={data.length <= 30}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      )

    case "pie":
      return (
        <PieChart>
          <Pie
            data={data}
            dataKey={yKeys[0] ?? "value"}
            nameKey={xKey}
            cx="50%"
            cy="50%"
            outerRadius={70}
            innerRadius={30}
            label={({ name, percent }: { name?: string; percent?: number }) =>
              `${name ?? ""} ${((percent ?? 0) * 100).toFixed(0)}%`
            }
            labelLine={false}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip {...tooltipStyle} />
        </PieChart>
      )

    default:
      return <div className="text-xs text-muted-foreground">Chart unavailable</div>
  }
}
