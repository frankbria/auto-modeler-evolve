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

  // Box plot uses SVG renderer (Recharts has no native box plot)
  if (chart_type === "boxplot") {
    return (
      <div className="mt-2 rounded-lg border bg-card p-3">
        {title && (
          <p className="mb-2 text-xs font-semibold text-muted-foreground">{title}</p>
        )}
        <BoxPlotChart data={data} xLabel={x_label} yLabel={y_label} />
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

interface SelectedCell {
  row: string
  col: string
  value: number | null
}

function HeatmapChart({
  data,
  columns,
}: {
  data: Record<string, unknown>[]
  columns: string[]
}) {
  const [selected, setSelected] = useState<SelectedCell | null>(null)
  const cellSize = Math.max(28, Math.min(48, Math.floor(240 / (columns.length + 1))))
  const labelWidth = 64

  function handleCellClick(rowLabel: string, col: string, val: number | null | undefined) {
    const value = val ?? null
    if (selected && selected.row === rowLabel && selected.col === col) {
      setSelected(null)
    } else {
      setSelected({ row: rowLabel, col, value })
    }
  }

  return (
    <div className="overflow-x-auto">
      <div style={{ display: "inline-block", minWidth: labelWidth + columns.length * cellSize }}>
        {/* Column header row */}
        <div style={{ display: "flex", marginLeft: labelWidth }}>
          {columns.map((col) => (
            <div
              key={col}
              style={{ width: cellSize, fontSize: 9, textAlign: "center", overflow: "hidden" }}
              className={`font-medium px-0.5 truncate ${selected?.col === col ? "text-primary font-bold" : "text-muted-foreground"}`}
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
                className={`font-medium truncate ${selected?.row === rowLabel ? "text-primary font-bold" : "text-muted-foreground"}`}
                title={rowLabel}
              >
                {rowLabel.length > 8 ? rowLabel.slice(0, 8) + "…" : rowLabel}
              </div>
              {/* Cells */}
              {columns.map((col) => {
                const val = row[col] as number | null | undefined
                const isSelected = selected?.row === rowLabel && selected?.col === col
                return (
                  <div
                    key={col}
                    role="button"
                    tabIndex={0}
                    onClick={() => handleCellClick(rowLabel, col, val)}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && handleCellClick(rowLabel, col, val)}
                    className="focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1"
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
                      border: isSelected
                        ? "2px solid hsl(var(--primary))"
                        : "1px solid hsl(var(--border))",
                      cursor: "pointer",
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
        {/* Selected cell tooltip */}
        {selected && (
          <div
            className="mt-1.5 flex items-center gap-1.5 rounded border bg-card px-2 py-1"
            style={{ marginLeft: labelWidth }}
          >
            <span className="text-[10px] font-medium text-foreground">
              {selected.row} × {selected.col}
            </span>
            <span className="text-[10px] text-muted-foreground">r =</span>
            <span
              className={`text-[10px] font-semibold ${
                selected.value != null && selected.value > 0 ? "text-blue-600" : "text-red-600"
              }`}
            >
              {selected.value != null ? selected.value.toFixed(3) : "N/A"}
            </span>
            <button
              className="ml-auto text-[9px] text-muted-foreground hover:text-foreground"
              onClick={() => setSelected(null)}
            >
              ✕
            </button>
          </div>
        )}
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
  // Default Y-axis label to first key when not provided
  const effectiveYLabel = yLabel || yKeys[0] || ""
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
            label={effectiveYLabel ? { value: effectiveYLabel, angle: -90, position: "insideLeft", style: axisStyle } : undefined}
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
          <YAxis
            tick={axisStyle}
            label={effectiveYLabel ? { value: effectiveYLabel, angle: -90, position: "insideLeft", style: axisStyle } : undefined}
          />
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

// ---------------------------------------------------------------------------
// Box plot SVG renderer
// ---------------------------------------------------------------------------

interface BoxRow {
  group: string
  min: number
  q1: number
  median: number
  q3: number
  max: number
  mean?: number
  count?: number
}

function BoxPlotChart({
  data,
  xLabel,
  yLabel,
}: {
  data: Record<string, unknown>[]
  xLabel: string
  yLabel: string
}) {
  const rows = data as unknown as BoxRow[]
  if (rows.length === 0) return <p className="text-xs text-muted-foreground">No data</p>

  const SVG_H = 200
  const SVG_W = Math.max(300, rows.length * 80 + 60)
  const PAD = { top: 16, right: 16, bottom: 40, left: 52 }
  const plotH = SVG_H - PAD.top - PAD.bottom
  const plotW = SVG_W - PAD.left - PAD.right

  const allVals = rows.flatMap((r) => [r.min, r.q1, r.median, r.q3, r.max])
  const yMin = Math.min(...allVals)
  const yMax = Math.max(...allVals)
  const yRange = yMax - yMin || 1

  const scaleY = (v: number) => PAD.top + plotH - ((v - yMin) / yRange) * plotH
  const boxW = Math.min(40, plotW / rows.length / 1.5)
  const boxCx = (i: number) => PAD.left + (i + 0.5) * (plotW / rows.length)

  // Y axis ticks (4 evenly spaced)
  const ticks = Array.from({ length: 5 }, (_, i) => yMin + (yRange * i) / 4)

  const BOX_COLOR = "#6366f1"
  const MED_COLOR = "#ec4899"

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="w-full"
      style={{ height: SVG_H }}
      aria-label="Box plot chart"
    >
      {/* Y axis ticks */}
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={PAD.left}
            x2={PAD.left + plotW}
            y1={scaleY(t)}
            y2={scaleY(t)}
            stroke="hsl(var(--border))"
            strokeWidth={0.5}
          />
          <text
            x={PAD.left - 6}
            y={scaleY(t) + 4}
            textAnchor="end"
            fontSize={9}
            fill="hsl(var(--muted-foreground))"
          >
            {t.toLocaleString(undefined, { maximumFractionDigits: 1 })}
          </text>
        </g>
      ))}

      {/* Y axis label */}
      <text
        transform={`rotate(-90)`}
        x={-(PAD.top + plotH / 2)}
        y={14}
        textAnchor="middle"
        fontSize={9}
        fill="hsl(var(--muted-foreground))"
      >
        {yLabel}
      </text>

      {/* Boxes */}
      {rows.map((row, i) => {
        const cx = boxCx(i)
        const x0 = cx - boxW / 2

        const yQ1 = scaleY(row.q1)
        const yQ3 = scaleY(row.q3)
        const yMed = scaleY(row.median)
        const yMinS = scaleY(row.min)
        const yMaxS = scaleY(row.max)
        const capW = boxW * 0.4

        return (
          <g key={row.group} role="img" aria-label={`${row.group}: median ${row.median}`}>
            {/* Whisker stem */}
            <line x1={cx} x2={cx} y1={yMinS} y2={yMaxS} stroke={BOX_COLOR} strokeWidth={1.5} />
            {/* Min cap */}
            <line x1={cx - capW / 2} x2={cx + capW / 2} y1={yMinS} y2={yMinS} stroke={BOX_COLOR} strokeWidth={1.5} />
            {/* Max cap */}
            <line x1={cx - capW / 2} x2={cx + capW / 2} y1={yMaxS} y2={yMaxS} stroke={BOX_COLOR} strokeWidth={1.5} />
            {/* IQR box */}
            <rect
              x={x0}
              y={yQ3}
              width={boxW}
              height={Math.max(1, yQ1 - yQ3)}
              fill={`${BOX_COLOR}33`}
              stroke={BOX_COLOR}
              strokeWidth={1.5}
              rx={2}
            />
            {/* Median line */}
            <line x1={x0} x2={x0 + boxW} y1={yMed} y2={yMed} stroke={MED_COLOR} strokeWidth={2} />

            {/* X-axis label */}
            <text
              x={cx}
              y={SVG_H - PAD.bottom + 14}
              textAnchor="middle"
              fontSize={9}
              fill="hsl(var(--muted-foreground))"
            >
              {String(row.group).length > 10 ? String(row.group).slice(0, 9) + "…" : String(row.group)}
            </text>
          </g>
        )
      })}

      {/* X axis label */}
      <text
        x={PAD.left + plotW / 2}
        y={SVG_H - 4}
        textAnchor="middle"
        fontSize={9}
        fill="hsl(var(--muted-foreground))"
      >
        {xLabel}
      </text>

      {/* Legend */}
      <g>
        <rect x={PAD.left} y={PAD.top - 12} width={8} height={8} fill={`${BOX_COLOR}33`} stroke={BOX_COLOR} rx={1} />
        <text x={PAD.left + 10} y={PAD.top - 5} fontSize={8} fill="hsl(var(--muted-foreground))">IQR</text>
        <line x1={PAD.left + 28} x2={PAD.left + 36} y1={PAD.top - 9} y2={PAD.top - 9} stroke={MED_COLOR} strokeWidth={2} />
        <text x={PAD.left + 38} y={PAD.top - 5} fontSize={8} fill="hsl(var(--muted-foreground))">Median</text>
      </g>
    </svg>
  )
}
