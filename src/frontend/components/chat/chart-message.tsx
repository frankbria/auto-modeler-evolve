"use client"

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

    case "scatter":
      return (
        <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="x" type="number" name={xLabel} tick={axisStyle} />
          <YAxis dataKey="y" type="number" name={yLabel} tick={axisStyle} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} {...tooltipStyle} />
          <Scatter data={data} fill={COLORS[0]} />
        </ScatterChart>
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
