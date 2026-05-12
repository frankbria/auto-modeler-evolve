"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { DeploymentsOverviewResult, DeploymentStatusRow } from "@/lib/types"

interface DeploymentsOverviewCardProps {
  result: DeploymentsOverviewResult
}

function HealthBar({ score }: { score: number }) {
  const color =
    score >= 75 ? "bg-emerald-500" : score >= 50 ? "bg-amber-500" : "bg-rose-500"
  return (
    <div
      role="progressbar"
      aria-valuenow={score}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Health score: ${score}%`}
      className="h-1.5 w-24 rounded-full bg-gray-200 overflow-hidden"
    >
      <div className={`h-full rounded-full ${color}`} style={{ width: `${score}%` }} />
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  if (status === "healthy") {
    return <Badge className="text-xs bg-emerald-100 text-emerald-800 border-emerald-300">Healthy</Badge>
  }
  if (status === "warning") {
    return <Badge className="text-xs bg-amber-100 text-amber-800 border-amber-300">Warning</Badge>
  }
  return <Badge className="text-xs bg-rose-100 text-rose-800 border-rose-300">Critical</Badge>
}

function EnvBadge({ env }: { env: string }) {
  if (env === "production") {
    return <Badge className="text-xs bg-green-100 text-green-800 border-green-300">Production</Badge>
  }
  return <Badge className="text-xs bg-slate-100 text-slate-600 border-slate-300">Staging</Badge>
}

function DeploymentRow({ dep }: { dep: DeploymentStatusRow }) {
  const borderColor =
    dep.status === "healthy"
      ? "border-l-emerald-400"
      : dep.status === "warning"
      ? "border-l-amber-400"
      : "border-l-rose-400"

  return (
    <li
      className={`rounded-md border-l-4 ${borderColor} bg-white/60 px-3 py-2 space-y-1`}
      data-testid="deployment-row"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-gray-800 truncate max-w-[200px]">
          {dep.algorithm_plain || dep.algorithm || "Model"} → {dep.target_column || "target"}
        </span>
        <EnvBadge env={dep.environment} />
        <StatusBadge status={dep.status} />
        {dep.api_key_enabled && (
          <Badge className="text-xs bg-amber-100 text-amber-700 border-amber-300">
            <span aria-hidden="true">🔑</span> Protected
          </Badge>
        )}
      </div>

      <p className="text-xs text-gray-500">
        Project: <span className="font-medium text-gray-700">{dep.project_name}</span>
      </p>

      <div className="flex flex-wrap items-center gap-4">
        <div>
          <HealthBar score={dep.health_score} />
          <span className="text-xs text-gray-500">{dep.health_score}% health</span>
        </div>
        <span className="text-xs text-gray-600">
          <span className="font-medium">{dep.request_count.toLocaleString()}</span> total predictions
        </span>
        <span className="text-xs text-gray-600">
          <span className="font-medium">{dep.predictions_last_7d}</span> last 7 days
        </span>
        {dep.predictions_today > 0 && (
          <span className="text-xs text-gray-600">
            <span className="font-medium">{dep.predictions_today}</span> today
          </span>
        )}
      </div>

      {dep.top_issue && (
        <p className="text-xs text-amber-700 italic">{dep.top_issue}</p>
      )}
    </li>
  )
}

export function DeploymentsOverviewCard({ result }: DeploymentsOverviewCardProps) {
  const borderColor =
    result.critical_count > 0
      ? "border-rose-300"
      : result.warning_count > 0
      ? "border-amber-300"
      : "border-emerald-300"

  const bgColor =
    result.critical_count > 0
      ? "bg-rose-50/40"
      : result.warning_count > 0
      ? "bg-amber-50/40"
      : "bg-emerald-50/40"

  if (result.total_deployments === 0) {
    return (
      <figure
        role="region"
        aria-label="No active deployments"
        className={`rounded-lg border ${borderColor} ${bgColor} overflow-hidden`}
      >
        <Card className="border-0 bg-transparent shadow-none">
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              <span aria-hidden="true">🚀</span>
              Active Deployments
              <Badge className="ml-auto text-xs bg-slate-100 text-slate-600 border-slate-300">0 live</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3">
            <p className="text-sm text-gray-600">{result.summary}</p>
          </CardContent>
        </Card>
        <figcaption className="sr-only">No active deployments found.</figcaption>
      </figure>
    )
  }

  return (
    <figure
      role="region"
      aria-label="Deployment status overview"
      className={`rounded-lg border ${borderColor} ${bgColor} overflow-hidden`}
    >
      <Card className="border-0 bg-transparent shadow-none">
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className="text-sm font-semibold text-gray-800 flex flex-wrap items-center gap-2">
            <span aria-hidden="true">🚀</span>
            Active Deployments
            <Badge className="text-xs bg-gray-100 text-gray-700 border-gray-300">
              {result.total_deployments} live
            </Badge>
            {result.production_count > 0 && (
              <Badge className="text-xs bg-green-100 text-green-800 border-green-300">
                {result.production_count} production
              </Badge>
            )}
            {result.healthy_count > 0 && (
              <Badge className="text-xs bg-emerald-100 text-emerald-700 border-emerald-300" data-testid="healthy-badge">
                {result.healthy_count} healthy
              </Badge>
            )}
            {result.warning_count > 0 && (
              <Badge className="text-xs bg-amber-100 text-amber-700 border-amber-300" data-testid="warning-badge">
                {result.warning_count} warning
              </Badge>
            )}
            {result.critical_count > 0 && (
              <Badge className="text-xs bg-rose-100 text-rose-700 border-rose-300" data-testid="critical-badge">
                {result.critical_count} critical
              </Badge>
            )}
          </CardTitle>
        </CardHeader>

        <CardContent className="px-4 pb-3 space-y-3">
          {/* Stats row */}
          <div className="flex flex-wrap gap-4 text-xs text-gray-600">
            <span>
              <span className="font-semibold text-gray-800">{result.total_predictions.toLocaleString()}</span>{" "}
              total predictions
            </span>
            <span>
              <span className="font-semibold text-gray-800">{result.avg_health_score}%</span>{" "}
              avg health
            </span>
            <span>
              {result.staging_count > 0 && (
                <>{result.staging_count} staging</>
              )}
            </span>
          </div>

          {/* Deployment list */}
          <ul className="space-y-2" aria-label="Deployment list">
            {result.deployments.map((dep) => (
              <DeploymentRow key={dep.deployment_id} dep={dep} />
            ))}
          </ul>

          {/* Summary */}
          <p className="text-xs text-gray-500 italic">{result.summary}</p>
        </CardContent>
      </Card>
      <figcaption className="sr-only">
        {result.total_deployments} active deployments: {result.healthy_count} healthy,{" "}
        {result.warning_count} warning, {result.critical_count} critical.
      </figcaption>
    </figure>
  )
}
