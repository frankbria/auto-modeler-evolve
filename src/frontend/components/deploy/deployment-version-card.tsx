"use client"

import { useState, useEffect, useCallback } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { api } from "@/lib/api"
import type { DeploymentVersion, DeploymentVersionHistory } from "@/lib/types"

interface DeploymentVersionCardProps {
  deploymentId: string
  /** Called after a successful rollback so parent can refresh deployment state */
  onRollback?: (newModelRunId: string) => void
}

function _primaryMetric(metrics: Record<string, number>): { name: string; value: string } | null {
  if (metrics.r2 !== undefined) return { name: "R²", value: metrics.r2.toFixed(3) }
  if (metrics.accuracy !== undefined) return { name: "Accuracy", value: (metrics.accuracy * 100).toFixed(1) + "%" }
  const keys = Object.keys(metrics)
  if (!keys.length) return null
  return { name: keys[0], value: metrics[keys[0]].toFixed(3) }
}

function _algoLabel(algo: string | null): string {
  if (!algo) return "Unknown"
  return algo.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function VersionRow({
  version,
  onRollback,
  rolling,
}: {
  version: DeploymentVersion
  onRollback: (versionNumber: number) => void
  rolling: boolean
}) {
  const metric = _primaryMetric(version.metrics)
  const dateLabel = version.deployed_at
    ? new Date(version.deployed_at).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—"

  return (
    <div
      data-testid={`version-row-${version.version_number}`}
      className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${
        version.is_current ? "bg-indigo-50 border border-indigo-200" : "bg-muted/40"
      }`}
    >
      {/* Version badge */}
      <Badge
        variant="outline"
        className={`shrink-0 font-mono tabular-nums w-8 justify-center ${
          version.is_current ? "border-indigo-500 text-indigo-700" : "text-muted-foreground"
        }`}
      >
        v{version.version_number}
      </Badge>

      {/* Algorithm & metric */}
      <div className="flex-1 min-w-0">
        <div className="font-medium text-foreground truncate">{_algoLabel(version.algorithm)}</div>
        <div className="text-xs text-muted-foreground">
          {metric ? `${metric.name} ${metric.value} · ` : ""}
          {dateLabel}
        </div>
      </div>

      {/* Status or rollback */}
      {version.is_current ? (
        <Badge className="bg-indigo-100 text-indigo-700 border-indigo-300 shrink-0">
          Current
        </Badge>
      ) : (
        <Button
          variant="outline"
          size="sm"
          disabled={rolling}
          onClick={() => onRollback(version.version_number)}
          data-testid={`rollback-btn-${version.version_number}`}
          className="shrink-0 text-xs"
        >
          {rolling ? "Restoring…" : `Restore v${version.version_number}`}
        </Button>
      )}
    </div>
  )
}

export function DeploymentVersionCard({
  deploymentId,
  onRollback,
}: DeploymentVersionCardProps) {
  const [history, setHistory] = useState<DeploymentVersionHistory | null>(null)
  const [rolling, setRolling] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmVersion, setConfirmVersion] = useState<number | null>(null)

  const load = useCallback(async () => {
    try {
      const data = await api.deploy.getVersions(deploymentId)
      setHistory(data)
    } catch {
      // silently ignore — versions panel is enhancement, not core
    }
  }, [deploymentId])

  useEffect(() => {
    load()
  }, [load])

  const handleRollback = async (versionNumber: number) => {
    if (confirmVersion !== versionNumber) {
      setConfirmVersion(versionNumber)
      return
    }
    setRolling(true)
    setError(null)
    setConfirmVersion(null)
    try {
      const result = await api.deploy.rollback(deploymentId, versionNumber)
      await load()
      onRollback?.(result.model_run_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Rollback failed")
    } finally {
      setRolling(false)
    }
  }

  // Only show when 2+ versions exist
  if (!history || history.versions.length < 2) return null

  return (
    <div
      data-testid="deployment-version-card"
      className="rounded-xl border-2 border-indigo-200 bg-card p-4 space-y-3"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base font-semibold text-foreground">Version History</span>
          <Badge variant="outline" className="text-xs font-mono">
            {history.versions.length} versions
          </Badge>
        </div>
        <span className="text-xs text-muted-foreground">
          Current: v{history.current_version_number}
        </span>
      </div>

      {confirmVersion !== null && (
        <div
          data-testid="rollback-confirm"
          className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm"
        >
          <span className="font-medium text-amber-800">
            Restore v{confirmVersion}? The live endpoint will switch to that model version.
          </span>
          <div className="flex gap-2 mt-2">
            <Button
              size="sm"
              variant="destructive"
              disabled={rolling}
              onClick={() => handleRollback(confirmVersion)}
              data-testid="confirm-rollback-btn"
            >
              Yes, restore v{confirmVersion}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setConfirmVersion(null)}
              data-testid="cancel-rollback-btn"
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      {error && (
        <p className="text-sm text-destructive" data-testid="rollback-error">
          {error}
        </p>
      )}

      <div className="space-y-1.5">
        {history.versions.map((v) => (
          <VersionRow
            key={v.id}
            version={v}
            onRollback={handleRollback}
            rolling={rolling}
          />
        ))}
      </div>

      <p className="text-xs text-muted-foreground">
        Rollback restores the live endpoint to a previous model version — the prediction URL stays the same.
      </p>
    </div>
  )
}
