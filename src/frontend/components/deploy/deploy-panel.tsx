"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import type { Deployment } from "@/lib/types"

interface DeployPanelProps {
  modelRunId: string
  displayName: string
  targetColumn: string | null
  problemType: string | null
  onDeployed?: (deployment: Deployment) => void
  onMessageOut?: (msg: string) => void
}

export function DeployPanel({
  modelRunId,
  displayName,
  targetColumn,
  problemType,
  onDeployed,
  onMessageOut,
}: DeployPanelProps) {
  const [deploying, setDeploying] = useState(false)
  const [deployment, setDeployment] = useState<Deployment | null>(null)
  const [batchFile, setBatchFile] = useState<File | null>(null)
  const [batchRunning, setBatchRunning] = useState(false)
  const [undeploying, setUndeploying] = useState(false)

  const dashboardUrl =
    typeof window !== "undefined" && deployment
      ? `${window.location.origin}/predict/${deployment.id}`
      : ""

  async function handleDeploy() {
    setDeploying(true)
    try {
      const dep = await api.deploy.deploy(modelRunId)
      setDeployment(dep)
      onDeployed?.(dep)
      const url = `${window.location.origin}/predict/${dep.id}`
      onMessageOut?.(
        `Your model is live! Share this link with your team to make predictions:\n\n${url}\n\nAnyone with the link can paste in new numbers and get a forecast — no login required.`
      )
    } catch {
      onMessageOut?.("Deployment failed. Please try again.")
    } finally {
      setDeploying(false)
    }
  }

  async function handleUndeploy() {
    if (!deployment) return
    setUndeploying(true)
    try {
      await api.deploy.undeploy(deployment.id)
      setDeployment(null)
      onMessageOut?.(`${displayName} has been undeployed. You can redeploy it at any time.`)
    } catch {
      onMessageOut?.("Undeploy failed. Please try again.")
    } finally {
      setUndeploying(false)
    }
  }

  async function handleBatchPredict() {
    if (!batchFile || !deployment) return
    setBatchRunning(true)
    try {
      const response = await api.deploy.predictBatch(deployment.id, batchFile)
      if (!response.ok) throw new Error("Batch prediction failed")
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `predictions_${deployment.id.slice(0, 8)}.csv`
      a.click()
      URL.revokeObjectURL(url)
      onMessageOut?.(
        `Batch predictions complete! The CSV has been downloaded with a \`predicted_${targetColumn ?? "value"}\` column for each row.`
      )
    } catch {
      onMessageOut?.("Batch prediction failed. Make sure your CSV has the right column names.")
    } finally {
      setBatchRunning(false)
    }
  }

  return (
    <div className="space-y-4 p-4">
      <div>
        <h3 className="text-sm font-semibold">Deploy Model</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Deploy <strong>{displayName}</strong> as a live prediction API and shareable dashboard.
        </p>
      </div>

      {!deployment ? (
        <Card>
          <CardContent className="pt-4">
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs">
                <span className="text-muted-foreground">Model:</span>
                <span className="font-medium">{displayName}</span>
              </div>
              {targetColumn && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Predicts:</span>
                  <Badge variant="outline">{targetColumn}</Badge>
                </div>
              )}
              {problemType && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-muted-foreground">Type:</span>
                  <Badge variant="secondary">{problemType}</Badge>
                </div>
              )}
              <Button
                className="w-full"
                size="sm"
                onClick={handleDeploy}
                disabled={deploying}
              >
                {deploying ? "Deploying..." : "Deploy Model"}
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card className="border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm text-green-800 dark:text-green-200">
                Live
                <Badge variant="outline" className="border-green-500 text-green-700 dark:text-green-300">
                  Active
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-xs text-green-700 dark:text-green-300">
                <p className="font-medium">Prediction Dashboard</p>
                <a
                  href={dashboardUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1 block break-all underline hover:no-underline"
                >
                  {dashboardUrl}
                </a>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>Requests served:</span>
                <span className="font-medium">{deployment.request_count}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="w-full border-red-200 text-red-700 hover:bg-red-50 dark:border-red-900 dark:text-red-400"
                onClick={handleUndeploy}
                disabled={undeploying}
              >
                {undeploying ? "Undeploying..." : "Undeploy"}
              </Button>
            </CardContent>
          </Card>

          {/* Batch prediction */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Batch Predictions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Upload a CSV with the same columns as your training data. Download predictions for every row.
              </p>
              <input
                type="file"
                accept=".csv"
                className="text-xs"
                onChange={(e) => setBatchFile(e.target.files?.[0] ?? null)}
              />
              <Button
                size="sm"
                className="w-full"
                onClick={handleBatchPredict}
                disabled={!batchFile || batchRunning}
              >
                {batchRunning ? "Running..." : "Run Batch Predictions"}
              </Button>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
