"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import type { Deployment } from "@/lib/types"

interface DeploymentPanelProps {
  projectId: string
  selectedRunId: string | null
  algorithmName: string | null
  onDeployed?: (deployment: Deployment) => void
}

export function DeploymentPanel({
  projectId,
  selectedRunId,
  algorithmName,
  onDeployed,
}: DeploymentPanelProps) {
  const [deploying, setDeploying] = useState(false)
  const [deployment, setDeployment] = useState<Deployment | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [undeploying, setUndeploying] = useState(false)

  const handleDeploy = async () => {
    if (!selectedRunId) return
    setDeploying(true)
    setError(null)
    try {
      const result = await api.deploy.deploy(selectedRunId)
      setDeployment(result)
      onDeployed?.(result)
    } catch (e) {
      setError("Deployment failed. Please try again.")
      console.error(e)
    } finally {
      setDeploying(false)
    }
  }

  const handleUndeploy = async () => {
    if (!deployment) return
    setUndeploying(true)
    try {
      await api.deploy.undeploy(deployment.id)
      setDeployment(null)
    } catch (e) {
      console.error(e)
    } finally {
      setUndeploying(false)
    }
  }

  if (!selectedRunId) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center">
        <p className="text-sm text-muted-foreground">
          Select a model in the <strong>Models</strong> tab before deploying.
        </p>
      </div>
    )
  }

  if (deployment) {
    const dashboardUrl = `${typeof window !== "undefined" ? window.location.origin : ""}${deployment.dashboard_url}`
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-green-700 dark:text-green-400">
            Model deployed
          </span>
          <Badge variant="outline" className="ml-auto">
            {deployment.algorithm ?? "Unknown"}
          </Badge>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Live Prediction Dashboard</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div>
              <p className="text-xs font-medium text-muted-foreground">Dashboard URL</p>
              <a
                href={deployment.dashboard_url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-0.5 block truncate text-primary underline-offset-2 hover:underline"
              >
                {dashboardUrl}
              </a>
              <p className="mt-1 text-xs text-muted-foreground">
                Share this link — anyone can paste in values and see predictions.
              </p>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground">API Endpoint</p>
              <code className="mt-0.5 block rounded bg-muted px-2 py-1 text-xs">
                POST {typeof window !== "undefined" ? `${window.location.origin}/api` : "http://localhost:8000"}
                {deployment.endpoint_path}
              </code>
              <p className="mt-1 text-xs text-muted-foreground">
                Send JSON with feature values, get back a prediction.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Requests: {deployment.request_count}</span>
              {deployment.last_predicted_at && (
                <span>
                  · Last used:{" "}
                  {new Date(deployment.last_predicted_at).toLocaleDateString()}
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end">
          <Button
            variant="outline"
            size="sm"
            onClick={handleUndeploy}
            disabled={undeploying}
          >
            {undeploying ? "Undeploying..." : "Undeploy"}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg border p-4">
        <h4 className="text-sm font-semibold">Ready to deploy</h4>
        <p className="mt-1 text-xs text-muted-foreground">
          Deploying <strong>{algorithmName ?? "selected model"}</strong> will create a live
          prediction API endpoint and an interactive dashboard you can share with anyone.
          No code required.
        </p>
        <ul className="mt-3 space-y-1 text-xs text-muted-foreground">
          <li>✓ Auto-generated prediction form with your feature columns</li>
          <li>✓ JSON API endpoint for developers</li>
          <li>✓ Batch prediction via CSV upload</li>
          <li>✓ Usage statistics dashboard</li>
        </ul>
      </div>

      {error && (
        <p className="text-xs text-destructive">{error}</p>
      )}

      <Button
        onClick={handleDeploy}
        disabled={deploying}
        className="w-full"
      >
        {deploying ? "Deploying..." : "Deploy Model"}
      </Button>
    </div>
  )
}
