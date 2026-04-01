"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { WebhookConfig } from "@/lib/types"

// ---------------------------------------------------------------------------
// WebhookCard — register & manage webhook notifications for a deployment
// ---------------------------------------------------------------------------

const ALL_EVENT_TYPES = [
  { key: "batch_complete", label: "Batch complete" },
  { key: "drift_detected", label: "Drift detected" },
  { key: "health_degraded", label: "Health degraded" },
]

interface WebhookCardProps {
  deploymentId: string
}

export function WebhookCard({ deploymentId }: WebhookCardProps) {
  const [webhooks, setWebhooks] = useState<WebhookConfig[]>([])
  const [loading, setLoading] = useState(true)
  const [addingUrl, setAddingUrl] = useState("")
  const [addingEvents, setAddingEvents] = useState<string[]>([
    "batch_complete",
    "drift_detected",
    "health_degraded",
  ])
  const [showForm, setShowForm] = useState(false)
  const [adding, setAdding] = useState(false)
  const [newSecret, setNewSecret] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.deploy
      .getWebhooks(deploymentId)
      .then(setWebhooks)
      .catch(() => setWebhooks([]))
      .finally(() => setLoading(false))
  }, [deploymentId])

  async function handleAdd() {
    if (!addingUrl.trim()) return
    setAdding(true)
    setError(null)
    setNewSecret(null)
    try {
      const created = await api.deploy.createWebhook(
        deploymentId,
        addingUrl.trim(),
        addingEvents
      )
      setWebhooks((prev) => [...prev, created])
      if (created.secret) setNewSecret(created.secret)
      setAddingUrl("")
      setAddingEvents(["batch_complete", "drift_detected", "health_degraded"])
      setShowForm(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create webhook")
    } finally {
      setAdding(false)
    }
  }

  async function handleDelete(webhookId: string) {
    await api.deploy.deleteWebhook(deploymentId, webhookId).catch(() => null)
    setWebhooks((prev) => prev.filter((w) => w.id !== webhookId))
  }

  async function handleTest(webhookId: string) {
    setTestResults((prev) => ({ ...prev, [webhookId]: "testing..." }))
    try {
      const result = await api.deploy.testWebhook(deploymentId, webhookId)
      setTestResults((prev) => ({
        ...prev,
        [webhookId]: result.success
          ? `OK (HTTP ${result.status_code})`
          : `Failed (HTTP ${result.status_code})`,
      }))
    } catch {
      setTestResults((prev) => ({ ...prev, [webhookId]: "Network error" }))
    }
  }

  function toggleEvent(key: string) {
    setAddingEvents((prev) =>
      prev.includes(key) ? prev.filter((e) => e !== key) : [...prev, key]
    )
  }

  if (loading) return null

  return (
    <Card className="border-sky-500 border-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <span aria-hidden="true">🔔</span> Webhook Notifications
          <Badge variant="outline" className="ml-auto">
            {webhooks.length} registered
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-muted-foreground text-xs">
          AutoModeler sends a signed POST request to your URL when key events
          occur. Verify the{" "}
          <code className="bg-muted px-1 rounded">
            X-AutoModeler-Signature
          </code>{" "}
          header (HMAC-SHA256 of the request body) to confirm authenticity.
        </p>

        {/* Secret revealed once after creation */}
        {newSecret && (
          <div
            className="bg-amber-50 border border-amber-300 rounded p-2 text-xs space-y-1"
            role="alert"
          >
            <p className="font-semibold text-amber-800">
              Save this secret — it will not be shown again
            </p>
            <code className="block bg-amber-100 rounded px-2 py-1 break-all select-all">
              {newSecret}
            </code>
            <Button
              size="sm"
              variant="outline"
              className="h-6 text-xs"
              onClick={() => {
                navigator.clipboard.writeText(newSecret)
              }}
            >
              Copy
            </Button>
          </div>
        )}

        {/* Existing webhooks */}
        {webhooks.length > 0 && (
          <div className="space-y-2">
            {webhooks.map((wh) => (
              <div
                key={wh.id}
                className="border rounded p-2 space-y-1"
                data-testid="webhook-row"
              >
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-mono text-xs truncate max-w-[200px]">
                    {wh.url}
                  </span>
                  <div className="flex gap-1 flex-wrap">
                    {wh.event_types.map((et) => (
                      <Badge
                        key={et}
                        variant="secondary"
                        className="text-[10px] py-0"
                      >
                        {et.replace("_", " ")}
                      </Badge>
                    ))}
                  </div>
                  <div className="ml-auto flex gap-1">
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 text-xs"
                      onClick={() => handleTest(wh.id)}
                    >
                      Test
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 text-xs text-rose-600"
                      onClick={() => handleDelete(wh.id)}
                    >
                      Remove
                    </Button>
                  </div>
                </div>
                {testResults[wh.id] && (
                  <p
                    className={`text-xs ${
                      testResults[wh.id].startsWith("OK")
                        ? "text-emerald-600"
                        : "text-rose-600"
                    }`}
                  >
                    {testResults[wh.id]}
                  </p>
                )}
                {wh.last_fired_at && (
                  <p className="text-muted-foreground text-[10px]">
                    Last fired:{" "}
                    {new Date(wh.last_fired_at).toLocaleString()}{" "}
                    {wh.last_status_code != null && (
                      <span>— HTTP {wh.last_status_code}</span>
                    )}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Add webhook form */}
        {showForm ? (
          <div className="border rounded p-2 space-y-2">
            <input
              type="url"
              placeholder="https://your-server.com/webhook"
              value={addingUrl}
              onChange={(e) => setAddingUrl(e.target.value)}
              className="w-full border rounded px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-sky-400"
              aria-label="Webhook URL"
            />
            <fieldset>
              <legend className="text-xs text-muted-foreground mb-1">
                Trigger on:
              </legend>
              <div className="flex flex-wrap gap-2">
                {ALL_EVENT_TYPES.map(({ key, label }) => (
                  <label
                    key={key}
                    className="flex items-center gap-1 text-xs cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={addingEvents.includes(key)}
                      onChange={() => toggleEvent(key)}
                      aria-label={label}
                    />
                    {label}
                  </label>
                ))}
              </div>
            </fieldset>
            {error && <p className="text-rose-600 text-xs">{error}</p>}
            <div className="flex gap-2">
              <Button
                size="sm"
                className="h-7 text-xs"
                onClick={handleAdd}
                disabled={adding || !addingUrl.trim() || addingEvents.length === 0}
              >
                {adding ? "Saving…" : "Save webhook"}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 text-xs"
                onClick={() => {
                  setShowForm(false)
                  setError(null)
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs"
            onClick={() => setShowForm(true)}
          >
            + Add webhook
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
