"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { WebhookRegisteredInfo } from "@/lib/types"

const EVENT_LABELS: Record<string, string> = {
  batch_complete: "Batch Complete",
  drift_detected: "Drift Detected",
  health_degraded: "Health Degraded",
  quota_alert: "Quota Alert",
}

interface WebhookRegisteredCardProps {
  info: WebhookRegisteredInfo
}

export function WebhookRegisteredCard({ info }: WebhookRegisteredCardProps) {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(info.secret).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <figure
      role="region"
      aria-label="Webhook registered"
      className="rounded-lg border border-emerald-300 bg-emerald-50/40 overflow-hidden"
    >
      <Card className="border-0 bg-transparent shadow-none">
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className="text-sm font-semibold text-emerald-800 flex items-center gap-2">
            <span aria-hidden="true">🔔</span>
            Webhook Registered
            <Badge className="ml-auto text-xs bg-emerald-100 text-emerald-800 border-emerald-300">
              Active
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3 space-y-3">
          <div className="text-sm text-gray-700">
            <span className="font-medium">URL: </span>
            <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded break-all">
              {info.url}
            </code>
          </div>

          <div className="flex flex-wrap gap-1">
            {info.event_types.map((et) => (
              <Badge
                key={et}
                className="text-xs bg-blue-100 text-blue-800 border-blue-300"
              >
                {EVENT_LABELS[et] ?? et}
              </Badge>
            ))}
          </div>

          <div
            role="alert"
            className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 space-y-1"
          >
            <p className="text-xs font-semibold text-amber-800">
              Signing secret — shown once
            </p>
            <div className="flex items-center gap-2">
              <code
                className="text-xs break-all text-amber-900 flex-1"
                data-testid="webhook-secret"
              >
                {info.secret}
              </code>
              <Button
                size="sm"
                variant="outline"
                className="shrink-0 text-xs h-6 px-2 border-amber-400 text-amber-800 hover:bg-amber-100"
                onClick={handleCopy}
                aria-label="Copy signing secret to clipboard"
              >
                {copied ? "Copied!" : "Copy"}
              </Button>
            </div>
            <p className="text-xs text-amber-700">
              Store this secret to verify the{" "}
              <code className="text-xs">X-AutoModeler-Signature</code> header on
              each dispatch.
            </p>
          </div>

          <p className="text-xs text-gray-500">
            Say &ldquo;test my webhook&rdquo; to confirm the URL is reachable.
          </p>
        </CardContent>
      </Card>
      <figcaption className="sr-only">
        Webhook registered at {info.url} for events:{" "}
        {info.event_types.join(", ")}. Signing secret displayed once.
      </figcaption>
    </figure>
  )
}
