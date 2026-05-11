"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { WebhookRemovedChatInfo } from "@/lib/types"

interface WebhookRemovedChatCardProps {
  info: WebhookRemovedChatInfo
}

export function WebhookRemovedChatCard({ info }: WebhookRemovedChatCardProps) {
  const removed = info.removed ?? []

  return (
    <figure
      role="region"
      aria-label="Webhook removed"
      className="rounded-lg border border-rose-300 bg-rose-50/40 overflow-hidden"
    >
      <Card className="border-0 bg-transparent shadow-none">
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className="text-sm font-semibold text-rose-800 flex items-center gap-2">
            <span aria-hidden="true">🗑️</span>
            Webhook Removed
            <Badge
              className="ml-auto text-xs bg-rose-100 text-rose-800 border-rose-300"
              data-testid="removed-count-badge"
            >
              {removed.length} removed
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3 space-y-2">
          {removed.length > 0 ? (
            <ul className="space-y-1">
              {removed.map((url) => (
                <li key={url} className="text-xs text-rose-700">
                  <code className="bg-rose-100 px-1.5 py-0.5 rounded break-all">
                    {url}
                  </code>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-500 italic">
              No matching webhooks were found to remove.
            </p>
          )}
          <p className="text-xs text-gray-500">{info.summary}</p>
        </CardContent>
      </Card>
      <figcaption className="sr-only">{info.summary}</figcaption>
    </figure>
  )
}
