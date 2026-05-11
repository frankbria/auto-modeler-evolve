"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { WebhookTestChatResult } from "@/lib/types"

interface WebhookTestChatCardProps {
  result: WebhookTestChatResult
}

export function WebhookTestChatCard({ result }: WebhookTestChatCardProps) {
  const hasWebhook = result.url !== null
  const borderClass = !hasWebhook
    ? "border-slate-300 bg-slate-50/40"
    : result.success
      ? "border-emerald-300 bg-emerald-50/40"
      : "border-rose-300 bg-rose-50/40"

  const titleClass = !hasWebhook
    ? "text-slate-800"
    : result.success
      ? "text-emerald-800"
      : "text-rose-800"

  return (
    <figure
      role="region"
      aria-label="Webhook test result"
      className={`rounded-lg border overflow-hidden ${borderClass}`}
    >
      <Card className="border-0 bg-transparent shadow-none">
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle
            className={`text-sm font-semibold flex items-center gap-2 ${titleClass}`}
          >
            <span aria-hidden="true">⚡</span>
            Webhook Test
            {hasWebhook && (
              <Badge
                className={
                  result.success
                    ? "ml-auto text-xs bg-emerald-100 text-emerald-800 border-emerald-300"
                    : "ml-auto text-xs bg-rose-100 text-rose-800 border-rose-300"
                }
                data-testid="test-status-badge"
              >
                {result.success ? "Success" : "Failed"}
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3 space-y-2">
          {!hasWebhook ? (
            <p className="text-sm text-gray-500 italic">
              No webhooks registered. Register one first with &ldquo;register a
              webhook at https://...&rdquo;
            </p>
          ) : (
            <>
              <div className="flex items-center gap-2 text-sm">
                <span className="text-gray-600 font-medium">URL:</span>
                <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded break-all flex-1">
                  {result.url}
                </code>
              </div>
              {result.status_code !== null && (
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-gray-600 font-medium">HTTP:</span>
                  <Badge
                    className={
                      result.success
                        ? "text-xs bg-emerald-100 text-emerald-800 border-emerald-300"
                        : "text-xs bg-rose-100 text-rose-800 border-rose-300"
                    }
                  >
                    {result.status_code}
                  </Badge>
                </div>
              )}
              {!result.success && result.url && (
                <p className="text-xs text-rose-700">
                  Check that {result.url} is publicly accessible and accepts POST
                  requests.
                </p>
              )}
            </>
          )}
          <p className="text-xs text-gray-500">{result.summary}</p>
        </CardContent>
      </Card>
      <figcaption className="sr-only">{result.summary}</figcaption>
    </figure>
  )
}
