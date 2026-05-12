"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ApiKeyResultInfo } from "@/lib/types"

interface ApiKeyChatCardProps {
  result: ApiKeyResultInfo
}

export function ApiKeyChatCard({ result }: ApiKeyChatCardProps) {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    if (!result.api_key) return
    navigator.clipboard.writeText(result.api_key).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // --- GENERATED / REGENERATED ---
  if (result.action === "generated" || result.action === "regenerated") {
    return (
      <figure
        role="region"
        aria-label="API key generated"
        className="rounded-lg border border-amber-300 bg-amber-50/40 overflow-hidden"
      >
        <Card className="border-0 bg-transparent shadow-none">
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-sm font-semibold text-amber-800 flex items-center gap-2">
              <span aria-hidden="true">🔑</span>
              API Key {result.action === "regenerated" ? "Regenerated" : "Generated"}
              <Badge className="ml-auto text-xs bg-amber-100 text-amber-800 border-amber-300">
                Protected
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3 space-y-3">
            <p className="text-sm text-gray-700">
              Your prediction endpoint now requires an{" "}
              <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">
                Authorization: Bearer &lt;key&gt;
              </code>{" "}
              header.
            </p>

            {result.api_key && (
              <div
                role="alert"
                className="rounded-md border border-amber-400 bg-amber-100 px-3 py-2 space-y-1"
              >
                <p className="text-xs font-semibold text-amber-900">
                  API Key — shown once, store it now
                </p>
                <div className="flex items-center gap-2">
                  <code
                    className="text-xs break-all text-amber-900 flex-1"
                    data-testid="api-key-value"
                  >
                    {result.api_key}
                  </code>
                  <Button
                    size="sm"
                    variant="outline"
                    className="shrink-0 text-xs h-6 px-2 border-amber-500 text-amber-900 hover:bg-amber-200"
                    onClick={handleCopy}
                    aria-label="Copy API key to clipboard"
                  >
                    {copied ? "Copied!" : "Copy"}
                  </Button>
                </div>
                <p className="text-xs text-amber-800">
                  This key cannot be retrieved again. Share it with authorised
                  users and add it to your integration code.
                </p>
              </div>
            )}

            <div className="text-xs text-gray-600 space-y-1">
              <p>
                <span className="font-medium">Example usage: </span>
                <code className="bg-gray-100 px-1 py-0.5 rounded">
                  curl -H &quot;Authorization: Bearer &lt;key&gt;&quot; ...
                </code>
              </p>
              <p className="text-gray-500">
                Say &ldquo;remove API key protection&rdquo; to make the endpoint
                public again, or &ldquo;regenerate my API key&rdquo; to issue a
                new key.
              </p>
            </div>
          </CardContent>
        </Card>
      </figure>
    )
  }

  // --- DISABLED ---
  if (result.action === "disabled") {
    return (
      <figure
        role="region"
        aria-label="API key protection removed"
        className="rounded-lg border border-slate-300 bg-slate-50/40 overflow-hidden"
      >
        <Card className="border-0 bg-transparent shadow-none">
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-sm font-semibold text-slate-700 flex items-center gap-2">
              <span aria-hidden="true">🔓</span>
              API Key Protection Removed
              <Badge className="ml-auto text-xs bg-slate-200 text-slate-700 border-slate-300">
                Open Access
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3 space-y-2">
            <p className="text-sm text-gray-700">
              Your prediction endpoint is now publicly accessible — no API key
              required.
            </p>
            <p className="text-xs text-gray-500">
              Say &ldquo;generate an API key&rdquo; to re-enable protection.
            </p>
          </CardContent>
        </Card>
      </figure>
    )
  }

  // --- STATUS ---
  const isProtected = result.is_protected
  const borderClass = isProtected
    ? "border-amber-300 bg-amber-50/30"
    : "border-slate-300 bg-slate-50/30"
  const titleClass = isProtected ? "text-amber-800" : "text-slate-700"

  return (
    <figure
      role="region"
      aria-label="API key status"
      className={`rounded-lg border overflow-hidden ${borderClass}`}
    >
      <Card className="border-0 bg-transparent shadow-none">
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className={`text-sm font-semibold flex items-center gap-2 ${titleClass}`}>
            <span aria-hidden="true">{isProtected ? "🔑" : "🔓"}</span>
            API Key Status
            <Badge
              className={`ml-auto text-xs ${
                isProtected
                  ? "bg-amber-100 text-amber-800 border-amber-300"
                  : "bg-slate-200 text-slate-700 border-slate-300"
              }`}
            >
              {isProtected ? "Protected" : "Open Access"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3 space-y-2">
          <p className="text-sm text-gray-700">{result.summary}</p>
          <p className="text-xs text-gray-500">
            {isProtected
              ? 'Say "regenerate my API key" to issue a new key, or "remove API key protection" to open the endpoint.'
              : 'Say "generate an API key" to protect your prediction endpoint.'}
          </p>
        </CardContent>
      </Card>
    </figure>
  )
}
