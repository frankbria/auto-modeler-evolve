"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { IntegrationSnippets } from "@/lib/types"

// ---------------------------------------------------------------------------
// IntegrationCard — developer handoff: ready-to-paste code snippets
// ---------------------------------------------------------------------------

interface IntegrationCardProps {
  deploymentId: string
}

type Tab = "curl" | "python" | "javascript"

export function IntegrationCard({ deploymentId }: IntegrationCardProps) {
  const [snippets, setSnippets] = useState<IntegrationSnippets | null>(null)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState<Tab>("curl")
  const [copied, setCopied] = useState<Tab | null>(null)

  const loadSnippets = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.deploy.getIntegration(deploymentId)
      setSnippets(data)
    } catch {
      // silently fail — card stays hidden
    } finally {
      setLoading(false)
    }
  }, [deploymentId])

  useEffect(() => {
    if (expanded && !snippets) {
      loadSnippets()
    }
  }, [expanded, snippets, loadSnippets])

  async function copySnippet(tab: Tab) {
    if (!snippets) return
    const text = snippets[tab]
    try {
      await navigator.clipboard.writeText(text)
      setCopied(tab)
      setTimeout(() => setCopied(null), 2000)
    } catch {
      // clipboard not available (e.g. test env)
    }
  }

  const tabLabel: Record<Tab, string> = {
    curl: "curl",
    python: "Python",
    javascript: "JavaScript",
  }

  const currentSnippet = snippets ? snippets[activeTab] : ""

  return (
    <Card data-testid="integration-card">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Developer Integration</CardTitle>
          <div className="flex items-center gap-2">
            <Badge className="bg-blue-100 text-blue-800 border-blue-200 text-xs">
              API snippets
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              className="text-xs h-6 px-2"
              onClick={() => setExpanded(!expanded)}
              data-testid="integration-toggle"
            >
              {expanded ? "Hide" : "Show code"}
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Share with your developer to integrate predictions into reporting tools.
        </p>
      </CardHeader>

      {expanded && (
        <CardContent className="space-y-3" data-testid="integration-content">
          {loading && (
            <p className="text-xs text-muted-foreground animate-pulse">
              Generating code snippets…
            </p>
          )}

          {snippets && (
            <>
              {/* Endpoint info */}
              <div className="rounded-md bg-muted/50 border px-3 py-2 space-y-1">
                <p className="text-xs text-muted-foreground">Prediction endpoint</p>
                <p
                  className="text-xs font-mono text-foreground break-all"
                  data-testid="endpoint-url"
                >
                  {snippets.endpoint_url}
                </p>
                {snippets.target_column && (
                  <p className="text-xs text-muted-foreground">
                    Predicts: <span className="font-medium">{snippets.target_column}</span>
                    {snippets.algorithm && (
                      <> · Model: <span className="font-medium">{snippets.algorithm.replace(/_/g, " ")}</span></>
                    )}
                  </p>
                )}
              </div>

              {/* Tab bar */}
              <div className="flex gap-1" role="tablist">
                {(["curl", "python", "javascript"] as Tab[]).map((tab) => (
                  <button
                    key={tab}
                    role="tab"
                    aria-selected={activeTab === tab}
                    onClick={() => setActiveTab(tab)}
                    data-testid={`tab-${tab}`}
                    className={`text-xs px-3 py-1 rounded-md border transition-colors ${
                      activeTab === tab
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-background text-muted-foreground border-border hover:border-primary/50"
                    }`}
                  >
                    {tabLabel[tab]}
                  </button>
                ))}
              </div>

              {/* Code block */}
              <div className="relative">
                <pre
                  className="text-xs font-mono bg-zinc-900 text-zinc-100 rounded-md p-3 overflow-x-auto whitespace-pre-wrap leading-relaxed"
                  data-testid="code-block"
                >
                  {currentSnippet}
                </pre>
                <Button
                  size="sm"
                  variant="secondary"
                  className="absolute top-2 right-2 text-xs h-6 px-2"
                  onClick={() => copySnippet(activeTab)}
                  data-testid="copy-button"
                >
                  {copied === activeTab ? "Copied!" : "Copy"}
                </Button>
              </div>

              {/* Batch prediction note */}
              <div className="rounded-md bg-blue-50 border border-blue-200 p-3 space-y-1">
                <p className="text-xs font-medium text-blue-800">Batch predictions</p>
                <p className="text-xs text-blue-700 font-mono break-all">
                  {snippets.batch_note}
                </p>
              </div>

              {/* OpenAPI docs link */}
              <p className="text-xs text-muted-foreground">
                Full API docs:{" "}
                <span className="font-mono text-primary">{snippets.openapi_url}</span>
              </p>
            </>
          )}
        </CardContent>
      )}
    </Card>
  )
}
