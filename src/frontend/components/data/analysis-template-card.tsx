"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { TemplateSavedInfo, TemplateListInfo, TemplateReplayInfo } from "@/lib/types"

// ---------------------------------------------------------------------------
// TemplateSavedCard — confirmation after saving an analysis template
// ---------------------------------------------------------------------------

interface TemplateSavedCardProps {
  info: TemplateSavedInfo
}

export function TemplateSavedCard({ info }: TemplateSavedCardProps) {
  return (
    <figure aria-label={`Analysis template saved: ${info.name}`}>
      <Card className="border-emerald-200">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <span aria-hidden="true">💾</span>
            <CardTitle className="text-sm">Template Saved</CardTitle>
            <Badge className="bg-emerald-100 text-emerald-800 border-emerald-200 text-xs ml-auto">
              {info.query_count} {info.query_count === 1 ? "query" : "queries"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="rounded-md bg-emerald-50 border border-emerald-200 p-3">
            <p className="text-xs text-muted-foreground mb-1">Template name</p>
            <p className="text-sm font-semibold text-emerald-900">{info.name}</p>
          </div>

          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Saved queries ({info.query_count})</p>
            <ul className="space-y-1">
              {info.queries.map((q, i) => (
                <li key={i} className="text-xs text-foreground bg-muted rounded px-2 py-1 truncate">
                  {i + 1}. {q}
                </li>
              ))}
            </ul>
          </div>

          <p className="text-xs text-muted-foreground">
            Replay anytime by saying &ldquo;replay my <strong>{info.name}</strong> template&rdquo;
          </p>
        </CardContent>
      </Card>
    </figure>
  )
}

// ---------------------------------------------------------------------------
// TemplateListCard — shows all saved templates for a project
// ---------------------------------------------------------------------------

interface TemplateListCardProps {
  info: TemplateListInfo
  onReplay?: (templateName: string) => void
}

export function TemplateListCard({ info, onReplay }: TemplateListCardProps) {
  if (info.count === 0) {
    return (
      <figure aria-label="No saved analysis templates">
        <Card className="border-gray-200">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <span aria-hidden="true">📋</span>
              <CardTitle className="text-sm">Analysis Templates</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">
              No templates saved yet. Say &ldquo;save this analysis as a template called [name]&rdquo; to create one.
            </p>
          </CardContent>
        </Card>
      </figure>
    )
  }

  return (
    <figure aria-label={`${info.count} saved analysis templates`}>
      <Card className="border-blue-200">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <span aria-hidden="true">📋</span>
            <CardTitle className="text-sm">Analysis Templates</CardTitle>
            <Badge className="bg-blue-100 text-blue-800 border-blue-200 text-xs ml-auto">
              {info.count} saved
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          {info.templates.map((t) => (
            <div
              key={t.id}
              className="flex items-center justify-between rounded-md border border-muted bg-muted/30 px-3 py-2"
            >
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium truncate">{t.name}</p>
                <p className="text-xs text-muted-foreground">
                  {t.query_count} {t.query_count === 1 ? "query" : "queries"}
                </p>
              </div>
              {onReplay && (
                <Button
                  size="sm"
                  variant="outline"
                  className="ml-2 text-xs shrink-0"
                  onClick={() => onReplay(t.name)}
                  aria-label={`Replay ${t.name} template`}
                >
                  Replay
                </Button>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    </figure>
  )
}

// ---------------------------------------------------------------------------
// TemplateReplayCard — shows template queries as clickable chips
// ---------------------------------------------------------------------------

interface TemplateReplayCardProps {
  info: TemplateReplayInfo
  onQueryClick?: (query: string) => void
}

export function TemplateReplayCard({ info, onQueryClick }: TemplateReplayCardProps) {
  return (
    <figure aria-label={`Replay template: ${info.name}`}>
      <Card className="border-purple-200">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <span aria-hidden="true">▶️</span>
            <CardTitle className="text-sm">Replay: {info.name}</CardTitle>
            <Badge className="bg-purple-100 text-purple-800 border-purple-200 text-xs ml-auto">
              {info.query_count} {info.query_count === 1 ? "query" : "queries"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Click each query to re-run it on your current data:
          </p>
          <div className="flex flex-col gap-1">
            {info.queries.map((q, i) => (
              <button
                key={i}
                className="text-left text-xs rounded-md border border-purple-200 bg-purple-50 hover:bg-purple-100 px-3 py-2 transition-colors text-purple-900 w-full"
                onClick={() => onQueryClick?.(q)}
                aria-label={`Run query: ${q}`}
              >
                <span className="text-purple-400 mr-2 font-mono">{i + 1}.</span>
                {q}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </figure>
  )
}
