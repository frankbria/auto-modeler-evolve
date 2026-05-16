"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { AlertRuleEntry, AlertRuleEventResult } from "@/lib/types"

function formatRelative(iso: string | null): string {
  if (!iso) return "never"
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function RuleRow({ rule }: { rule: AlertRuleEntry }) {
  return (
    <li
      role="listitem"
      aria-label={`Alert rule: ${rule.name}`}
      className="py-2 border-b border-gray-100 last:border-0 space-y-1"
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-medium text-gray-800">{rule.name}</span>
        <Badge className="text-xs bg-violet-100 text-violet-800 border-violet-300 shrink-0">
          {rule.trigger_count} fired
        </Badge>
      </div>
      <p className="text-xs text-gray-600">{rule.description}</p>
      <p className="text-xs text-gray-400">
        Last triggered: {formatRelative(rule.last_triggered_at)}
      </p>
    </li>
  )
}

interface AlertRuleCardProps {
  result: AlertRuleEventResult
}

export function AlertRuleCard({ result }: AlertRuleCardProps) {
  if (result.action === "created") {
    return (
      <figure
        role="region"
        aria-label="Alert rule created"
        className="rounded-lg border border-violet-300 bg-violet-50/40 overflow-hidden"
      >
        <Card className="border-0 bg-transparent shadow-none">
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-sm font-semibold text-violet-800 flex items-center gap-2">
              <span aria-hidden="true">🔔</span>
              Alert Rule Created
              <Badge className="ml-auto text-xs bg-violet-100 text-violet-800 border-violet-300">
                Active
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3 space-y-2">
            <p className="text-sm font-medium text-gray-800">{result.name}</p>
            {result.description && (
              <p className="text-xs text-gray-600">{result.description}</p>
            )}
            <p className="text-xs text-gray-500">{result.summary}</p>
            <p className="text-xs text-gray-400">
              Say &ldquo;list my alert rules&rdquo; to see all active rules.
            </p>
          </CardContent>
        </Card>
        <figcaption className="sr-only">
          {result.name}: {result.description}
        </figcaption>
      </figure>
    )
  }

  if (result.action === "list") {
    const rules = result.rules ?? []
    return (
      <figure
        role="region"
        aria-label="Active alert rules"
        className="rounded-lg border border-slate-300 bg-slate-50/40 overflow-hidden"
      >
        <Card className="border-0 bg-transparent shadow-none">
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-sm font-semibold text-slate-800 flex items-center gap-2">
              <span aria-hidden="true">🔔</span>
              Alert Rules
              <Badge className="ml-auto text-xs bg-slate-200 text-slate-700 border-slate-300">
                {result.count ?? rules.length}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3">
            {rules.length === 0 ? (
              <p className="text-sm text-gray-500 italic">
                No alert rules active. Say &ldquo;alert me when predicted value
                is below 50&rdquo; to create one.
              </p>
            ) : (
              <ul role="list" aria-label="Alert rules" className="divide-y divide-gray-100">
                {rules.map((rule) => (
                  <RuleRow key={rule.id} rule={rule} />
                ))}
              </ul>
            )}
            <p className="text-xs text-gray-400 mt-2">{result.summary}</p>
          </CardContent>
        </Card>
        <figcaption className="sr-only">{result.summary}</figcaption>
      </figure>
    )
  }

  if (result.action === "deleted") {
    return (
      <figure
        role="region"
        aria-label="Alert rules deleted"
        className="rounded-lg border border-rose-300 bg-rose-50/40 overflow-hidden"
      >
        <Card className="border-0 bg-transparent shadow-none">
          <CardHeader className="pb-2 pt-3 px-4">
            <CardTitle className="text-sm font-semibold text-rose-800 flex items-center gap-2">
              <span aria-hidden="true">🗑️</span>
              Alert Rules Removed
              <Badge className="ml-auto text-xs bg-rose-100 text-rose-800 border-rose-300">
                {result.deleted_count ?? 0} removed
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-3 space-y-1">
            {result.deleted_names && result.deleted_names.length > 0 && (
              <ul className="text-xs text-gray-600 list-disc list-inside">
                {result.deleted_names.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
            <p className="text-xs text-gray-500">{result.summary}</p>
          </CardContent>
        </Card>
        <figcaption className="sr-only">{result.summary}</figcaption>
      </figure>
    )
  }

  return null
}
