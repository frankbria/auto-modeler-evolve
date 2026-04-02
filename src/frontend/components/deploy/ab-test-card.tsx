"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { ABTest } from "@/lib/types"

// ---------------------------------------------------------------------------
// ABTestCard — champion-challenger A/B test management
//
// Allows analysts to split live prediction traffic between the current
// (champion) model and a newly trained (challenger) model so they can
// measure real-world performance differences before committing.
// ---------------------------------------------------------------------------

interface ABTestCardProps {
  deploymentId: string
  /** Called after a successful promote so the parent can refresh the deployment */
  onPromoted?: () => void
}

export function ABTestCard({ deploymentId, onPromoted }: ABTestCardProps) {
  const [test, setTest] = useState<ABTest | null>(null)
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [challengerId, setChallengerId] = useState("")
  const [splitPct, setSplitPct] = useState(80)
  const [saving, setSaving] = useState(false)
  const [confirmPromote, setConfirmPromote] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadTest = useCallback(() => {
    api.deploy
      .getAbTest(deploymentId)
      .then(setTest)
      .catch(() => setTest(null))
      .finally(() => setLoading(false))
  }, [deploymentId])

  useEffect(() => {
    loadTest()
  }, [loadTest])

  async function handleCreate() {
    if (!challengerId.trim()) return
    setSaving(true)
    setError(null)
    try {
      const created = await api.deploy.createAbTest(
        deploymentId,
        challengerId.trim(),
        splitPct
      )
      setTest(created)
      setShowForm(false)
      setChallengerId("")
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start A/B test")
    } finally {
      setSaving(false)
    }
  }

  async function handleEnd() {
    setError(null)
    try {
      await api.deploy.endAbTest(deploymentId)
      setTest(null)
    } catch {
      setError("Failed to end A/B test")
    }
  }

  async function handlePromote() {
    setError(null)
    try {
      await api.deploy.promoteChallenger(deploymentId)
      setTest(null)
      setConfirmPromote(false)
      onPromoted?.()
    } catch {
      setError("Failed to promote challenger")
    }
  }

  if (loading) return null

  return (
    <Card className="border-purple-500/30">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold flex items-center gap-2">
          <span aria-hidden="true">⚗️</span> Champion-Challenger A/B Test
          {test && test.is_active && (
            <Badge
              className="ml-auto text-xs bg-purple-100 text-purple-800 border-purple-300"
              variant="outline"
            >
              Live
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4 text-sm">
        {!test && !showForm && (
          <div className="space-y-2">
            <p className="text-muted-foreground text-xs">
              Split live prediction traffic between this model (champion) and a
              newly trained one (challenger) to measure real-world performance
              before committing.
            </p>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setShowForm(true)}
            >
              Start A/B Test
            </Button>
          </div>
        )}

        {!test && showForm && (
          <div className="space-y-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">
                Challenger Deployment ID
              </label>
              <input
                type="text"
                className="w-full rounded border border-input bg-background px-2 py-1 text-xs placeholder:text-muted-foreground"
                placeholder="Paste the deployment ID of the challenger model"
                value={challengerId}
                onChange={(e) => setChallengerId(e.target.value)}
                data-testid="challenger-id-input"
              />
              <p className="text-xs text-muted-foreground">
                Find the ID in the challenger&apos;s deployment panel URL.
              </p>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">
                Champion traffic share:{" "}
                <span className="font-semibold text-foreground">{splitPct}%</span>
              </label>
              <input
                type="range"
                min={50}
                max={99}
                step={5}
                value={splitPct}
                onChange={(e) => setSplitPct(Number(e.target.value))}
                className="w-full"
                aria-label={`Champion traffic share: ${splitPct}%`}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Champion {splitPct}%</span>
                <span>Challenger {100 - splitPct}%</span>
              </div>
            </div>

            {error && (
              <p className="text-xs text-destructive" role="alert">
                {error}
              </p>
            )}

            <div className="flex gap-2">
              <Button size="sm" onClick={handleCreate} disabled={saving}>
                {saving ? "Starting…" : "Start Test"}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setShowForm(false)
                  setError(null)
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        {test && test.is_active && (
          <div className="space-y-4">
            {/* Split visualiser */}
            <div>
              <div className="flex justify-between text-xs mb-1 font-medium">
                <span>Champion ({test.champion_algorithm ?? "Model"})</span>
                <span>Challenger ({test.challenger_algorithm ?? "Model"})</span>
              </div>
              <div className="flex h-2 rounded overflow-hidden">
                <div
                  className="bg-purple-500"
                  style={{ width: `${test.champion_split_pct}%` }}
                  title={`Champion: ${test.champion_split_pct}%`}
                />
                <div
                  className="bg-amber-400"
                  style={{ width: `${test.challenger_split_pct}%` }}
                  title={`Challenger: ${test.challenger_split_pct}%`}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground mt-0.5">
                <span>{test.champion_split_pct}% of traffic</span>
                <span>{test.challenger_split_pct}% of traffic</span>
              </div>
            </div>

            {/* Per-variant metrics */}
            <div className="grid grid-cols-2 gap-2">
              <VariantMetricsBox
                label="Champion"
                color="purple"
                metrics={test.champion_metrics}
              />
              <VariantMetricsBox
                label="Challenger"
                color="amber"
                metrics={test.challenger_metrics}
              />
            </div>

            {/* Significance */}
            <SignificanceBadge significance={test.significance} />

            {error && (
              <p className="text-xs text-destructive" role="alert">
                {error}
              </p>
            )}

            {/* Actions */}
            {confirmPromote ? (
              <div className="rounded bg-amber-50 border border-amber-200 p-3 space-y-2 text-xs text-amber-800">
                <p>
                  <strong>Promote challenger?</strong> The challenger&apos;s model
                  will replace the champion. Your prediction URL stays the same.
                </p>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    className="bg-amber-600 hover:bg-amber-700 text-white"
                    onClick={handlePromote}
                  >
                    Yes, promote
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setConfirmPromote(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex gap-2 flex-wrap">
                <Button
                  size="sm"
                  className="bg-amber-600 hover:bg-amber-700 text-white"
                  onClick={() => setConfirmPromote(true)}
                >
                  Promote Challenger
                </Button>
                <Button size="sm" variant="outline" onClick={handleEnd}>
                  End Test
                </Button>
                <Button size="sm" variant="ghost" onClick={loadTest}>
                  Refresh
                </Button>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface VariantMetricsBoxProps {
  label: string
  color: "purple" | "amber"
  metrics: {
    request_count: number
    avg_confidence: number | null
    p95_ms: number | null
    avg_prediction: number | null
  }
}

function VariantMetricsBox({ label, color, metrics }: VariantMetricsBoxProps) {
  const border =
    color === "purple" ? "border-purple-200" : "border-amber-200"
  const bg = color === "purple" ? "bg-purple-50" : "bg-amber-50"
  const text = color === "purple" ? "text-purple-700" : "text-amber-700"

  return (
    <div className={`rounded border ${border} ${bg} p-2 space-y-1`}>
      <p className={`text-xs font-semibold ${text}`}>{label}</p>
      <dl className="space-y-0.5 text-xs">
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Requests</dt>
          <dd className="font-medium">{metrics.request_count}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Avg confidence</dt>
          <dd className="font-medium">
            {metrics.avg_confidence != null
              ? `${(metrics.avg_confidence * 100).toFixed(1)}%`
              : "—"}
          </dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">p95 latency</dt>
          <dd className="font-medium">
            {metrics.p95_ms != null ? `${metrics.p95_ms}ms` : "—"}
          </dd>
        </div>
        {metrics.avg_prediction != null && (
          <div className="flex justify-between">
            <dt className="text-muted-foreground">Avg prediction</dt>
            <dd className="font-medium">{metrics.avg_prediction.toFixed(2)}</dd>
          </div>
        )}
      </dl>
    </div>
  )
}

interface SignificanceBadgeProps {
  significance: {
    significant: boolean
    p_value: number | null
    note: string
  }
}

function SignificanceBadge({ significance }: SignificanceBadgeProps) {
  if (significance.p_value == null) {
    return (
      <p className="text-xs text-muted-foreground italic">{significance.note}</p>
    )
  }
  return (
    <div className="flex items-center gap-2 text-xs">
      <Badge
        variant="outline"
        className={
          significance.significant
            ? "bg-green-50 text-green-700 border-green-300"
            : "bg-muted text-muted-foreground"
        }
      >
        {significance.significant ? "Statistically significant" : "Not yet significant"}
      </Badge>
      <span className="text-muted-foreground">
        p = {significance.p_value} · {significance.note}
      </span>
    </div>
  )
}
