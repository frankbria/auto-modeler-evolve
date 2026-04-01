"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// ---------------------------------------------------------------------------
// ExportServiceCard — download the model as a self-contained FastAPI service
// ---------------------------------------------------------------------------

interface ExportServiceCardProps {
  deploymentId: string
  algorithm?: string | null
  targetColumn?: string | null
}

export function ExportServiceCard({
  deploymentId,
  algorithm,
  targetColumn,
}: ExportServiceCardProps) {
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleDownload() {
    setDownloading(true)
    setError(null)
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? ""
      const resp = await fetch(`${apiUrl}/api/deploy/${deploymentId}/export`)
      if (!resp.ok) {
        setError("Export failed. Please try again.")
        return
      }
      const blob = await resp.blob()
      const disposition = resp.headers.get("content-disposition") ?? ""
      const match = disposition.match(/filename="([^"]+)"/)
      const filename = match ? match[1] : `automodeler_${deploymentId}.zip`

      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch {
      setError("Export failed. Please try again.")
    } finally {
      setDownloading(false)
    }
  }

  return (
    <Card data-testid="export-service-card" className="border-emerald-500/30">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <span aria-hidden="true">📦</span> Export as Service
          </CardTitle>
          <Badge className="bg-emerald-100 text-emerald-800 border-emerald-200 text-xs">
            ZIP download
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground">
          Download the model as a standalone FastAPI service — your developer can run it
          with a single command on any server.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* What's included */}
        <div className="rounded-md bg-muted/50 border px-3 py-2 space-y-1">
          <p className="text-xs font-medium text-foreground">What&apos;s included</p>
          <ul className="text-xs text-muted-foreground space-y-0.5">
            <li>
              <span className="font-mono">server.py</span> — FastAPI prediction server
            </li>
            <li>
              <span className="font-mono">model_pipeline.joblib</span> — preprocessing pipeline
            </li>
            <li>
              <span className="font-mono">model.joblib</span> — trained{" "}
              {algorithm ? (
                <span className="font-medium">
                  {algorithm.replace(/_/g, " ")}
                </span>
              ) : (
                "model"
              )}
            </li>
            <li>
              <span className="font-mono">requirements.txt</span> — Python dependencies
            </li>
            <li>
              <span className="font-mono">README.md</span> — setup instructions
            </li>
          </ul>
        </div>

        {/* Quick start snippet */}
        <div className="rounded-md bg-zinc-900 text-zinc-100 px-3 py-2 space-y-1">
          <p className="text-xs text-zinc-400 mb-1">Quick start (after unzipping)</p>
          <pre className="text-xs font-mono leading-relaxed whitespace-pre">
            {`pip install -r requirements.txt\nuvicorn server:app --host 0.0.0.0 --port 8000`}
          </pre>
        </div>

        {targetColumn && (
          <p className="text-xs text-muted-foreground">
            Predicts:{" "}
            <span
              className="font-medium text-foreground"
              data-testid="export-target-column"
            >
              {targetColumn}
            </span>
            {algorithm && (
              <>
                {" "}
                · Algorithm:{" "}
                <span className="font-medium text-foreground">
                  {algorithm.replace(/_/g, " ")}
                </span>
              </>
            )}
          </p>
        )}

        {error && (
          <p className="text-xs text-red-600" data-testid="export-error">
            {error}
          </p>
        )}

        <Button
          size="sm"
          className="w-full bg-emerald-600 hover:bg-emerald-700 text-white"
          onClick={handleDownload}
          disabled={downloading}
          data-testid="export-download-button"
          aria-label="Download model as self-contained ZIP service"
        >
          {downloading ? "Preparing download…" : "Download as ZIP"}
        </Button>
      </CardContent>
    </Card>
  )
}
