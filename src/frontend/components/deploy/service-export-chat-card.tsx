"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ServiceExportChatResult } from "@/lib/types"

interface ServiceExportChatCardProps {
  result: ServiceExportChatResult
}

export function ServiceExportChatCard({ result }: ServiceExportChatCardProps) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? ""
  const downloadUrl = `${apiUrl}${result.download_url}`

  const algoLabel = result.algorithm
    ? result.algorithm.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
    : "Model"

  return (
    <Card
      data-testid="service-export-chat-card"
      role="region"
      aria-label="Model service export"
      className="border-indigo-500/30"
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <span aria-hidden="true">📦</span> Model Package Ready
          </CardTitle>
          <div className="flex gap-1.5">
            <Badge className="bg-indigo-100 text-indigo-800 border-indigo-200 text-xs">
              ZIP download
            </Badge>
            <Badge variant="outline" className="text-xs capitalize">
              {result.problem_type}
            </Badge>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Your{" "}
          <span className="font-medium text-foreground">{algoLabel}</span> model
          (predicts <span className="font-medium text-foreground">{result.target_column}</span>)
          is packaged as a self-contained service your developer can run with one command.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Included files */}
        <div className="rounded-md bg-muted/50 border px-3 py-2 space-y-1">
          <p className="text-xs font-medium text-foreground">What&apos;s included</p>
          <ul className="text-xs text-muted-foreground space-y-0.5" data-testid="included-files">
            {result.included_files.map((file) => (
              <li key={file}>
                <span className="font-mono">{file}</span>
                {file === "server.py" && " — FastAPI prediction server"}
                {file === "model_pipeline.joblib" && " — preprocessing pipeline"}
                {file === "model.joblib" && ` — trained ${algoLabel.toLowerCase()}`}
                {file === "requirements.txt" && " — Python dependencies"}
                {file === "README.md" && " — setup instructions"}
              </li>
            ))}
          </ul>
        </div>

        {/* Quick start */}
        <div
          className="rounded-md bg-zinc-900 text-zinc-100 px-3 py-2 space-y-1"
          data-testid="quickstart-block"
        >
          <p className="text-xs text-zinc-400 mb-1">Quick start (after unzipping)</p>
          <pre className="text-xs font-mono leading-relaxed whitespace-pre">
            {`pip install -r requirements.txt\nuvicorn server:app --host 0.0.0.0 --port 8000`}
          </pre>
        </div>

        {result.feature_count > 0 && (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">{result.feature_count}</span>{" "}
            feature{result.feature_count !== 1 ? "s" : ""} included in the model input schema.
          </p>
        )}

        {/* Download link */}
        <a
          href={downloadUrl}
          download
          data-testid="service-export-download-link"
          aria-label={`Download ${algoLabel} model service as ZIP`}
          className="flex w-full items-center justify-center rounded-md bg-indigo-600 hover:bg-indigo-700 px-4 py-2 text-sm font-medium text-white transition-colors"
        >
          Download as ZIP
        </a>
      </CardContent>
    </Card>
  )
}
