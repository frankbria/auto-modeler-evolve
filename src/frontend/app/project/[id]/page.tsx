"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useParams } from "next/navigation"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { ChartMessage } from "@/components/chat/chart-message"
import { api } from "@/lib/api"
import { useAppStore } from "@/lib/store"
import type { Dataset, DataInsight } from "@/lib/types"

const WELCOME_MESSAGE =
  "Hi! I'm your data modeling assistant. Upload a CSV file to get started, or ask me anything about your data."

export default function ProjectWorkspace() {
  const params = useParams<{ id: string }>()
  const projectId = params.id

  const {
    currentProject,
    setCurrentProject,
    currentDataset,
    dataPreview,
    columnStats,
    dataInsights,
    setDataset,
    messages,
    addMessage,
    setMessages,
    isStreaming,
    setStreaming,
    appendToLastMessage,
    attachChartToLastMessage,
  } = useAppStore()

  const [chatInput, setChatInput] = useState("")
  const [uploading, setUploading] = useState(false)
  const [loadingProject, setLoadingProject] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    async function load() {
      try {
        const [project, history] = await Promise.all([
          api.projects.get(projectId),
          api.chat.history(projectId),
        ])
        setCurrentProject(project)
        if (history?.messages && history.messages.length > 0) {
          setMessages(history.messages)
        } else {
          setMessages([
            {
              role: "assistant",
              content: WELCOME_MESSAGE,
              timestamp: new Date().toISOString(),
            },
          ])
        }
      } catch {
        setMessages([
          {
            role: "assistant",
            content: WELCOME_MESSAGE,
            timestamp: new Date().toISOString(),
          },
        ])
      } finally {
        setLoadingProject(false)
      }
    }
    load()
  }, [projectId, setCurrentProject, setMessages])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSendMessage = useCallback(async () => {
    const text = chatInput.trim()
    if (!text || isStreaming) return

    setChatInput("")
    addMessage({
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    })
    addMessage({
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
    })
    setStreaming(true)

    try {
      const response = await api.chat.send(projectId, text)
      const reader = response.body?.getReader()
      if (!reader) {
        setStreaming(false)
        return
      }

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split("\n\n")
        buffer = parts.pop() ?? ""

        for (const part of parts) {
          const trimmed = part.trim()
          if (trimmed.startsWith("data: ")) {
            try {
              const json = JSON.parse(trimmed.slice(6))
              if (json.type === "token") {
                appendToLastMessage(json.content)
              } else if (json.type === "chart" && json.chart) {
                attachChartToLastMessage(json.chart)
              } else if (json.type === "done") {
                setStreaming(false)
              }
            } catch {
              // skip malformed JSON
            }
          }
        }
      }
    } catch {
      appendToLastMessage("\n\n[Connection error. Please try again.]")
    } finally {
      setStreaming(false)
    }
  }, [
    chatInput,
    isStreaming,
    projectId,
    addMessage,
    setStreaming,
    appendToLastMessage,
    attachChartToLastMessage,
  ])

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (!file) return

      setUploading(true)
      try {
        const result = await api.data.upload(projectId, file)
        const dataset: Dataset = {
          id: result.dataset_id,
          project_id: projectId,
          filename: result.filename,
          row_count: result.row_count,
          column_count: result.column_count,
          uploaded_at: new Date().toISOString(),
        }
        setDataset(dataset, result.preview, result.column_stats, result.insights)

        // Surface upload insights in chat
        if (result.insights && result.insights.length > 0) {
          const insightLines = result.insights
            .slice(0, 3)
            .map((i: DataInsight) => `- ${i.title}: ${i.detail}`)
            .join("\n")
          addMessage({
            role: "assistant",
            content: `I've analyzed **${result.filename}** (${result.row_count.toLocaleString()} rows, ${result.column_count} columns). Here's what I noticed:\n\n${insightLines}\n\nWhat would you like to explore?`,
            timestamp: new Date().toISOString(),
          })
        }
      } catch {
        addMessage({
          role: "assistant",
          content:
            "There was a problem uploading your file. Please make sure it is a valid CSV and try again.",
          timestamp: new Date().toISOString(),
        })
      } finally {
        setUploading(false)
      }
    },
    [projectId, setDataset, addMessage]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
    disabled: uploading,
  })

  if (loadingProject) {
    return (
      <div className="flex h-[calc(100vh-3rem)] items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading project...</p>
      </div>
    )
  }

  return (
    <div className="flex h-[calc(100vh-3rem)]">
      {/* Chat Panel */}
      <div className="flex w-2/5 flex-col border-r">
        <div className="border-b px-4 py-3">
          <h2 className="text-sm font-semibold">
            {currentProject?.name ?? "Chat"}
          </h2>
        </div>

        <ScrollArea className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-3 p-4">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[90%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                    msg.role === "user"
                      ? "bg-muted text-foreground"
                      : "border bg-card text-card-foreground"
                  }`}
                >
                  {msg.content}
                  {isStreaming &&
                    i === messages.length - 1 &&
                    msg.role === "assistant" &&
                    msg.content === "" && (
                      <span className="inline-flex gap-1">
                        <span className="animate-pulse">.</span>
                        <span className="animate-pulse delay-100">.</span>
                        <span className="animate-pulse delay-200">.</span>
                      </span>
                    )}
                  {msg.chart && <ChartMessage spec={msg.chart} />}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <div className="border-t p-3">
          <div className="flex gap-2">
            <Input
              placeholder="Ask about your data..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
              disabled={isStreaming}
            />
            <Button
              onClick={handleSendMessage}
              disabled={isStreaming || !chatInput.trim()}
            >
              Send
            </Button>
          </div>
        </div>
      </div>

      {/* Data Panel */}
      <div className="flex w-3/5 flex-col overflow-hidden">
        {currentDataset ? (
          <DataPreviewPanel
            filename={currentDataset.filename}
            rowCount={currentDataset.row_count}
            columnCount={currentDataset.column_count}
            preview={dataPreview}
            stats={columnStats}
            insights={dataInsights}
          />
        ) : (
          <UploadPanel
            getRootProps={getRootProps}
            getInputProps={getInputProps}
            isDragActive={isDragActive}
            uploading={uploading}
          />
        )}
      </div>
    </div>
  )
}

function UploadPanel({
  getRootProps,
  getInputProps,
  isDragActive,
  uploading,
}: {
  getRootProps: ReturnType<typeof useDropzone>["getRootProps"]
  getInputProps: ReturnType<typeof useDropzone>["getInputProps"]
  isDragActive: boolean
  uploading: boolean
}) {
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div
        {...getRootProps()}
        className={`flex h-64 w-full max-w-md cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors ${
          isDragActive
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
        } ${uploading ? "pointer-events-none opacity-50" : ""}`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <p className="text-sm text-muted-foreground">Uploading...</p>
        ) : isDragActive ? (
          <p className="text-sm font-medium">Drop your CSV here</p>
        ) : (
          <>
            <p className="text-sm font-medium">Drop your CSV here</p>
            <p className="mt-1 text-xs text-muted-foreground">
              or click to browse
            </p>
          </>
        )}
      </div>
    </div>
  )
}

function DataPreviewPanel({
  filename,
  rowCount,
  columnCount,
  preview,
  stats,
  insights,
}: {
  filename: string
  rowCount: number
  columnCount: number
  preview: Record<string, unknown>[]
  stats: import("@/lib/types").ColumnStat[]
  insights: DataInsight[]
}) {
  const columns = preview.length > 0 ? Object.keys(preview[0]) : []

  const severityClass = (s: DataInsight["severity"]) =>
    s === "critical"
      ? "bg-red-50 border-red-200 text-red-800 dark:bg-red-950 dark:border-red-900 dark:text-red-200"
      : s === "warning"
      ? "bg-amber-50 border-amber-200 text-amber-800 dark:bg-amber-950 dark:border-amber-900 dark:text-amber-200"
      : "bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-950 dark:border-blue-900 dark:text-blue-200"

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 border-b px-4 py-3">
        <h2 className="text-sm font-semibold">{filename}</h2>
        <Badge variant="outline">{rowCount.toLocaleString()} rows</Badge>
        <Badge variant="outline">{columnCount} columns</Badge>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4">
          {/* Insights panel */}
          {insights.length > 0 && (
            <div className="mb-5">
              <h3 className="mb-2 text-sm font-semibold">Insights</h3>
              <div className="flex flex-col gap-2">
                {insights.map((insight, i) => (
                  <div
                    key={i}
                    className={`rounded-lg border px-3 py-2 text-xs ${severityClass(insight.severity)}`}
                  >
                    <p className="font-semibold">{insight.title}</p>
                    <p className="mt-0.5 opacity-80">{insight.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Column Stats */}
          {stats.length > 0 && (
            <div className="mb-6">
              <h3 className="mb-3 text-sm font-semibold">Column Statistics</h3>
              <div className="grid grid-cols-2 gap-2 lg:grid-cols-3">
                {stats.map((col) => (
                  <Card key={col.name} size="sm">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <span className="truncate">{col.name}</span>
                        <Badge variant="secondary">{col.dtype}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-1 text-xs text-muted-foreground">
                        <p>
                          Nulls: {col.null_count} ({col.null_pct.toFixed(1)}%)
                        </p>
                        <p>Unique: {col.unique_count}</p>
                        {col.mean != null && (
                          <p>
                            Mean: {Number(col.mean).toFixed(2)} | Std:{" "}
                            {col.std != null ? Number(col.std).toFixed(2) : "N/A"}
                          </p>
                        )}
                        {col.min != null && col.max != null && (
                          <p>
                            Range: {col.min} – {col.max}
                          </p>
                        )}
                        {col.outliers && col.outliers.count > 0 && (
                          <p className="text-amber-600 dark:text-amber-400">
                            {col.outliers.count} outlier
                            {col.outliers.count !== 1 ? "s" : ""} (
                            {col.outliers.pct.toFixed(1)}%)
                          </p>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          <Separator className="my-4" />

          {/* Data Table */}
          <h3 className="mb-3 text-sm font-semibold">
            Data Preview (first {preview.length} rows)
          </h3>
          {preview.length > 0 && (
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b bg-muted/50">
                    {columns.map((col) => (
                      <th
                        key={col}
                        className="whitespace-nowrap px-3 py-2 font-medium"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, i) => (
                    <tr key={i} className="border-b last:border-b-0">
                      {columns.map((col) => (
                        <td
                          key={col}
                          className="max-w-[200px] truncate whitespace-nowrap px-3 py-1.5"
                        >
                          {row[col] == null ? (
                            <span className="text-muted-foreground/50">null</span>
                          ) : (
                            String(row[col])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
