"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useParams, useRouter } from "next/navigation"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { ChartMessage } from "@/components/chat/chart-message"
import { ModelTrainingPanel } from "@/components/models/model-training-panel"
import {
  FeatureSuggestionsPanel,
  FeatureImportancePanel,
} from "@/components/features/feature-suggestions"
import { ValidationPanel } from "@/components/validation/validation-panel"
import { DeploymentPanel } from "@/components/deploy/deployment-panel"
import { api } from "@/lib/api"
import { useAppStore } from "@/lib/store"
import type {
  Dataset,
  DataInsight,
  FeatureSuggestion,
  FeatureImportanceEntry,
  FeatureSetResult,
  ChatMessage as ChatMsg,
} from "@/lib/types"

const WELCOME_MESSAGE =
  "Hi! I'm your data modeling assistant. Upload a CSV file to get started, or ask me anything about your data."

function buildWelcomeBackMessage(projectName: string, messages: ChatMsg[]): string {
  const msgCount = messages.length
  // Find the last assistant message to summarise what was happening
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant")
  const snippet = lastAssistant?.content?.slice(0, 120).replace(/\n/g, " ") ?? ""
  const lastActive = messages[messages.length - 1]?.timestamp
  const sinceMin = lastActive
    ? Math.round((Date.now() - new Date(lastActive).getTime()) / 60_000)
    : 0
  const sinceStr =
    sinceMin < 60
      ? `${sinceMin} minute${sinceMin !== 1 ? "s" : ""} ago`
      : `${Math.round(sinceMin / 60)} hour${Math.round(sinceMin / 60) !== 1 ? "s" : ""} ago`

  const context = snippet ? `Last we were: "${snippet}..."` : ""
  return (
    `Welcome back to **${projectName}**! You have ${msgCount} messages in this session (last active ${sinceStr}). ` +
    (context ? `${context} ` : "") +
    `What would you like to work on?`
  )
}

type RightTab = "data" | "features" | "importance" | "models" | "validate" | "deploy"

export default function ProjectWorkspace() {
  const params = useParams<{ id: string }>()
  const router = useRouter()
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
  const [activeTab, setActiveTab] = useState<RightTab>("data")
  const [rightPanelVisible, setRightPanelVisible] = useState(true)

  // Feature engineering state
  const [featureSuggestions, setFeatureSuggestions] = useState<FeatureSuggestion[]>([])
  const [loadingFeatures, setLoadingFeatures] = useState(false)
  const [targetColumn, setTargetColumn] = useState("")
  const [importanceFeatures, setImportanceFeatures] = useState<FeatureImportanceEntry[]>([])
  const [importanceProblemType, setImportanceProblemType] = useState("")
  const [loadingImportance, setLoadingImportance] = useState(false)

  // Validation state
  const [selectedModelRunId, setSelectedModelRunId] = useState<string | null>(null)
  const [selectedModelAlgorithm, setSelectedModelAlgorithm] = useState<string | null>(null)

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
          const msgs: ChatMsg[] = history.messages
          // Add a "welcome back" context message if this is a returning visit
          // (history has real conversation, not just the initial greeting)
          const hasConversation = msgs.some((m) => m.role === "user")
          if (hasConversation) {
            const welcomeBack: ChatMsg = {
              role: "assistant",
              content: buildWelcomeBackMessage(project.name, msgs),
              timestamp: new Date().toISOString(),
            }
            setMessages([...msgs, welcomeBack])
          } else {
            setMessages(msgs)
          }
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

  // Load feature suggestions when switching to features tab
  useEffect(() => {
    if (activeTab === "features" && currentDataset && featureSuggestions.length === 0) {
      setLoadingFeatures(true)
      api.features
        .suggestions(currentDataset.id)
        .then((r) => setFeatureSuggestions(r.suggestions))
        .catch(() => setFeatureSuggestions([]))
        .finally(() => setLoadingFeatures(false))
    }
  }, [activeTab, currentDataset, featureSuggestions.length])

  const handleLoadImportance = useCallback(async () => {
    if (!currentDataset || !targetColumn.trim()) return
    setLoadingImportance(true)
    try {
      const result = await api.features.importance(currentDataset.id, targetColumn.trim())
      setImportanceFeatures(result.features)
      setImportanceProblemType(result.problem_type)
    } finally {
      setLoadingImportance(false)
    }
  }, [currentDataset, targetColumn])

  const handleFeatureApplied = useCallback(
    (result: FeatureSetResult) => {
      addMessage({
        role: "assistant",
        content: `I've applied your feature transformations. ${result.new_columns.length} new column${result.new_columns.length !== 1 ? "s" : ""} were created: ${result.new_columns.slice(0, 5).join(", ")}${result.new_columns.length > 5 ? "..." : ""}. The dataset now has ${result.total_columns} columns total.`,
        timestamp: new Date().toISOString(),
      })
    },
    [addMessage]
  )

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
        setFeatureSuggestions([]) // reset on new upload

        if (result.insights && result.insights.length > 0) {
          const insightLines = result.insights
            .slice(0, 3)
            .map((i: DataInsight) => `- ${i.title}: ${i.detail}`)
            .join("\n")
          addMessage({
            role: "assistant",
            content: `I've analyzed **${result.filename}** (${result.row_count.toLocaleString()} rows, ${result.column_count} columns). Here's what I noticed:\n\n${insightLines}\n\nWhat would you like to explore? You can also check the **Features** tab to see transformation suggestions.`,
            timestamp: new Date().toISOString(),
          })
        }
        // Show right panel on upload if hidden
        setRightPanelVisible(true)
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

  const handleLoadSample = useCallback(async () => {
    setUploading(true)
    try {
      const result = await api.data.loadSample(projectId)
      const dataset: Dataset = {
        id: result.dataset_id,
        project_id: projectId,
        filename: result.filename,
        row_count: result.row_count,
        column_count: result.column_count,
        uploaded_at: new Date().toISOString(),
      }
      setDataset(dataset, result.preview, result.column_stats, result.insights ?? [])
      setFeatureSuggestions([])
      setRightPanelVisible(true)
      addMessage({
        role: "assistant",
        content: `I've loaded the sample sales dataset — **${result.row_count} rows** across 5 product lines and 4 regions. This data contains monthly sales figures with date, product, region, revenue, and units sold.\n\nYou can use this to try predicting **revenue** using the other columns. Ask me anything about the data, or jump to the **Features** tab to get started.`,
        timestamp: new Date().toISOString(),
      })
    } catch {
      addMessage({
        role: "assistant",
        content: "There was a problem loading the sample data. Please try again.",
        timestamp: new Date().toISOString(),
      })
    } finally {
      setUploading(false)
    }
  }, [projectId, setDataset, addMessage])

  if (loadingProject) {
    return (
      <div className="flex h-[calc(100vh-3rem)] items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading project...</p>
      </div>
    )
  }

  return (
    <div className="flex h-screen flex-col">
      {/* Top bar */}
      <div className="flex h-10 shrink-0 items-center gap-3 border-b px-4">
        <button
          onClick={() => router.push("/")}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          ← Projects
        </button>
        <span className="text-xs text-muted-foreground">/</span>
        <span className="text-xs font-medium truncate">
          {currentProject?.name ?? "Loading..."}
        </span>
        <div className="ml-auto">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => setRightPanelVisible((v) => !v)}
          >
            {rightPanelVisible ? "Hide panel" : "Show panel"}
          </Button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Chat Panel */}
        <div
          className={`flex flex-col border-r ${rightPanelVisible ? "w-2/5" : "flex-1"} transition-all`}
        >
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

        {/* Right Panel */}
        {rightPanelVisible && (
          <div className="flex w-3/5 flex-col overflow-hidden">
            {currentDataset ? (
              <>
                {/* Tab Bar */}
                <div className="flex border-b overflow-x-auto">
                  {(["data", "features", "importance", "models", "validate", "deploy"] as RightTab[]).map((tab) => {
                    const labels: Record<RightTab, string> = {
                      data: "Data",
                      features: "Features",
                      importance: "Importance",
                      models: "Models",
                      validate: "Validate",
                      deploy: "Deploy",
                    }
                    return (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`shrink-0 px-4 py-2.5 text-xs font-medium capitalize transition-colors ${
                          activeTab === tab
                            ? "border-b-2 border-primary text-foreground"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {labels[tab]}
                      </button>
                    )
                  })}
                </div>

                {activeTab === "data" && (
                  <DataPreviewPanel
                    filename={currentDataset.filename}
                    rowCount={currentDataset.row_count}
                    columnCount={currentDataset.column_count}
                    preview={dataPreview}
                    stats={columnStats}
                    insights={dataInsights}
                  />
                )}

                {activeTab === "features" && (
                  <ScrollArea className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Feature Suggestions</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          Select transformations to apply. Approved features will be added as new columns.
                        </p>
                      </div>
                      {loadingFeatures ? (
                        <p className="text-xs text-muted-foreground">Analyzing columns...</p>
                      ) : (
                        <FeatureSuggestionsPanel
                          datasetId={currentDataset.id}
                          suggestions={featureSuggestions}
                          onApplied={handleFeatureApplied}
                        />
                      )}
                    </div>
                  </ScrollArea>
                )}

                {activeTab === "importance" && (
                  <ScrollArea className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Feature Importance</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          Select a column to predict and see which features are most useful.
                        </p>
                      </div>
                      <div className="mb-4 flex gap-2">
                        <Input
                          placeholder="Target column (e.g. revenue)"
                          value={targetColumn}
                          onChange={(e) => setTargetColumn(e.target.value)}
                          className="text-xs"
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleLoadImportance()
                          }}
                        />
                        <Button
                          size="sm"
                          onClick={handleLoadImportance}
                          disabled={!targetColumn.trim() || loadingImportance}
                        >
                          {loadingImportance ? "..." : "Analyze"}
                        </Button>
                      </div>
                      {importanceFeatures.length > 0 && (
                        <FeatureImportancePanel
                          features={importanceFeatures}
                          targetColumn={targetColumn}
                          problemType={importanceProblemType}
                        />
                      )}
                    </div>
                  </ScrollArea>
                )}

                {activeTab === "validate" && currentDataset && (
                  <div className="flex flex-1 flex-col overflow-hidden">
                    <ValidationPanel
                      projectId={projectId}
                      selectedRunId={selectedModelRunId}
                      algorithmName={selectedModelAlgorithm}
                    />
                  </div>
                )}

                {activeTab === "deploy" && currentDataset && (
                  <ScrollArea className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Deploy Model</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          One-click deployment as a live prediction API + shareable dashboard.
                        </p>
                      </div>
                      <DeploymentPanel
                        projectId={projectId}
                        selectedRunId={selectedModelRunId}
                        algorithmName={selectedModelAlgorithm}
                        onDeployed={(dep) => {
                          addMessage({
                            role: "assistant",
                            content: `Your model is live! Share this link with anyone: ${dep.dashboard_url}\n\nThey can fill in values and get instant predictions — no code required. Developers can also use the API endpoint directly: POST ${dep.endpoint_path}`,
                            timestamp: new Date().toISOString(),
                          })
                        }}
                      />
                    </div>
                  </ScrollArea>
                )}

                {activeTab === "models" && currentDataset && (
                  <ScrollArea className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Model Training</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          Train and compare ML models on your dataset. Make sure you have set a target column in the Features tab first.
                        </p>
                      </div>
                      <ModelTrainingPanel
                        projectId={projectId}
                        onModelSelected={(runId, algorithm) => {
                          setSelectedModelRunId(runId)
                          setSelectedModelAlgorithm(algorithm)
                          addMessage({
                            role: "assistant",
                            content: `I have selected this model for your project. You can now go to the **Validate** tab to run cross-validation, see error analysis, and understand feature importance. Or we can deploy it as a live prediction API whenever you're ready.`,
                            timestamp: new Date().toISOString(),
                          })
                        }}
                        onModelDownload={(runId) => {
                          window.open(api.models.downloadUrl(runId), "_blank")
                        }}
                      />
                    </div>
                  </ScrollArea>
                )}
              </>
            ) : (
              <UploadPanel
                getRootProps={getRootProps}
                getInputProps={getInputProps}
                isDragActive={isDragActive}
                uploading={uploading}
                onLoadSample={handleLoadSample}
              />
            )}
          </div>
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
  onLoadSample,
}: {
  getRootProps: ReturnType<typeof useDropzone>["getRootProps"]
  getInputProps: ReturnType<typeof useDropzone>["getInputProps"]
  isDragActive: boolean
  uploading: boolean
  onLoadSample: () => void
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
      <div
        {...getRootProps()}
        className={`flex h-56 w-full max-w-md cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors ${
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
            <p className="mt-1 text-xs text-muted-foreground">or click to browse</p>
            <p className="mt-3 text-xs text-muted-foreground max-w-xs text-center">
              AutoModeler will profile your data, suggest features, train models,
              and help you deploy — all through conversation.
            </p>
          </>
        )}
      </div>

      {!uploading && (
        <div className="flex flex-col items-center gap-1">
          <p className="text-xs text-muted-foreground">Don&apos;t have a dataset handy?</p>
          <button
            onClick={onLoadSample}
            className="text-xs text-primary hover:underline underline-offset-2"
          >
            Load sample sales data (200 rows, 5 columns)
          </button>
        </div>
      )}
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
                            Range: {col.min} - {col.max}
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
