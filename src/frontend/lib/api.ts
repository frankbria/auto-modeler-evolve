import type {
  UploadResponse,
  Project,
  ChatMessage,
  QueryResponse,
  FeatureSuggestion,
  FeatureSetResult,
  TargetResult,
  FeatureImportanceResult,
  ModelRecommendation,
  ModelRun,
  TrainingStatus,
  ModelComparison,
  ValidationMetricsResponse,
  GlobalExplanationResponse,
  RowExplanationResponse,
  Deployment,
  PredictionResult,
} from "./types"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export const api = {
  projects: {
    list: (): Promise<Project[]> =>
      fetch(`${API_URL}/api/projects`).then((r) => r.json()),

    create: (name: string, description?: string): Promise<Project> =>
      fetch(`${API_URL}/api/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      }).then((r) => r.json()),

    get: (id: string): Promise<Project> =>
      fetch(`${API_URL}/api/projects/${id}`).then((r) => r.json()),

    update: (
      id: string,
      body: { name?: string; description?: string }
    ): Promise<Project> =>
      fetch(`${API_URL}/api/projects/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json()),

    duplicate: (id: string): Promise<Project> =>
      fetch(`${API_URL}/api/projects/${id}/duplicate`, {
        method: "POST",
      }).then((r) => r.json()),

    delete: (id: string): Promise<Response> =>
      fetch(`${API_URL}/api/projects/${id}`, { method: "DELETE" }),
  },

  data: {
    upload: (projectId: string, file: File): Promise<UploadResponse> => {
      const form = new FormData()
      form.append("project_id", projectId)
      form.append("file", file)
      return fetch(`${API_URL}/api/data/upload`, {
        method: "POST",
        body: form,
      }).then((r) => r.json())
    },

    loadSample: (projectId: string): Promise<UploadResponse> =>
      fetch(`${API_URL}/api/data/sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId }),
      }).then((r) => r.json()),

    sampleInfo: (): Promise<{ filename: string; row_count: number; column_count: number; columns: string[]; description: string }> =>
      fetch(`${API_URL}/api/data/sample/info`).then((r) => r.json()),

    preview: (
      datasetId: string
    ): Promise<UploadResponse> =>
      fetch(`${API_URL}/api/data/${datasetId}/preview`).then((r) => r.json()),

    profile: (datasetId: string) =>
      fetch(`${API_URL}/api/data/${datasetId}/profile`).then((r) => r.json()),

    query: (datasetId: string, question: string): Promise<QueryResponse> =>
      fetch(`${API_URL}/api/data/${datasetId}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }).then((r) => r.json()),

    timeseries: (
      datasetId: string,
      valueColumn?: string,
      window?: number
    ): Promise<{
      dataset_id: string
      date_columns: string[]
      value_columns: string[]
      date_column?: string
      value_column?: string
      chart_spec: import("./types").ChartSpec | null
      message?: string
    }> => {
      const params = new URLSearchParams()
      if (valueColumn) params.set("value_column", valueColumn)
      if (window) params.set("window", window.toString())
      const qs = params.toString() ? `?${params}` : ""
      return fetch(`${API_URL}/api/data/${datasetId}/timeseries${qs}`).then((r) => r.json())
    },

    correlations: (datasetId: string): Promise<{
      dataset_id: string
      chart_spec: import("./types").ChartSpec | null
      pairs?: Array<{ col_a: string; col_b: string; correlation: number }>
      message?: string
    }> =>
      fetch(`${API_URL}/api/data/${datasetId}/correlations`).then((r) => r.json()),
  },

  chat: {
    send: (projectId: string, message: string): Promise<Response> =>
      fetch(`${API_URL}/api/chat/${projectId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      }),

    history: (projectId: string): Promise<{ messages: ChatMessage[] }> =>
      fetch(`${API_URL}/api/chat/${projectId}/history`).then((r) => r.json()),
  },

  features: {
    suggestions: (
      datasetId: string
    ): Promise<{ dataset_id: string; suggestions: FeatureSuggestion[] }> =>
      fetch(`${API_URL}/api/features/${datasetId}/suggestions`).then((r) =>
        r.json()
      ),

    apply: (
      datasetId: string,
      transformations: { column: string; transform_type: string; params?: Record<string, unknown> }[]
    ): Promise<FeatureSetResult> =>
      fetch(`${API_URL}/api/features/${datasetId}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transformations }),
      }).then((r) => r.json()),

    setTarget: (
      datasetId: string,
      targetColumn: string,
      featureSetId?: string
    ): Promise<TargetResult> =>
      fetch(`${API_URL}/api/features/${datasetId}/target`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: targetColumn,
          feature_set_id: featureSetId,
        }),
      }).then((r) => r.json()),

    importance: (
      datasetId: string,
      targetColumn: string
    ): Promise<FeatureImportanceResult> =>
      fetch(
        `${API_URL}/api/features/${datasetId}/importance?target_column=${encodeURIComponent(targetColumn)}`
      ).then((r) => r.json()),
  },

  models: {
    recommendations: (projectId: string): Promise<{
      project_id: string
      problem_type: string
      target_column: string
      n_rows: number
      n_features: number
      recommendations: ModelRecommendation[]
    }> =>
      fetch(`${API_URL}/api/models/${projectId}/recommendations`).then((r) =>
        r.json()
      ),

    train: (
      projectId: string,
      algorithms: string[]
    ): Promise<TrainingStatus> =>
      fetch(`${API_URL}/api/models/${projectId}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ algorithms }),
      }).then((r) => r.json()),

    runs: (projectId: string): Promise<{ project_id: string; runs: ModelRun[] }> =>
      fetch(`${API_URL}/api/models/${projectId}/runs`).then((r) => r.json()),

    compare: (projectId: string): Promise<ModelComparison> =>
      fetch(`${API_URL}/api/models/${projectId}/compare`).then((r) => r.json()),

    comparisonRadar: (projectId: string): Promise<{ chart: import("./types").ChartSpec } | null> =>
      fetch(`${API_URL}/api/models/${projectId}/comparison-radar`).then((r) =>
        r.status === 204 ? null : r.json()
      ),

    select: (modelRunId: string): Promise<ModelRun> =>
      fetch(`${API_URL}/api/models/${modelRunId}/select`, {
        method: "POST",
      }).then((r) => r.json()),

    downloadUrl: (modelRunId: string): string =>
      `${API_URL}/api/models/${modelRunId}/download`,

    reportUrl: (modelRunId: string): string =>
      `${API_URL}/api/models/${modelRunId}/report`,

    trainingStreamUrl: (projectId: string): string =>
      `${API_URL}/api/models/${projectId}/training-stream`,
  },

  validation: {
    metrics: (modelRunId: string): Promise<ValidationMetricsResponse> =>
      fetch(`${API_URL}/api/validate/${modelRunId}/metrics`).then((r) => r.json()),

    explain: (modelRunId: string): Promise<GlobalExplanationResponse> =>
      fetch(`${API_URL}/api/validate/${modelRunId}/explain`).then((r) => r.json()),

    explainRow: (modelRunId: string, rowIndex: number): Promise<RowExplanationResponse> =>
      fetch(`${API_URL}/api/validate/${modelRunId}/explain/${rowIndex}`).then((r) => r.json()),
  },

  deploy: {
    deploy: (modelRunId: string): Promise<Deployment> =>
      fetch(`${API_URL}/api/deploy/${modelRunId}`, { method: "POST" }).then((r) =>
        r.json()
      ),

    list: (): Promise<Deployment[]> =>
      fetch(`${API_URL}/api/deployments`).then((r) => r.json()),

    get: (deploymentId: string): Promise<Deployment> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}`).then((r) => r.json()),

    undeploy: (deploymentId: string): Promise<Response> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}`, { method: "DELETE" }),

    predict: (
      deploymentId: string,
      inputData: Record<string, unknown>
    ): Promise<PredictionResult> =>
      fetch(`${API_URL}/api/predict/${deploymentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      }).then((r) => r.json()),
  },
}
