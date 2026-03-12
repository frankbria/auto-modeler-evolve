import type {
  UploadResponse,
  Project,
  ChatMessage,
  QueryResponse,
  FeatureSuggestion,
  FeatureSetResult,
  TargetResult,
  FeatureImportanceResult,
  RecommendResponse,
  TrainResponse,
  ModelComparison,
  ModelRun,
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

    preview: (
      datasetId: string
    ): Promise<{ rows: Record<string, unknown>[] }> =>
      fetch(`${API_URL}/api/data/${datasetId}/preview`).then((r) => r.json()),

    profile: (datasetId: string) =>
      fetch(`${API_URL}/api/data/${datasetId}/profile`).then((r) => r.json()),

    query: (datasetId: string, question: string): Promise<QueryResponse> =>
      fetch(`${API_URL}/api/data/${datasetId}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }).then((r) => r.json()),
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
    recommend: (
      projectId: string,
      datasetId: string,
      targetColumn: string
    ): Promise<RecommendResponse> =>
      fetch(
        `${API_URL}/api/models/${projectId}/recommend?dataset_id=${encodeURIComponent(datasetId)}&target_column=${encodeURIComponent(targetColumn)}`
      ).then((r) => r.json()),

    train: (
      projectId: string,
      datasetId: string,
      targetColumn: string,
      algorithms: string[],
      featureSetId?: string
    ): Promise<TrainResponse> =>
      fetch(`${API_URL}/api/models/${projectId}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          target_column: targetColumn,
          algorithms,
          feature_set_id: featureSetId ?? null,
        }),
      }).then((r) => r.json()),

    runs: (projectId: string): Promise<{ runs: ModelRun[] }> =>
      fetch(`${API_URL}/api/models/${projectId}/runs`).then((r) => r.json()),

    compare: (projectId: string): Promise<ModelComparison> =>
      fetch(`${API_URL}/api/models/${projectId}/compare`).then((r) => r.json()),

    select: (modelRunId: string): Promise<{ success: boolean }> =>
      fetch(`${API_URL}/api/models/${modelRunId}/select`, { method: "POST" }).then((r) =>
        r.json()
      ),
  },
}
