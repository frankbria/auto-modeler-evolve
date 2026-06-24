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
  DatasetListItem,
  JoinKeySuggestion,
  MergeResponse,
  TuningResult,
  ProjectNarrative,
  ModelVersionHistory,
  ProjectAlerts,
  AnomalyResult,
  DatasetRefreshResult,
  DataDictionary,
  CrosstabResult,
  ComputeResult,
  SegmentComparisonResult,
  ForecastResult,
  DataReadinessResult,
  TargetCorrelationResult,
  ProjectHealthSummary,
  AuthResponse,
  User,
} from "./types"
import { getToken, clearToken } from "./auth-token"

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

/**
 * Public prediction-serving endpoints (`/api/predict/<id>…`) authenticate with a
 * per-deployment API key that the caller passes explicitly — NOT the user's JWT.
 * The backend feeds their `Authorization` header into `_verify_api_key`, so
 * injecting the user token would be rejected as an invalid API key, and a 401
 * there means "bad key", not "expired session". `/api/predict/compare` is the
 * one exception: it is an owner-only endpoint that genuinely uses the user JWT.
 */
function isPublicPredictionUrl(url: string): boolean {
  return url.includes("/api/predict/") && !url.includes("/api/predict/compare")
}

/**
 * Auth-aware fetch — the single chokepoint for every API call.
 *
 * Injects `Authorization: Bearer <token>` when the user is logged in, and on a
 * 401 clears the (now invalid) token and bounces to /login so the user can
 * re-authenticate. When no token is stored the call is passed through verbatim
 * so unauthenticated/public surfaces (and existing call signatures) are
 * unchanged. Public prediction-serving endpoints are exempt from both behaviours
 * (see `isPublicPredictionUrl`).
 */
export async function apiFetch(
  input: string,
  init?: RequestInit,
  opts: { suppressLoginRedirect?: boolean } = {}
): Promise<Response> {
  const token = getToken()
  const publicPrediction = isPublicPredictionUrl(input)

  let res: Response
  if (token && !publicPrediction) {
    const headers = new Headers(init?.headers)
    if (!headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${token}`)
    }
    res = await fetch(input, { ...init, headers })
  } else {
    res = init === undefined ? await fetch(input) : await fetch(input, init)
  }

  if (res.status === 401 && !publicPrediction) {
    clearToken()
    // `suppressLoginRedirect` is for passive session probes (e.g. the nav's
    // /api/auth/me hydration), which run on every page — including the public
    // /predict/[id] — and must NOT bounce an anonymous visitor to /login just
    // because a stale token happens to sit in localStorage.
    if (!opts.suppressLoginRedirect) {
      redirectToLogin()
    }
  }
  return res
}

function redirectToLogin(): void {
  if (typeof window === "undefined") return
  if (window.location.pathname === "/login") return
  try {
    window.location.assign("/login")
  } catch {
    // jsdom / non-browser environments don't implement navigation — ignore.
  }
}

/**
 * Typed HTTP error (#17). Carries the status, statusText, and parsed response
 * body so the UI can show the backend's `{detail: ...}` message and tests can
 * assert on specifics. Use `instanceof ApiError` to distinguish HTTP failures
 * from network/other errors.
 */
export class ApiError extends Error {
  readonly status: number
  readonly statusText: string
  readonly body: unknown

  constructor(res: Response, body: unknown) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? (body as { detail?: unknown }).detail
        : undefined
    super(
      typeof detail === "string" && detail
        ? detail
        : `HTTP ${res.status} ${res.statusText}`.trim()
    )
    this.name = "ApiError"
    this.status = res.status
    this.statusText = res.statusText
    this.body = body
  }
}

/**
 * Unwrap a JSON response, throwing `ApiError` on non-2xx (#17).
 *
 * Replaces the bare `r => r.json()` that ~114 methods used, which let
 * HTTP 4xx/5xx error bodies resolve as success objects and render as garbage.
 * On error it best-effort parses the body (json, else null) for the message;
 * a 204 No Content resolves to `null`. Returns `any` (not a generic) so each
 * `.then(unwrapJson)` keeps the call site's declared return type: `.then`'s
 * callback type param defaults to `unknown`, not the contextual Promise type,
 * so a generic would erase every method's return type to `unknown`.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- intentional glue: see doc above
export async function unwrapJson(r: Response): Promise<any> {
  if (!r.ok) {
    const body = await r.json().catch(() => null)
    throw new ApiError(r, body)
  }
  if (r.status === 204) return null
  return r.json()
}

/**
 * Consume a Server-Sent Events stream over an authenticated fetch.
 *
 * EventSource can't send an Authorization header, so owner-scoped streams use
 * this instead: `apiFetch` attaches the bearer token in the header, and each
 * `data:` frame is parsed and handed to `onEvent`. Resolves when the stream
 * ends or `signal` aborts. Keeping auth in the header avoids ever putting the
 * session token in a URL (where it would leak into logs / history).
 */
export async function streamSSE(
  url: string,
  onEvent: (data: unknown) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await apiFetch(url, { signal })
  if (!res.ok || !res.body) return
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split("\n\n")
    buffer = frames.pop() ?? ""
    for (const frame of frames) {
      const dataLine = frame
        .split("\n")
        .find((line) => line.startsWith("data:"))
      if (!dataLine) continue
      const payload = dataLine.slice(5).trim()
      if (!payload) continue
      try {
        onEvent(JSON.parse(payload))
      } catch {
        // malformed frame — ignore
      }
    }
  }
}

/**
 * Download an owner-scoped file over an authenticated fetch.
 *
 * `<a href>` and `window.open` can't carry the `Authorization` header, so every
 * owner-scoped download routes through here instead: `apiFetch` attaches the
 * bearer token in the header, the response body is read as a Blob, and a
 * client-side object-URL download is triggered. The session token therefore
 * never appears in the URL (where it would leak into proxy/server logs and
 * browser history) — the reason this was deferred in #25 and finished in #28.
 *
 * `url` may be absolute (e.g. an `api.*Url(...)` builder result) or a relative
 * backend path from an API response (`result.download_url`), which is resolved
 * against `API_URL`. The filename is taken from the response's
 * `Content-Disposition` header when present, otherwise `fallbackFilename`.
 * Throws if the request fails so callers can surface an error state.
 */
export async function downloadFile(
  url: string,
  fallbackFilename = "download"
): Promise<void> {
  const target = /^https?:\/\//i.test(url) ? url : `${API_URL}${url}`
  // Defense in depth: `apiFetch` attaches the bearer token, so only ever send
  // it to our own API origin. A tainted/cross-origin absolute URL must never
  // receive the Authorization header (token-leak guard).
  if (new URL(target).origin !== new URL(API_URL).origin) {
    throw new Error("Refusing to download from a non-API origin")
  }
  const res = await apiFetch(target)
  if (!res.ok) {
    throw new Error(`Download failed (${res.status})`)
  }
  const blob = await res.blob()

  const disposition = res.headers.get("content-disposition") ?? ""
  // RFC 5987 `filename*=UTF-8''…` is percent-encoded and must be decoded; the
  // plain `filename="…"` form is literal and may itself contain a `%` (e.g. a
  // target-column name), so decoding it would throw URIError and abort a valid
  // download. Prefer the extended form, decode it defensively, else use plain.
  const extended = disposition.match(/filename\*=(?:UTF-8'')?"?([^";]+)"?/i)
  const plain = disposition.match(/filename=\s*"?([^";]+?)"?\s*(?:;|$)/i)
  let filename = fallbackFilename
  if (extended) {
    const raw = extended[1].trim()
    try {
      filename = decodeURIComponent(raw)
    } catch {
      filename = raw
    }
  } else if (plain) {
    filename = plain[1].trim()
  }

  const objectUrl = URL.createObjectURL(blob)
  try {
    const a = document.createElement("a")
    a.href = objectUrl
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

export const api = {
  auth: {
    register: (email: string, password: string, name?: string): Promise<AuthResponse> =>
      fetch(`${API_URL}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, ...(name ? { name } : {}) }),
      }).then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? "Registration failed")
        return r.json()
      }),

    login: (email: string, password: string): Promise<AuthResponse> =>
      fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      }).then(async (r) => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? "Invalid email or password")
        return r.json()
      }),

    me: (): Promise<User> =>
      // Passive session probe — suppress the global 401→/login redirect so a
      // stale token on a public page doesn't bounce an anonymous visitor.
      apiFetch(`${API_URL}/api/auth/me`, undefined, {
        suppressLoginRedirect: true,
      }).then((r) => {
        if (!r.ok) throw new Error("Not authenticated")
        return r.json()
      }),
  },

  projects: {
    list: (): Promise<Project[]> =>
      apiFetch(`${API_URL}/api/projects`).then(unwrapJson),

    create: (name: string, description?: string): Promise<Project> =>
      apiFetch(`${API_URL}/api/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      }).then(unwrapJson),

    get: (id: string): Promise<Project> =>
      apiFetch(`${API_URL}/api/projects/${id}`).then(unwrapJson),

    update: (
      id: string,
      body: { name?: string; description?: string }
    ): Promise<Project> =>
      apiFetch(`${API_URL}/api/projects/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(unwrapJson),

    duplicate: (id: string): Promise<Project> =>
      apiFetch(`${API_URL}/api/projects/${id}/duplicate`, {
        method: "POST",
      }).then(unwrapJson),

    delete: (id: string): Promise<Response> =>
      apiFetch(`${API_URL}/api/projects/${id}`, { method: "DELETE" }),

    narrative: (id: string): Promise<ProjectNarrative> =>
      apiFetch(`${API_URL}/api/projects/${id}/narrative`, { method: "POST" }).then(unwrapJson),

    executiveBriefing: (id: string): Promise<import("./types").ExecutiveBriefingResult> =>
      apiFetch(`${API_URL}/api/projects/${id}/executive-briefing`).then(unwrapJson),

    alerts: (id: string): Promise<ProjectAlerts> =>
      apiFetch(`${API_URL}/api/projects/${id}/alerts`).then(unwrapJson),

    healthSummary: (id: string): Promise<ProjectHealthSummary> =>
      apiFetch(`${API_URL}/api/projects/${id}/health-summary`).then(unwrapJson),

    setAutoRetrain: (id: string, enabled: boolean): Promise<Response> =>
      apiFetch(`${API_URL}/api/projects/${id}/auto-retrain`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      }),

    analysisTemplates: (id: string): Promise<import("./types").AnalysisTemplate[]> =>
      apiFetch(`${API_URL}/api/projects/${id}/analysis-templates`).then(unwrapJson),

    createAnalysisTemplate: (
      id: string,
      name: string,
      queries: string[]
    ): Promise<import("./types").AnalysisTemplate> =>
      apiFetch(`${API_URL}/api/projects/${id}/analysis-templates`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, queries }),
      }).then(unwrapJson),

    deleteAnalysisTemplate: (
      projectId: string,
      templateId: string
    ): Promise<void> =>
      apiFetch(`${API_URL}/api/projects/${projectId}/analysis-templates/${templateId}`, {
        method: "DELETE",
      }).then(() => undefined),

    crossComparison: (): Promise<import("./types").CrossProjectComparisonResult> =>
      apiFetch(`${API_URL}/api/projects/cross-comparison`).then(unwrapJson),
  },

  data: {
    upload: (projectId: string, file: File): Promise<UploadResponse> => {
      const form = new FormData()
      form.append("project_id", projectId)
      form.append("file", file)
      return apiFetch(`${API_URL}/api/data/upload`, {
        method: "POST",
        body: form,
      }).then(unwrapJson)
    },

    loadSample: (projectId: string): Promise<UploadResponse> =>
      apiFetch(`${API_URL}/api/data/sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId }),
      }).then(unwrapJson),

    uploadFromUrl: (
      projectId: string,
      url: string,
      filename?: string
    ): Promise<UploadResponse & { source: string }> =>
      apiFetch(`${API_URL}/api/data/upload-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, url, filename }),
      }).then(unwrapJson),

    sampleInfo: (): Promise<{ filename: string; row_count: number; column_count: number; columns: string[]; description: string }> =>
      apiFetch(`${API_URL}/api/data/sample/info`).then(unwrapJson),

    uploadDb: (
      projectId: string,
      file: File
    ): Promise<{ project_id: string; db_filename: string; db_path: string; tables: string[]; table_count: number }> => {
      const form = new FormData()
      form.append("project_id", projectId)
      form.append("file", file)
      return apiFetch(`${API_URL}/api/data/upload-db`, {
        method: "POST",
        body: form,
      }).then(unwrapJson)
    },

    extractDb: (
      projectId: string,
      dbPath: string,
      tableName: string,
      query?: string
    ): Promise<UploadResponse & { table_name: string; query: string; source: string }> =>
      apiFetch(`${API_URL}/api/data/extract-db`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, db_path: dbPath, table_name: tableName, query }),
      }).then(unwrapJson),

    preview: (
      datasetId: string
    ): Promise<UploadResponse> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/preview`).then(unwrapJson),

    profile: (datasetId: string) =>
      apiFetch(`${API_URL}/api/data/${datasetId}/profile`).then(unwrapJson),

    query: (datasetId: string, question: string): Promise<QueryResponse> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }).then(unwrapJson),

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
      return apiFetch(`${API_URL}/api/data/${datasetId}/timeseries${qs}`).then(unwrapJson)
    },

    correlations: (datasetId: string): Promise<{
      dataset_id: string
      chart_spec: import("./types").ChartSpec | null
      pairs?: Array<{ col_a: string; col_b: string; correlation: number }>
      message?: string
    }> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/correlations`).then(unwrapJson),

    boxplot: (
      datasetId: string,
      column: string,
      groupby?: string
    ): Promise<import("./types").ChartSpec> => {
      const params = new URLSearchParams({ column })
      if (groupby) params.set("groupby", groupby)
      return apiFetch(`${API_URL}/api/data/${datasetId}/boxplot?${params}`).then(unwrapJson)
    },

    clean: (
      datasetId: string,
      operation: import("./types").CleanOperation
    ): Promise<import("./types").CleanResult> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/clean`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(operation),
      }).then(unwrapJson),

    detectAnomalies: (
      datasetId: string,
      features: string[],
      contamination?: number,
      nTop?: number
    ): Promise<AnomalyResult> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/anomalies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features,
          contamination: contamination ?? 0.05,
          n_top: nTop ?? 20,
        }),
      }).then(unwrapJson),

    listByProject: (projectId: string): Promise<DatasetListItem[]> =>
      apiFetch(`${API_URL}/api/data/project/${projectId}/datasets`).then(unwrapJson),

    joinKeys: (
      datasetId1: string,
      datasetId2: string
    ): Promise<{
      dataset_id_1: string
      dataset_id_2: string
      join_key_suggestions: JoinKeySuggestion[]
      common_column_count: number
    }> =>
      apiFetch(`${API_URL}/api/data/join-keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id_1: datasetId1, dataset_id_2: datasetId2 }),
      }).then(unwrapJson),

    merge: (
      projectId: string,
      body: {
        dataset_id_1: string
        dataset_id_2: string
        join_key: string
        how: string
        suffix_left?: string
        suffix_right?: string
        save_as_filename?: string
      }
    ): Promise<MergeResponse> =>
      apiFetch(`${API_URL}/api/data/${projectId}/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(unwrapJson),

    refresh: (datasetId: string, file: File): Promise<DatasetRefreshResult> => {
      const form = new FormData()
      form.append("file", file)
      return apiFetch(`${API_URL}/api/data/${datasetId}/refresh`, {
        method: "POST",
        body: form,
      }).then(unwrapJson)
    },

    getDictionary: (datasetId: string): Promise<DataDictionary> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/dictionary`).then(unwrapJson),

    generateDictionary: (datasetId: string): Promise<DataDictionary> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/dictionary`, { method: "POST" }).then(unwrapJson),

    getCrosstab: (
      datasetId: string,
      rows: string,
      cols: string,
      values?: string,
      agg: string = "sum"
    ): Promise<CrosstabResult> => {
      const params = new URLSearchParams({ rows, cols, agg })
      if (values) params.set("values", values)
      return apiFetch(`${API_URL}/api/data/${datasetId}/crosstab?${params}`).then(unwrapJson)
    },

    computeColumn: (
      datasetId: string,
      name: string,
      expression: string
    ): Promise<ComputeResult> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/compute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, expression }),
      }).then(unwrapJson),

    compareSegments: (
      datasetId: string,
      col: string,
      val1: string,
      val2: string
    ): Promise<SegmentComparisonResult> =>
      apiFetch(
        `${API_URL}/api/data/${datasetId}/compare-segments?col=${encodeURIComponent(col)}&val1=${encodeURIComponent(val1)}&val2=${encodeURIComponent(val2)}`
      ).then(unwrapJson),

    getForecast: (
      datasetId: string,
      target?: string,
      periods?: number
    ): Promise<{ dataset_id: string; date_columns: string[]; value_columns: string[]; forecast: ForecastResult }> => {
      const params = new URLSearchParams()
      if (target) params.set("target", target)
      if (periods !== undefined) params.set("periods", String(periods))
      const qs = params.toString()
      return apiFetch(
        `${API_URL}/api/data/${datasetId}/forecast${qs ? `?${qs}` : ""}`
      ).then(unwrapJson)
    },

    getReadinessCheck: (
      datasetId: string,
      target?: string
    ): Promise<DataReadinessResult> => {
      const params = new URLSearchParams()
      if (target) params.set("target", target)
      const qs = params.toString()
      return apiFetch(
        `${API_URL}/api/data/${datasetId}/readiness-check${qs ? `?${qs}` : ""}`
      ).then(unwrapJson)
    },

    getTargetCorrelations: (
      datasetId: string,
      target: string,
      topN?: number
    ): Promise<TargetCorrelationResult> => {
      const params = new URLSearchParams({ target })
      if (topN !== undefined) params.set("top_n", String(topN))
      return apiFetch(
        `${API_URL}/api/data/${datasetId}/target-correlations?${params.toString()}`
      ).then(unwrapJson)
    },

    renameColumn: (
      datasetId: string,
      oldName: string,
      newName: string
    ): Promise<{ dataset_id: string; old_name: string; new_name: string; row_count: number; column_count: number }> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/rename-column`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      }).then(unwrapJson),

    getDataStory: (datasetId: string, target?: string): Promise<import("./types").DataStory> => {
      const params = new URLSearchParams()
      if (target) params.set("target", target)
      const qs = params.toString()
      return apiFetch(
        `${API_URL}/api/data/${datasetId}/story${qs ? `?${qs}` : ""}`
      ).then(unwrapJson)
    },

    setFilter: (
      datasetId: string,
      conditions: import("./types").FilterCondition[]
    ): Promise<import("./types").FilterSetResult> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/set-filter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conditions }),
      }).then(unwrapJson),

    clearFilter: (datasetId: string): Promise<{ dataset_id: string; cleared: boolean }> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/clear-filter`, {
        method: "DELETE",
      }).then(unwrapJson),

    getActiveFilter: (datasetId: string): Promise<import("./types").ActiveFilter> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/active-filter`).then(unwrapJson),

    getColumnProfile: (
      datasetId: string,
      col: string
    ): Promise<import("./types").ColumnProfile> =>
      apiFetch(
        `${API_URL}/api/data/${datasetId}/column-profile?col=${encodeURIComponent(col)}`
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getClusters: (
      datasetId: string,
      features?: string[],
      nClusters?: number
    ): Promise<import("./types").ClusteringResult> => {
      const params = new URLSearchParams()
      if (features && features.length > 0) params.set("features", features.join(","))
      if (nClusters !== undefined) params.set("n_clusters", String(nClusters))
      const qs = params.toString()
      return apiFetch(
        `${API_URL}/api/data/${datasetId}/clusters${qs ? `?${qs}` : ""}`
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
    },

    compareTimeWindows: (
      datasetId: string,
      dateCol: string,
      p1Name: string,
      p1Start: string,
      p1End: string,
      p2Name: string,
      p2Start: string,
      p2End: string
    ): Promise<import("./types").TimeWindowComparison> => {
      const params = new URLSearchParams({
        date_col: dateCol,
        p1_name: p1Name,
        p1_start: p1Start,
        p1_end: p1End,
        p2_name: p2Name,
        p2_start: p2Start,
        p2_end: p2End,
      })
      return apiFetch(`${API_URL}/api/data/${datasetId}/compare-time-windows?${params}`).then(
        (r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`)
          return r.json()
        }
      )
    },

    getTopN: (
      datasetId: string,
      col: string,
      n = 10,
      order: "asc" | "desc" = "desc"
    ): Promise<import("./types").TopNResult> => {
      const params = new URLSearchParams({ col, n: String(n), order })
      return apiFetch(`${API_URL}/api/data/${datasetId}/top-n?${params}`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
    },

    getRecords: (
      datasetId: string,
      n = 20,
      where = "",
      offset = 0
    ): Promise<import("./types").RecordTableResult> => {
      const params = new URLSearchParams({ n: String(n), offset: String(offset) })
      if (where) params.set("where", where)
      return apiFetch(`${API_URL}/api/data/${datasetId}/records?${params}`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
    },

    downloadDatasetUrl: (datasetId: string): string =>
      `${API_URL}/api/data/${datasetId}/download`,

    getSummaryStats: (
      datasetId: string
    ): Promise<import("./types").SummaryStatsResult> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/summary-stats`).then(unwrapJson),

    getValueCounts: (
      datasetId: string,
      col: string,
      n: number = 20
    ): Promise<import("./types").ValueCountResult> =>
      apiFetch(
        `${API_URL}/api/data/${datasetId}/value-counts?col=${encodeURIComponent(col)}&n=${n}`
      ).then(unwrapJson),

    getPairCorrelation: (
      datasetId: string,
      col1: string,
      col2: string
    ): Promise<import("./types").PairCorrelationResult> =>
      apiFetch(
        `${API_URL}/api/data/${datasetId}/pair-correlation?col1=${encodeURIComponent(col1)}&col2=${encodeURIComponent(col2)}`
      ).then(unwrapJson),

    getStatQuery: (
      datasetId: string,
      agg: string,
      col?: string
    ): Promise<import("./types").StatQueryResult> => {
      const params = new URLSearchParams({ agg })
      if (col) params.set("col", col)
      return apiFetch(
        `${API_URL}/api/data/${datasetId}/stat-query?${params.toString()}`
      ).then(unwrapJson)
    },

    getGroupTrends: (
      datasetId: string,
      dateCol: string,
      groupCol: string,
      valueCol: string
    ): Promise<import("./types").GroupTrendResult> => {
      const params = new URLSearchParams({
        date_col: dateCol,
        group_col: groupCol,
        value_col: valueCol,
      })
      return apiFetch(
        `${API_URL}/api/data/${datasetId}/group-trends?${params.toString()}`
      ).then(unwrapJson)
    },

    predictionOpportunities: (
      datasetId: string
    ): Promise<import("./types").PredictionOpportunitiesResult> =>
      apiFetch(`${API_URL}/api/data/${datasetId}/prediction-opportunities`).then(
        (r) => r.json()
      ),

    compareDatasets: (
      baselineId: string,
      newId: string
    ): Promise<import("./types").DatasetComparisonResult> =>
      apiFetch(
        `${API_URL}/api/data/compare?baseline_id=${encodeURIComponent(baselineId)}&new_id=${encodeURIComponent(newId)}`
      ).then(unwrapJson),
  },

  chat: {
    send: (projectId: string, message: string): Promise<Response> =>
      apiFetch(`${API_URL}/api/chat/${projectId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      }),

    history: (projectId: string): Promise<{ messages: ChatMessage[] }> =>
      apiFetch(`${API_URL}/api/chat/${projectId}/history`).then(unwrapJson),
  },

  features: {
    suggestions: (
      datasetId: string
    ): Promise<{ dataset_id: string; suggestions: FeatureSuggestion[] }> =>
      apiFetch(`${API_URL}/api/features/${datasetId}/suggestions`).then(unwrapJson),

    apply: (
      datasetId: string,
      transformations: { column: string; transform_type: string; params?: Record<string, unknown> }[]
    ): Promise<FeatureSetResult> =>
      apiFetch(`${API_URL}/api/features/${datasetId}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transformations }),
      }).then(unwrapJson),

    setTarget: (
      datasetId: string,
      targetColumn: string,
      featureSetId?: string
    ): Promise<TargetResult> =>
      apiFetch(`${API_URL}/api/features/${datasetId}/target`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: targetColumn,
          feature_set_id: featureSetId,
        }),
      }).then(unwrapJson),

    importance: (
      datasetId: string,
      targetColumn: string
    ): Promise<FeatureImportanceResult> =>
      apiFetch(
        `${API_URL}/api/features/${datasetId}/importance?target_column=${encodeURIComponent(targetColumn)}`
      ).then(unwrapJson),

    // Pipeline step management (incremental add/undo)
    getSteps: (featureSetId: string): Promise<{
      feature_set_id: string
      step_count: number
      steps: Array<{ index: number; column: string; transform_type: string; params?: Record<string, unknown> }>
    }> =>
      apiFetch(`${API_URL}/api/features/${featureSetId}/steps`).then(unwrapJson),

    addStep: (
      featureSetId: string,
      step: { column: string; transform_type: string; params?: Record<string, unknown> }
    ): Promise<{
      feature_set_id: string
      step_index: number
      step_count: number
      new_columns: string[]
      total_columns: number
      preview: Record<string, unknown>[]
    }> =>
      apiFetch(`${API_URL}/api/features/${featureSetId}/steps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(step),
      }).then(unwrapJson),

    removeStep: (
      featureSetId: string,
      stepIndex: number
    ): Promise<{
      feature_set_id: string
      removed_step: { column: string; transform_type: string }
      step_count: number
      steps: Array<{ index: number; column: string; transform_type: string }>
      new_columns: string[]
      total_columns: number
    }> =>
      apiFetch(`${API_URL}/api/features/${featureSetId}/steps/${stepIndex}`, {
        method: "DELETE",
      }).then(unwrapJson),
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
      apiFetch(`${API_URL}/api/models/${projectId}/recommendations`).then(unwrapJson),

    classImbalance: (
      projectId: string
    ): Promise<import("./types").ClassImbalanceResult> =>
      apiFetch(`${API_URL}/api/models/${projectId}/imbalance`).then(unwrapJson),

    splitStrategy: (
      projectId: string
    ): Promise<import("./types").SplitStrategyInfo> =>
      apiFetch(`${API_URL}/api/models/${projectId}/split-strategy`).then(unwrapJson),

    featureSelection: (
      runId: string
    ): Promise<import("./types").FeatureSelectionResult> =>
      apiFetch(`${API_URL}/api/models/${runId}/feature-selection`).then(unwrapJson),

    thresholdAnalysis: (
      runId: string
    ): Promise<import("./types").ThresholdAnalysisResult> =>
      apiFetch(`${API_URL}/api/models/${runId}/threshold-analysis`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    sampleSizeAdequacy: (
      runId: string
    ): Promise<import("./types").SampleSizeAdequacyResult> =>
      apiFetch(`${API_URL}/api/models/${runId}/sample-size-adequacy`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    classFeatureImportance: (
      runId: string
    ): Promise<import("./types").ClassFeatureImportanceResult> =>
      apiFetch(`${API_URL}/api/models/${runId}/class-feature-importance`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    calibration: (
      runId: string
    ): Promise<import("./types").CalibrationData> =>
      apiFetch(`${API_URL}/api/models/${runId}/calibration`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    calibrationCheck: (
      runId: string,
      nBins: number = 10
    ): Promise<import("./types").CalibrationCheckResult> =>
      apiFetch(`${API_URL}/api/models/${runId}/calibration-check?n_bins=${nBins}`).then(
        (r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`)
          return r.json()
        }
      ),

    improvementSuggestions: (
      projectId: string
    ): Promise<import("./types").ModelImprovementResult> =>
      apiFetch(`${API_URL}/api/models/${projectId}/improvement-suggestions`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    modelSelection: (
      projectId: string,
      criteria: import("./types").SelectionCriteria = "balanced"
    ): Promise<import("./types").ModelSelectionResult> =>
      apiFetch(
        `${API_URL}/api/models/${projectId}/model-selection?criteria=${encodeURIComponent(criteria)}`
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    qualityScore: (runId: string): Promise<import("./types").ModelQualityScoreResult> =>
      apiFetch(`${API_URL}/api/models/${runId}/quality-score`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    promotionReadiness: (runId: string): Promise<import("./types").PromotionReadinessResult> =>
      apiFetch(`${API_URL}/api/models/${runId}/promotion-readiness`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    crossModelFeatures: (projectId: string): Promise<import("./types").CrossModelFeatureResult> =>
      apiFetch(`${API_URL}/api/models/${projectId}/cross-model-features`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    train: (
      projectId: string,
      algorithms: string[],
      imbalanceStrategy?: string | null,
      splitStrategy?: string | null,
      excludedFeatures?: string[] | null
    ): Promise<TrainingStatus> =>
      apiFetch(`${API_URL}/api/models/${projectId}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithms,
          imbalance_strategy: imbalanceStrategy ?? null,
          split_strategy: splitStrategy ?? null,
          excluded_features: excludedFeatures ?? null,
        }),
      }).then(unwrapJson),

    runs: (projectId: string): Promise<{ project_id: string; runs: ModelRun[] }> =>
      apiFetch(`${API_URL}/api/models/${projectId}/runs`).then(unwrapJson),

    compare: (projectId: string): Promise<ModelComparison> =>
      apiFetch(`${API_URL}/api/models/${projectId}/compare`).then(unwrapJson),

    comparisonRadar: (projectId: string): Promise<{ chart: import("./types").ChartSpec } | null> =>
      apiFetch(`${API_URL}/api/models/${projectId}/comparison-radar`).then((r) =>
        r.status === 204 ? null : r.json()
      ),

    select: (modelRunId: string): Promise<ModelRun> =>
      apiFetch(`${API_URL}/api/models/${modelRunId}/select`, {
        method: "POST",
      }).then(unwrapJson),

    downloadUrl: (modelRunId: string): string =>
      `${API_URL}/api/models/${modelRunId}/download`,

    reportUrl: (modelRunId: string): string =>
      `${API_URL}/api/models/${modelRunId}/report`,

    exportModelCardUrl: (runId: string): string =>
      `${API_URL}/api/models/${runId}/export-model-card`,

    trainingStreamUrl: (projectId: string): string =>
      `${API_URL}/api/models/${projectId}/training-stream`,

    readiness: (modelRunId: string): Promise<import("./types").ModelReadiness> =>
      apiFetch(`${API_URL}/api/models/${modelRunId}/readiness`).then(unwrapJson),

    tune: (modelRunId: string): Promise<TuningResult> =>
      apiFetch(`${API_URL}/api/models/${modelRunId}/tune`, { method: "POST" }).then(unwrapJson),

    retrain: (projectId: string): Promise<import("./types").RetrainResponse> =>
      apiFetch(`${API_URL}/api/models/${projectId}/retrain`, { method: "POST" }).then(unwrapJson),

    history: (projectId: string): Promise<ModelVersionHistory> =>
      apiFetch(`${API_URL}/api/models/${projectId}/history`).then(unwrapJson),

    getModelCard: (projectId: string): Promise<import("./types").ModelCard> =>
      apiFetch(`${API_URL}/api/models/${projectId}/model-card`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getSegmentPerformance: (
      modelRunId: string,
      col: string,
    ): Promise<import("./types").SegmentPerformanceResult> =>
      apiFetch(
        `${API_URL}/api/models/${modelRunId}/segment-performance?col=${encodeURIComponent(col)}`,
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getPredictionErrors: (
      modelRunId: string,
      n: number = 10,
    ): Promise<import("./types").PredictionErrorResult> =>
      apiFetch(`${API_URL}/api/models/${modelRunId}/prediction-errors?n=${n}`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),
  },

  validation: {
    metrics: (modelRunId: string): Promise<ValidationMetricsResponse> =>
      apiFetch(`${API_URL}/api/validate/${modelRunId}/metrics`).then(unwrapJson),

    explain: (modelRunId: string): Promise<GlobalExplanationResponse> =>
      apiFetch(`${API_URL}/api/validate/${modelRunId}/explain`).then(unwrapJson),

    explainRow: (modelRunId: string, rowIndex: number): Promise<RowExplanationResponse> =>
      apiFetch(`${API_URL}/api/validate/${modelRunId}/explain/${rowIndex}`).then(unwrapJson),
  },

  deploy: {
    deploy: (modelRunId: string): Promise<Deployment> =>
      apiFetch(`${API_URL}/api/deploy/${modelRunId}`, { method: "POST" }).then(unwrapJson),

    list: (): Promise<Deployment[]> =>
      apiFetch(`${API_URL}/api/deployments`).then(unwrapJson),

    listByProject: (projectId: string): Promise<Deployment[]> =>
      apiFetch(`${API_URL}/api/deployments?project_id=${projectId}`).then(unwrapJson),

    get: (deploymentId: string): Promise<Deployment> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}`).then(unwrapJson),

    undeploy: (deploymentId: string): Promise<Response> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}`, { method: "DELETE" }),

    predict: (
      deploymentId: string,
      inputData: Record<string, unknown>
    ): Promise<PredictionResult> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      }).then(unwrapJson),

    analytics: (deploymentId: string, days?: number): Promise<import("./types").DeploymentAnalytics> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/analytics${days ? `?days=${days}` : ""}`).then(unwrapJson),

    logs: (deploymentId: string, limit?: number, offset?: number): Promise<import("./types").PredictionLogsResponse> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/logs?limit=${limit ?? 20}&offset=${offset ?? 0}`).then(unwrapJson),

    drift: (deploymentId: string, window?: number): Promise<import("./types").DriftReport> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/drift${window ? `?window=${window}` : ""}`).then(unwrapJson),

    sla: (deploymentId: string): Promise<import("./types").SlaData> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/sla`).then(unwrapJson),

    goalSeekHistory: (deploymentId: string): Promise<import("./types").GoalSeekHistoryResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/goal-seek/history`).then(unwrapJson),

    deploymentChangelog: (deploymentId: string): Promise<import("./types").DeploymentChangelogResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/changelog`).then(unwrapJson),

    whatif: (
      deploymentId: string,
      base: Record<string, unknown>,
      overrides: Record<string, unknown>
    ): Promise<import("./types").WhatIfResult> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/whatif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base, overrides }),
      }).then(unwrapJson),

    submitFeedback: (
      deploymentId: string,
      body: {
        prediction_log_id?: string
        actual_value?: number
        actual_label?: string
        is_correct?: boolean
        comment?: string
      }
    ): Promise<import("./types").FeedbackRecord> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then(unwrapJson),

    feedbackAccuracy: (deploymentId: string): Promise<import("./types").FeedbackAccuracy> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/feedback-accuracy`).then(unwrapJson),

    trainingVsProduction: (deploymentId: string): Promise<import("./types").ProdPerformanceResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/training-vs-production`).then(unwrapJson),

    health: (deploymentId: string): Promise<import("./types").ModelHealth> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/health`).then(unwrapJson),

    explain: (deploymentId: string, inputs: Record<string, unknown>): Promise<import("./types").PredictionExplanation> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    scenarios: (
      deploymentId: string,
      base: Record<string, unknown>,
      scenarios: Array<{ label: string; overrides: Record<string, unknown> }>
    ): Promise<import("./types").ScenarioComparison> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/scenarios`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base, scenarios }),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    compareModels: (
      deploymentIds: string[],
      features: Record<string, unknown>
    ): Promise<import("./types").ComparisonResponse> =>
      apiFetch(`${API_URL}/api/predict/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ deployment_ids: deploymentIds, features }),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getIntegration: (
      deploymentId: string,
      baseUrl?: string
    ): Promise<import("./types").IntegrationSnippets> =>
      apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/integration${baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : ""}`
      ).then(unwrapJson),

    generateApiKey: (deploymentId: string): Promise<import("./types").ApiKeyResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/api-key`, { method: "POST" }).then(
        (r) => r.json()
      ),

    disableApiKey: (deploymentId: string): Promise<Response> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/api-key`, { method: "DELETE" }),

    getSchedules: (deploymentId: string): Promise<import("./types").BatchSchedule[]> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/schedules`).then(unwrapJson),

    createSchedule: (
      deploymentId: string,
      body: {
        frequency: string
        run_hour: number
        run_minute: number
        day_of_week?: number | null
        day_of_month?: number | null
      }
    ): Promise<import("./types").BatchSchedule> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/schedules`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    deleteSchedule: (deploymentId: string, scheduleId: string): Promise<void> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/schedules/${scheduleId}`, {
        method: "DELETE",
      }).then(() => undefined),

    triggerSchedule: (
      deploymentId: string,
      scheduleId: string
    ): Promise<{ status: string; schedule_id: string }> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/schedules/${scheduleId}/run`, {
        method: "POST",
      }).then(unwrapJson),

    getScheduleRuns: (
      deploymentId: string,
      scheduleId: string
    ): Promise<import("./types").BatchJobRun[]> =>
      apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/schedules/${scheduleId}/runs`
      ).then(unwrapJson),

    getVersions: (
      deploymentId: string
    ): Promise<import("./types").DeploymentVersionHistory> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/versions`).then(unwrapJson),

    rollback: (
      deploymentId: string,
      versionNumber: number
    ): Promise<import("./types").RollbackResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/rollback/${versionNumber}`, {
        method: "POST",
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    exportServiceUrl: (deploymentId: string): string =>
      `${API_URL}/api/deploy/${deploymentId}/export`,

    getWebhooks: (
      deploymentId: string
    ): Promise<import("./types").WebhookConfig[]> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/webhooks`).then(unwrapJson),

    createWebhook: (
      deploymentId: string,
      url: string,
      eventTypes: string[]
    ): Promise<import("./types").WebhookConfig> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/webhooks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, event_types: eventTypes }),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    deleteWebhook: (
      deploymentId: string,
      webhookId: string
    ): Promise<void> =>
      apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/webhooks/${webhookId}`,
        { method: "DELETE" }
      ).then(() => undefined),

    testWebhook: (
      deploymentId: string,
      webhookId: string
    ): Promise<import("./types").WebhookTestResult> =>
      apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/webhooks/${webhookId}/test`,
        { method: "POST" }
      ).then(unwrapJson),

    getAbTest: (
      deploymentId: string
    ): Promise<import("./types").ABTest> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/ab-test`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    createAbTest: (
      deploymentId: string,
      challengerId: string,
      championSplitPct: number
    ): Promise<import("./types").ABTest> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/ab-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          challenger_id: challengerId,
          champion_split_pct: championSplitPct,
        }),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    endAbTest: (deploymentId: string): Promise<void> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/ab-test`, {
        method: "DELETE",
      }).then(() => undefined),

    promoteChallenger: (
      deploymentId: string
    ): Promise<{ message: string; deployment: import("./types").Deployment }> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/ab-test/promote`, {
        method: "POST",
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    promoteToProduction: (
      deploymentId: string
    ): Promise<import("./types").EnvironmentPromotionResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/promote-to-production`, {
        method: "POST",
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    demoteToStaging: (
      deploymentId: string
    ): Promise<import("./types").EnvironmentPromotionResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/demote-to-staging`, {
        method: "POST",
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getPresets: (
      deploymentId: string
    ): Promise<import("./types").DeploymentPreset[]> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/presets`).then(unwrapJson),

    createPreset: (
      deploymentId: string,
      name: string,
      featureValues: Record<string, string | number>
    ): Promise<import("./types").DeploymentPreset> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/presets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, feature_values: featureValues }),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    deletePreset: (
      deploymentId: string,
      presetId: string
    ): Promise<void> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/presets/${presetId}`, {
        method: "DELETE",
      }).then(() => undefined),

    getSdkUrl: (
      deploymentId: string,
      language: "python" | "javascript",
      baseUrl?: string
    ): string => {
      const params = new URLSearchParams({ language })
      if (baseUrl) params.set("base_url", baseUrl)
      return `${API_URL}/api/deploy/${deploymentId}/sdk?${params.toString()}`
    },

    setRateLimit: async (
      deploymentId: string,
      rateLimitRpm: number | null,
      monthlyQuota: number | null
    ) => {
      const res = await apiFetch(`${API_URL}/api/deploy/${deploymentId}/rate-limit`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rate_limit_rpm: rateLimitRpm, monthly_quota: monthlyQuota }),
      })
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    quotaStatus: async (deploymentId: string) => {
      const res = await apiFetch(`${API_URL}/api/deploy/${deploymentId}/quota-status`)
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    covariateDrift: async (
      deploymentId: string
    ): Promise<import("./types").CovariateDriftAlertResult> => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/covariate-drift`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    predictionAudit: async (
      deploymentId: string
    ): Promise<import("./types").PredictionAuditResult> => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/prediction-audit`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    getAlertRules: async (
      deploymentId: string
    ): Promise<{ count: number; rules: import("./types").AlertRuleEntry[] }> => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/alert-rules`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    createAlertRule: async (
      deploymentId: string,
      name: string,
      conditionType: string,
      conditionOp: string,
      conditionValue: number | null,
      conditionClass: string | null
    ): Promise<import("./types").AlertRuleEntry> => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/alert-rules`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            condition_type: conditionType,
            condition_op: conditionOp,
            condition_value: conditionValue,
            condition_class: conditionClass,
          }),
        }
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    deleteAlertRule: async (
      deploymentId: string,
      ruleId: string
    ): Promise<void> => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/alert-rules/${ruleId}`,
        { method: "DELETE" }
      )
      if (!res.ok) throw new Error(await res.text())
    },

    accuracyAlertStatus: async (deploymentId: string) => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/accuracy-alert-status`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    setAccuracyAlert: async (deploymentId: string, threshold: number | null) => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/accuracy-alert`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threshold }),
        }
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    setConfidenceThreshold: async (deploymentId: string, threshold: number | null) => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/confidence-threshold`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threshold }),
        }
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    getConfidenceThresholdStatus: async (deploymentId: string): Promise<import("./types").ConfidenceThresholdConfig> => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/confidence-threshold-status`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    listInputValidationRules: async (deploymentId: string) => {
      const res = await apiFetch(`${API_URL}/api/deploy/${deploymentId}/input-validation-rules`)
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    createInputValidationRule: async (
      deploymentId: string,
      rule: {
        feature_name: string
        rule_type: "range" | "one_of" | "not_null"
        min_val?: number | null
        max_val?: number | null
        allowed_values?: string[] | null
      }
    ) => {
      const res = await apiFetch(`${API_URL}/api/deploy/${deploymentId}/input-validation-rules`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rule),
      })
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    deleteInputValidationRule: async (deploymentId: string, ruleId: string) => {
      const res = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/input-validation-rules/${ruleId}`,
        { method: "DELETE" }
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    getDashboardConfig: (deploymentId: string): Promise<import("@/lib/types").DashboardConfigResponse> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/dashboard-config`).then(unwrapJson),

    getDashboardMetadata: (deploymentId: string): Promise<import("@/lib/types").DashboardMetadata> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/dashboard-metadata`).then(unwrapJson),

    // --- PUBLIC prediction-surface reads (used by the anonymous /predict/[id]
    // page). These hit /api/predict/{id}/... which is gated by per-deployment
    // API keys, not user auth — so no Bearer token is attached and a 401 won't
    // bounce a visitor to /login. The /api/deploy/* siblings above stay
    // owner-scoped for the authenticated workspace.
    // These reject on a non-OK status (e.g. 404 for a missing/inactive
    // deployment) so the predict page's `.catch` shows the not-found/inactive
    // state instead of rendering an error body as if it were real data.
    getPublicInfo: (deploymentId: string): Promise<Deployment> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/info`).then((r) => {
        if (!r.ok) throw new Error("Deployment not found or inactive")
        return r.json()
      }),

    getPublicPresets: (
      deploymentId: string
    ): Promise<import("./types").DeploymentPreset[]> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/presets`).then((r) => {
        if (!r.ok) throw new Error("Deployment not found or inactive")
        return r.json()
      }),

    getPublicDashboardConfig: (deploymentId: string): Promise<import("@/lib/types").DashboardConfigResponse> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/dashboard-config`).then((r) => {
        if (!r.ok) throw new Error("Deployment not found or inactive")
        return r.json()
      }),

    getPublicDashboardMetadata: (deploymentId: string): Promise<import("@/lib/types").DashboardMetadata> =>
      apiFetch(`${API_URL}/api/predict/${deploymentId}/dashboard-metadata`).then((r) => {
        if (!r.ok) throw new Error("Deployment not found or inactive")
        return r.json()
      }),

    updateDashboardMetadata: async (
      deploymentId: string,
      opts: { title?: string; description?: string; clear?: boolean }
    ): Promise<import("@/lib/types").DashboardMetadata> => {
      const params = new URLSearchParams()
      if (opts.title !== undefined) params.set("title", opts.title)
      if (opts.description !== undefined) params.set("description", opts.description)
      if (opts.clear) params.set("clear", "true")
      const r = await apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/dashboard-metadata?${params}`,
        { method: "PUT" }
      )
      return r.json()
    },

    getEmbedCode: (deploymentId: string): Promise<import("@/lib/types").EmbedCodeResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/embed-code`).then(unwrapJson),

    getOutputAnomalies: (deploymentId: string, n = 50): Promise<import("@/lib/types").PredictionOutputAnomalyResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/output-anomalies?n=${n}`).then(unwrapJson),

    featureSweep: (
      deploymentId: string,
      direction: 'maximize' | 'minimize' = 'maximize',
      nSteps = 10
    ): Promise<import("@/lib/types").FeatureSweepResult> =>
      apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/feature-sweep?direction=${direction}&n_steps=${nSteps}`
      ).then(unwrapJson),

    getShareLink: (
      deploymentId: string,
      featureValues?: Record<string, string>
    ): Promise<import("@/lib/types").ShareLinkResult> => {
      const params = new URLSearchParams()
      if (featureValues && Object.keys(featureValues).length > 0) {
        params.set("features", JSON.stringify(featureValues))
      }
      const qs = params.toString()
      return apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/share-link${qs ? `?${qs}` : ""}`
      ).then(unwrapJson)
    },

    canaryStatus: (
      deploymentId: string,
      n = 200
    ): Promise<import("@/lib/types").CanaryStatusResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/canary/status?n=${n}`).then(unwrapJson),

    canaryStart: (
      deploymentId: string,
      versionNumber: number,
      trafficPct: number
    ): Promise<import("@/lib/types").CanaryStatusResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/canary/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version_number: versionNumber, traffic_pct: trafficPct }),
      }).then(unwrapJson),

    canaryCancel: (deploymentId: string): Promise<{ canary_is_active: boolean; message: string }> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/canary/cancel`, {
        method: "POST",
      }).then(unwrapJson),

    canaryPromote: (deploymentId: string): Promise<{ promoted: boolean; new_version_number: number; message: string }> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/canary/promote`, {
        method: "POST",
      }).then(unwrapJson),

    healthScorecard: (deploymentId: string, n?: number): Promise<import("./types").DeploymentHealthScorecardResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/health-scorecard${n ? `?n=${n}` : ""}`).then(unwrapJson),

    confidenceBand: (deploymentId: string, nDays?: number): Promise<import("./types").ConfidenceBandResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/confidence-band${nDays ? `?n_days=${nDays}` : ""}`).then(unwrapJson),

    outcomeCalibration: (deploymentId: string): Promise<import("./types").OutcomeCalibrationResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/outcome-calibration`).then(unwrapJson),

    getDegradationRetrainStatus: (deploymentId: string): Promise<import("./types").DegradationRetrainConfig> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/degradation-retrain-status`).then(unwrapJson),

    setDegradationRetrain: (deploymentId: string, enabled: boolean, accuracyThresholdPct?: number): Promise<import("./types").DegradationRetrainConfig> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/degradation-retrain`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled, accuracy_threshold_pct: accuracyThresholdPct ?? null }),
      }).then(unwrapJson),

    batchJobHistory: (deploymentId: string, n?: number): Promise<import("./types").BatchJobHistoryResult> =>
      apiFetch(`${API_URL}/api/deploy/${deploymentId}/batch-job-history${n ? `?n=${n}` : ""}`).then(unwrapJson),

    performanceDecayRate: (
      deploymentId: string,
      thresholdPct?: number,
    ): Promise<import("./types").PerformanceDecayResult> =>
      apiFetch(
        `${API_URL}/api/deploy/${deploymentId}/performance-decay-rate${thresholdPct !== undefined ? `?threshold_pct=${thresholdPct}` : ""}`,
      ).then(unwrapJson),
  },
}
