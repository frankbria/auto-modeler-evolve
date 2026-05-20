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

    narrative: (id: string): Promise<ProjectNarrative> =>
      fetch(`${API_URL}/api/projects/${id}/narrative`, { method: "POST" }).then((r) =>
        r.json()
      ),

    executiveBriefing: (id: string): Promise<import("./types").ExecutiveBriefingResult> =>
      fetch(`${API_URL}/api/projects/${id}/executive-briefing`).then((r) => r.json()),

    alerts: (id: string): Promise<ProjectAlerts> =>
      fetch(`${API_URL}/api/projects/${id}/alerts`).then((r) => r.json()),

    healthSummary: (id: string): Promise<ProjectHealthSummary> =>
      fetch(`${API_URL}/api/projects/${id}/health-summary`).then((r) => r.json()),

    analysisTemplates: (id: string): Promise<import("./types").AnalysisTemplate[]> =>
      fetch(`${API_URL}/api/projects/${id}/analysis-templates`).then((r) => r.json()),

    createAnalysisTemplate: (
      id: string,
      name: string,
      queries: string[]
    ): Promise<import("./types").AnalysisTemplate> =>
      fetch(`${API_URL}/api/projects/${id}/analysis-templates`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, queries }),
      }).then((r) => r.json()),

    deleteAnalysisTemplate: (
      projectId: string,
      templateId: string
    ): Promise<void> =>
      fetch(`${API_URL}/api/projects/${projectId}/analysis-templates/${templateId}`, {
        method: "DELETE",
      }).then(() => undefined),
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

    uploadFromUrl: (
      projectId: string,
      url: string,
      filename?: string
    ): Promise<UploadResponse & { source: string }> =>
      fetch(`${API_URL}/api/data/upload-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, url, filename }),
      }).then((r) => r.json()),

    sampleInfo: (): Promise<{ filename: string; row_count: number; column_count: number; columns: string[]; description: string }> =>
      fetch(`${API_URL}/api/data/sample/info`).then((r) => r.json()),

    uploadDb: (
      projectId: string,
      file: File
    ): Promise<{ project_id: string; db_filename: string; db_path: string; tables: string[]; table_count: number }> => {
      const form = new FormData()
      form.append("project_id", projectId)
      form.append("file", file)
      return fetch(`${API_URL}/api/data/upload-db`, {
        method: "POST",
        body: form,
      }).then((r) => r.json())
    },

    extractDb: (
      projectId: string,
      dbPath: string,
      tableName: string,
      query?: string
    ): Promise<UploadResponse & { table_name: string; query: string; source: string }> =>
      fetch(`${API_URL}/api/data/extract-db`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId, db_path: dbPath, table_name: tableName, query }),
      }).then((r) => r.json()),

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

    boxplot: (
      datasetId: string,
      column: string,
      groupby?: string
    ): Promise<import("./types").ChartSpec> => {
      const params = new URLSearchParams({ column })
      if (groupby) params.set("groupby", groupby)
      return fetch(`${API_URL}/api/data/${datasetId}/boxplot?${params}`).then((r) => r.json())
    },

    clean: (
      datasetId: string,
      operation: import("./types").CleanOperation
    ): Promise<import("./types").CleanResult> =>
      fetch(`${API_URL}/api/data/${datasetId}/clean`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(operation),
      }).then((r) => r.json()),

    detectAnomalies: (
      datasetId: string,
      features: string[],
      contamination?: number,
      nTop?: number
    ): Promise<AnomalyResult> =>
      fetch(`${API_URL}/api/data/${datasetId}/anomalies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features,
          contamination: contamination ?? 0.05,
          n_top: nTop ?? 20,
        }),
      }).then((r) => r.json()),

    listByProject: (projectId: string): Promise<DatasetListItem[]> =>
      fetch(`${API_URL}/api/data/project/${projectId}/datasets`).then((r) => r.json()),

    joinKeys: (
      datasetId1: string,
      datasetId2: string
    ): Promise<{
      dataset_id_1: string
      dataset_id_2: string
      join_key_suggestions: JoinKeySuggestion[]
      common_column_count: number
    }> =>
      fetch(`${API_URL}/api/data/join-keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id_1: datasetId1, dataset_id_2: datasetId2 }),
      }).then((r) => r.json()),

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
      fetch(`${API_URL}/api/data/${projectId}/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json()),

    refresh: (datasetId: string, file: File): Promise<DatasetRefreshResult> => {
      const form = new FormData()
      form.append("file", file)
      return fetch(`${API_URL}/api/data/${datasetId}/refresh`, {
        method: "POST",
        body: form,
      }).then((r) => r.json())
    },

    getDictionary: (datasetId: string): Promise<DataDictionary> =>
      fetch(`${API_URL}/api/data/${datasetId}/dictionary`).then((r) => r.json()),

    generateDictionary: (datasetId: string): Promise<DataDictionary> =>
      fetch(`${API_URL}/api/data/${datasetId}/dictionary`, { method: "POST" }).then((r) =>
        r.json()
      ),

    getCrosstab: (
      datasetId: string,
      rows: string,
      cols: string,
      values?: string,
      agg: string = "sum"
    ): Promise<CrosstabResult> => {
      const params = new URLSearchParams({ rows, cols, agg })
      if (values) params.set("values", values)
      return fetch(`${API_URL}/api/data/${datasetId}/crosstab?${params}`).then((r) => r.json())
    },

    computeColumn: (
      datasetId: string,
      name: string,
      expression: string
    ): Promise<ComputeResult> =>
      fetch(`${API_URL}/api/data/${datasetId}/compute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, expression }),
      }).then((r) => r.json()),

    compareSegments: (
      datasetId: string,
      col: string,
      val1: string,
      val2: string
    ): Promise<SegmentComparisonResult> =>
      fetch(
        `${API_URL}/api/data/${datasetId}/compare-segments?col=${encodeURIComponent(col)}&val1=${encodeURIComponent(val1)}&val2=${encodeURIComponent(val2)}`
      ).then((r) => r.json()),

    getForecast: (
      datasetId: string,
      target?: string,
      periods?: number
    ): Promise<{ dataset_id: string; date_columns: string[]; value_columns: string[]; forecast: ForecastResult }> => {
      const params = new URLSearchParams()
      if (target) params.set("target", target)
      if (periods !== undefined) params.set("periods", String(periods))
      const qs = params.toString()
      return fetch(
        `${API_URL}/api/data/${datasetId}/forecast${qs ? `?${qs}` : ""}`
      ).then((r) => r.json())
    },

    getReadinessCheck: (
      datasetId: string,
      target?: string
    ): Promise<DataReadinessResult> => {
      const params = new URLSearchParams()
      if (target) params.set("target", target)
      const qs = params.toString()
      return fetch(
        `${API_URL}/api/data/${datasetId}/readiness-check${qs ? `?${qs}` : ""}`
      ).then((r) => r.json())
    },

    getTargetCorrelations: (
      datasetId: string,
      target: string,
      topN?: number
    ): Promise<TargetCorrelationResult> => {
      const params = new URLSearchParams({ target })
      if (topN !== undefined) params.set("top_n", String(topN))
      return fetch(
        `${API_URL}/api/data/${datasetId}/target-correlations?${params.toString()}`
      ).then((r) => r.json())
    },

    renameColumn: (
      datasetId: string,
      oldName: string,
      newName: string
    ): Promise<{ dataset_id: string; old_name: string; new_name: string; row_count: number; column_count: number }> =>
      fetch(`${API_URL}/api/data/${datasetId}/rename-column`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      }).then((r) => r.json()),

    getDataStory: (datasetId: string, target?: string): Promise<import("./types").DataStory> => {
      const params = new URLSearchParams()
      if (target) params.set("target", target)
      const qs = params.toString()
      return fetch(
        `${API_URL}/api/data/${datasetId}/story${qs ? `?${qs}` : ""}`
      ).then((r) => r.json())
    },

    setFilter: (
      datasetId: string,
      conditions: import("./types").FilterCondition[]
    ): Promise<import("./types").FilterSetResult> =>
      fetch(`${API_URL}/api/data/${datasetId}/set-filter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conditions }),
      }).then((r) => r.json()),

    clearFilter: (datasetId: string): Promise<{ dataset_id: string; cleared: boolean }> =>
      fetch(`${API_URL}/api/data/${datasetId}/clear-filter`, {
        method: "DELETE",
      }).then((r) => r.json()),

    getActiveFilter: (datasetId: string): Promise<import("./types").ActiveFilter> =>
      fetch(`${API_URL}/api/data/${datasetId}/active-filter`).then((r) => r.json()),

    getColumnProfile: (
      datasetId: string,
      col: string
    ): Promise<import("./types").ColumnProfile> =>
      fetch(
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
      return fetch(
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
      return fetch(`${API_URL}/api/data/${datasetId}/compare-time-windows?${params}`).then(
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
      return fetch(`${API_URL}/api/data/${datasetId}/top-n?${params}`).then((r) => {
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
      return fetch(`${API_URL}/api/data/${datasetId}/records?${params}`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
    },

    downloadDatasetUrl: (datasetId: string): string =>
      `${API_URL}/api/data/${datasetId}/download`,

    getSummaryStats: (
      datasetId: string
    ): Promise<import("./types").SummaryStatsResult> =>
      fetch(`${API_URL}/api/data/${datasetId}/summary-stats`).then((r) =>
        r.json()
      ),

    getValueCounts: (
      datasetId: string,
      col: string,
      n: number = 20
    ): Promise<import("./types").ValueCountResult> =>
      fetch(
        `${API_URL}/api/data/${datasetId}/value-counts?col=${encodeURIComponent(col)}&n=${n}`
      ).then((r) => r.json()),

    getPairCorrelation: (
      datasetId: string,
      col1: string,
      col2: string
    ): Promise<import("./types").PairCorrelationResult> =>
      fetch(
        `${API_URL}/api/data/${datasetId}/pair-correlation?col1=${encodeURIComponent(col1)}&col2=${encodeURIComponent(col2)}`
      ).then((r) => r.json()),

    getStatQuery: (
      datasetId: string,
      agg: string,
      col?: string
    ): Promise<import("./types").StatQueryResult> => {
      const params = new URLSearchParams({ agg })
      if (col) params.set("col", col)
      return fetch(
        `${API_URL}/api/data/${datasetId}/stat-query?${params.toString()}`
      ).then((r) => r.json())
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
      return fetch(
        `${API_URL}/api/data/${datasetId}/group-trends?${params.toString()}`
      ).then((r) => r.json())
    },

    predictionOpportunities: (
      datasetId: string
    ): Promise<import("./types").PredictionOpportunitiesResult> =>
      fetch(`${API_URL}/api/data/${datasetId}/prediction-opportunities`).then(
        (r) => r.json()
      ),

    compareDatasets: (
      baselineId: string,
      newId: string
    ): Promise<import("./types").DatasetComparisonResult> =>
      fetch(
        `${API_URL}/api/data/compare?baseline_id=${encodeURIComponent(baselineId)}&new_id=${encodeURIComponent(newId)}`
      ).then((r) => r.json()),
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

    // Pipeline step management (incremental add/undo)
    getSteps: (featureSetId: string): Promise<{
      feature_set_id: string
      step_count: number
      steps: Array<{ index: number; column: string; transform_type: string; params?: Record<string, unknown> }>
    }> =>
      fetch(`${API_URL}/api/features/${featureSetId}/steps`).then((r) => r.json()),

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
      fetch(`${API_URL}/api/features/${featureSetId}/steps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(step),
      }).then((r) => r.json()),

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
      fetch(`${API_URL}/api/features/${featureSetId}/steps/${stepIndex}`, {
        method: "DELETE",
      }).then((r) => r.json()),
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

    classImbalance: (
      projectId: string
    ): Promise<import("./types").ClassImbalanceResult> =>
      fetch(`${API_URL}/api/models/${projectId}/imbalance`).then((r) => r.json()),

    splitStrategy: (
      projectId: string
    ): Promise<import("./types").SplitStrategyInfo> =>
      fetch(`${API_URL}/api/models/${projectId}/split-strategy`).then((r) => r.json()),

    featureSelection: (
      runId: string
    ): Promise<import("./types").FeatureSelectionResult> =>
      fetch(`${API_URL}/api/models/${runId}/feature-selection`).then((r) => r.json()),

    calibration: (
      runId: string
    ): Promise<import("./types").CalibrationData> =>
      fetch(`${API_URL}/api/models/${runId}/calibration`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    improvementSuggestions: (
      projectId: string
    ): Promise<import("./types").ModelImprovementResult> =>
      fetch(`${API_URL}/api/models/${projectId}/improvement-suggestions`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    modelSelection: (
      projectId: string,
      criteria: import("./types").SelectionCriteria = "balanced"
    ): Promise<import("./types").ModelSelectionResult> =>
      fetch(
        `${API_URL}/api/models/${projectId}/model-selection?criteria=${encodeURIComponent(criteria)}`
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    crossModelFeatures: (projectId: string): Promise<import("./types").CrossModelFeatureResult> =>
      fetch(`${API_URL}/api/models/${projectId}/cross-model-features`).then((r) => {
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
      fetch(`${API_URL}/api/models/${projectId}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithms,
          imbalance_strategy: imbalanceStrategy ?? null,
          split_strategy: splitStrategy ?? null,
          excluded_features: excludedFeatures ?? null,
        }),
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

    exportModelCardUrl: (runId: string): string =>
      `${API_URL}/api/models/${runId}/export-model-card`,

    trainingStreamUrl: (projectId: string): string =>
      `${API_URL}/api/models/${projectId}/training-stream`,

    readiness: (modelRunId: string): Promise<import("./types").ModelReadiness> =>
      fetch(`${API_URL}/api/models/${modelRunId}/readiness`).then((r) => r.json()),

    tune: (modelRunId: string): Promise<TuningResult> =>
      fetch(`${API_URL}/api/models/${modelRunId}/tune`, { method: "POST" }).then((r) =>
        r.json()
      ),

    retrain: (projectId: string): Promise<import("./types").RetrainResponse> =>
      fetch(`${API_URL}/api/models/${projectId}/retrain`, { method: "POST" }).then((r) =>
        r.json()
      ),

    history: (projectId: string): Promise<ModelVersionHistory> =>
      fetch(`${API_URL}/api/models/${projectId}/history`).then((r) => r.json()),

    getModelCard: (projectId: string): Promise<import("./types").ModelCard> =>
      fetch(`${API_URL}/api/models/${projectId}/model-card`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getSegmentPerformance: (
      modelRunId: string,
      col: string,
    ): Promise<import("./types").SegmentPerformanceResult> =>
      fetch(
        `${API_URL}/api/models/${modelRunId}/segment-performance?col=${encodeURIComponent(col)}`,
      ).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getPredictionErrors: (
      modelRunId: string,
      n: number = 10,
    ): Promise<import("./types").PredictionErrorResult> =>
      fetch(`${API_URL}/api/models/${modelRunId}/prediction-errors?n=${n}`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),
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

    listByProject: (projectId: string): Promise<Deployment[]> =>
      fetch(`${API_URL}/api/deployments?project_id=${projectId}`).then((r) => r.json()),

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

    analytics: (deploymentId: string, days?: number): Promise<import("./types").DeploymentAnalytics> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/analytics${days ? `?days=${days}` : ""}`).then((r) => r.json()),

    logs: (deploymentId: string, limit?: number, offset?: number): Promise<import("./types").PredictionLogsResponse> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/logs?limit=${limit ?? 20}&offset=${offset ?? 0}`).then((r) => r.json()),

    drift: (deploymentId: string, window?: number): Promise<import("./types").DriftReport> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/drift${window ? `?window=${window}` : ""}`).then((r) => r.json()),

    sla: (deploymentId: string): Promise<import("./types").SlaData> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/sla`).then((r) => r.json()),

    whatif: (
      deploymentId: string,
      base: Record<string, unknown>,
      overrides: Record<string, unknown>
    ): Promise<import("./types").WhatIfResult> =>
      fetch(`${API_URL}/api/predict/${deploymentId}/whatif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base, overrides }),
      }).then((r) => r.json()),

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
      fetch(`${API_URL}/api/predict/${deploymentId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json()),

    feedbackAccuracy: (deploymentId: string): Promise<import("./types").FeedbackAccuracy> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/feedback-accuracy`).then((r) => r.json()),

    trainingVsProduction: (deploymentId: string): Promise<import("./types").ProdPerformanceResult> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/training-vs-production`).then((r) => r.json()),

    health: (deploymentId: string): Promise<import("./types").ModelHealth> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/health`).then((r) => r.json()),

    explain: (deploymentId: string, inputs: Record<string, unknown>): Promise<import("./types").PredictionExplanation> =>
      fetch(`${API_URL}/api/predict/${deploymentId}/explain`, {
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
      fetch(`${API_URL}/api/predict/${deploymentId}/scenarios`, {
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
      fetch(`${API_URL}/api/predict/compare`, {
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
      fetch(
        `${API_URL}/api/deploy/${deploymentId}/integration${baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : ""}`
      ).then((r) => r.json()),

    generateApiKey: (deploymentId: string): Promise<import("./types").ApiKeyResult> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/api-key`, { method: "POST" }).then(
        (r) => r.json()
      ),

    disableApiKey: (deploymentId: string): Promise<Response> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/api-key`, { method: "DELETE" }),

    getSchedules: (deploymentId: string): Promise<import("./types").BatchSchedule[]> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/schedules`).then((r) => r.json()),

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
      fetch(`${API_URL}/api/deploy/${deploymentId}/schedules`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    deleteSchedule: (deploymentId: string, scheduleId: string): Promise<void> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/schedules/${scheduleId}`, {
        method: "DELETE",
      }).then(() => undefined),

    triggerSchedule: (
      deploymentId: string,
      scheduleId: string
    ): Promise<{ status: string; schedule_id: string }> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/schedules/${scheduleId}/run`, {
        method: "POST",
      }).then((r) => r.json()),

    getScheduleRuns: (
      deploymentId: string,
      scheduleId: string
    ): Promise<import("./types").BatchJobRun[]> =>
      fetch(
        `${API_URL}/api/deploy/${deploymentId}/schedules/${scheduleId}/runs`
      ).then((r) => r.json()),

    getVersions: (
      deploymentId: string
    ): Promise<import("./types").DeploymentVersionHistory> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/versions`).then((r) => r.json()),

    rollback: (
      deploymentId: string,
      versionNumber: number
    ): Promise<import("./types").RollbackResult> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/rollback/${versionNumber}`, {
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
      fetch(`${API_URL}/api/deploy/${deploymentId}/webhooks`).then((r) =>
        r.json()
      ),

    createWebhook: (
      deploymentId: string,
      url: string,
      eventTypes: string[]
    ): Promise<import("./types").WebhookConfig> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/webhooks`, {
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
      fetch(
        `${API_URL}/api/deploy/${deploymentId}/webhooks/${webhookId}`,
        { method: "DELETE" }
      ).then(() => undefined),

    testWebhook: (
      deploymentId: string,
      webhookId: string
    ): Promise<import("./types").WebhookTestResult> =>
      fetch(
        `${API_URL}/api/deploy/${deploymentId}/webhooks/${webhookId}/test`,
        { method: "POST" }
      ).then((r) => r.json()),

    getAbTest: (
      deploymentId: string
    ): Promise<import("./types").ABTest> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/ab-test`).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    createAbTest: (
      deploymentId: string,
      challengerId: string,
      championSplitPct: number
    ): Promise<import("./types").ABTest> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/ab-test`, {
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
      fetch(`${API_URL}/api/deploy/${deploymentId}/ab-test`, {
        method: "DELETE",
      }).then(() => undefined),

    promoteChallenger: (
      deploymentId: string
    ): Promise<{ message: string; deployment: import("./types").Deployment }> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/ab-test/promote`, {
        method: "POST",
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    promoteToProduction: (
      deploymentId: string
    ): Promise<import("./types").EnvironmentPromotionResult> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/promote-to-production`, {
        method: "POST",
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    demoteToStaging: (
      deploymentId: string
    ): Promise<import("./types").EnvironmentPromotionResult> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/demote-to-staging`, {
        method: "POST",
      }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),

    getPresets: (
      deploymentId: string
    ): Promise<import("./types").DeploymentPreset[]> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/presets`).then((r) => r.json()),

    createPreset: (
      deploymentId: string,
      name: string,
      featureValues: Record<string, string | number>
    ): Promise<import("./types").DeploymentPreset> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/presets`, {
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
      fetch(`${API_URL}/api/deploy/${deploymentId}/presets/${presetId}`, {
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
      const res = await fetch(`${API_URL}/api/deploy/${deploymentId}/rate-limit`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rate_limit_rpm: rateLimitRpm, monthly_quota: monthlyQuota }),
      })
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    quotaStatus: async (deploymentId: string) => {
      const res = await fetch(`${API_URL}/api/deploy/${deploymentId}/quota-status`)
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    covariateDrift: async (
      deploymentId: string
    ): Promise<import("./types").CovariateDriftAlertResult> => {
      const res = await fetch(
        `${API_URL}/api/deploy/${deploymentId}/covariate-drift`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    predictionAudit: async (
      deploymentId: string
    ): Promise<import("./types").PredictionAuditResult> => {
      const res = await fetch(
        `${API_URL}/api/deploy/${deploymentId}/prediction-audit`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    getAlertRules: async (
      deploymentId: string
    ): Promise<{ count: number; rules: import("./types").AlertRuleEntry[] }> => {
      const res = await fetch(
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
      const res = await fetch(
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
      const res = await fetch(
        `${API_URL}/api/deploy/${deploymentId}/alert-rules/${ruleId}`,
        { method: "DELETE" }
      )
      if (!res.ok) throw new Error(await res.text())
    },

    accuracyAlertStatus: async (deploymentId: string) => {
      const res = await fetch(
        `${API_URL}/api/deploy/${deploymentId}/accuracy-alert-status`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    setAccuracyAlert: async (deploymentId: string, threshold: number | null) => {
      const res = await fetch(
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
      const res = await fetch(
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
      const res = await fetch(
        `${API_URL}/api/deploy/${deploymentId}/confidence-threshold-status`
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    listInputValidationRules: async (deploymentId: string) => {
      const res = await fetch(`${API_URL}/api/deploy/${deploymentId}/input-validation-rules`)
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
      const res = await fetch(`${API_URL}/api/deploy/${deploymentId}/input-validation-rules`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rule),
      })
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    deleteInputValidationRule: async (deploymentId: string, ruleId: string) => {
      const res = await fetch(
        `${API_URL}/api/deploy/${deploymentId}/input-validation-rules/${ruleId}`,
        { method: "DELETE" }
      )
      if (!res.ok) throw new Error(await res.text())
      return res.json()
    },

    getDashboardConfig: (deploymentId: string): Promise<import("@/lib/types").DashboardConfigResponse> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/dashboard-config`).then((r) => r.json()),

    getDashboardMetadata: (deploymentId: string): Promise<import("@/lib/types").DashboardMetadata> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/dashboard-metadata`).then((r) => r.json()),

    updateDashboardMetadata: async (
      deploymentId: string,
      opts: { title?: string; description?: string; clear?: boolean }
    ): Promise<import("@/lib/types").DashboardMetadata> => {
      const params = new URLSearchParams()
      if (opts.title !== undefined) params.set("title", opts.title)
      if (opts.description !== undefined) params.set("description", opts.description)
      if (opts.clear) params.set("clear", "true")
      const r = await fetch(
        `${API_URL}/api/deploy/${deploymentId}/dashboard-metadata?${params}`,
        { method: "PUT" }
      )
      return r.json()
    },

    getEmbedCode: (deploymentId: string): Promise<import("@/lib/types").EmbedCodeResult> =>
      fetch(`${API_URL}/api/deploy/${deploymentId}/embed-code`).then((r) => r.json()),

    getShareLink: (
      deploymentId: string,
      featureValues?: Record<string, string>
    ): Promise<import("@/lib/types").ShareLinkResult> => {
      const params = new URLSearchParams()
      if (featureValues && Object.keys(featureValues).length > 0) {
        params.set("features", JSON.stringify(featureValues))
      }
      const qs = params.toString()
      return fetch(
        `${API_URL}/api/deploy/${deploymentId}/share-link${qs ? `?${qs}` : ""}`
      ).then((r) => r.json())
    },
  },
}
