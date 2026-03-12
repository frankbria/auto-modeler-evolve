export interface Project {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
  status: "exploring" | "modeling" | "deployed"
}

export interface ColumnStat {
  name: string
  dtype: string
  non_null_count: number
  null_count: number
  null_pct: number
  unique_count: number
  min?: number | null
  max?: number | null
  mean?: number | null
  std?: number | null
  sample_values: (string | number | null)[]
}

export interface Dataset {
  id: string
  project_id: string
  filename: string
  row_count: number
  column_count: number
  uploaded_at: string
}

export interface UploadResponse {
  dataset_id: string
  filename: string
  row_count: number
  column_count: number
  preview: Record<string, unknown>[]
  column_stats: ColumnStat[]
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  timestamp: string
}
