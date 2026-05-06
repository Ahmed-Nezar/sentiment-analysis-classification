export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

export type MetricMap = Record<string, number | string>

export type DetailItem = {
  label: string
  value: string
}

export type DashboardSection = 'models' | 'embeddings'
export type RunSortKey = 'newest' | 'name' | 'accuracy_desc' | 'accuracy_asc'

export type RunSummary = {
  relativePath: string
  section: DashboardSection
  family: string
  runId: string
  displayName: string
  generatedAt?: string
  status?: string
  metrics: MetricMap
  modelConfiguration: DetailItem[]
  runConfiguration: DetailItem[]
  embeddingConfiguration: DetailItem[]
}

export type RunSummariesResponse = {
  modelsRoot: string
  runs: RunSummary[]
}

export type RunFiltersState = {
  section: DashboardSection
  family: string
  runConfiguration: string
  query: string
}
