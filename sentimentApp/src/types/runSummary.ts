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

export type ClassScore = {
  label: string
  precision: number
  recall: number
  f1: number
  support: number
}

export type EvaluationDetails = {
  labels: string[]
  confusionMatrix: number[][]
  classScores: ClassScore[]
}

export type DashboardSection = 'models' | 'embeddings'
export type RunSortKey = 'newest' | 'name' | 'accuracy_desc' | 'accuracy_asc'
export type EvaluationMetricSet = 'original' | 'without_noise'

export type RunSummary = {
  relativePath: string
  section: DashboardSection
  family: string
  runId: string
  displayName: string
  generatedAt?: string
  status?: string
  metrics: MetricMap
  metricsWithoutNoise: MetricMap
  evaluationDetails?: EvaluationDetails
  evaluationDetailsWithoutNoise?: EvaluationDetails
  datasetName?: string
  trainedOnNoisyData?: boolean
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
  evaluationMetricSet: EvaluationMetricSet
  query: string
}
