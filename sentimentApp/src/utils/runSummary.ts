import type {
  DashboardSection,
  MetricMap,
  RunFiltersState,
  RunSortKey,
  RunSummary,
} from '../types/runSummary'

const METRIC_KEYS = [
  'accuracy',
  'precision',
  'recall',
  'f1',
  'loss',
]
const RUN_CONFIGURATION_FILTER_LABELS = new Set([
  'status',
  'epochs',
  'early_stopping',
  'early_stopped_at_epoch',
  'early_stopping_reason',
  'hyperparameter_optimization',
  'embedding_mode',
  'embedding_name',
  'embedding_type',
  'output_vector_dimension',
  'test_size',
  'random_state',
  'stratify',
])

export function formatLabel(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function formatMetric(value: number | string): string {
  if (typeof value === 'number') {
    if (Math.abs(value) <= 1) {
      return value.toFixed(4)
    }
    return Number.isInteger(value) ? String(value) : value.toFixed(2)
  }
  return value
}

export function extractPrimaryMetrics(metrics: MetricMap): MetricMap {
  return Object.fromEntries(
    Object.entries(metrics).filter(([key]) =>
      METRIC_KEYS.some((metricKey) => key.toLowerCase().includes(metricKey)),
    ),
  )
}

export function sortRuns(runs: RunSummary[]): RunSummary[] {
  return [...runs].sort((first, second) => {
    const firstDate = Date.parse(first.generatedAt ?? '')
    const secondDate = Date.parse(second.generatedAt ?? '')
    if (!Number.isNaN(firstDate) && !Number.isNaN(secondDate)) {
      return secondDate - firstDate
    }
    return first.displayName.localeCompare(second.displayName)
  })
}

function getAccuracy(run: RunSummary): number | null {
  const value = run.metrics.accuracy
  if (typeof value === 'number') {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function sortVisibleRuns(runs: RunSummary[], sortKey: RunSortKey): RunSummary[] {
  return [...runs].sort((first, second) => {
    if (sortKey === 'accuracy_desc' || sortKey === 'accuracy_asc') {
      const firstAccuracy = getAccuracy(first)
      const secondAccuracy = getAccuracy(second)
      if (firstAccuracy !== null && secondAccuracy !== null) {
        return sortKey === 'accuracy_desc'
          ? secondAccuracy - firstAccuracy
          : firstAccuracy - secondAccuracy
      }
      if (firstAccuracy !== null) {
        return -1
      }
      if (secondAccuracy !== null) {
        return 1
      }
    }

    if (sortKey === 'name') {
      return first.displayName.localeCompare(second.displayName)
    }

    return sortRuns([first, second])[0] === first ? -1 : 1
  })
}

export function filterRuns(
  runs: RunSummary[],
  filters: RunFiltersState,
): RunSummary[] {
  const normalizedQuery = filters.query.trim().toLowerCase()

  return runs.filter((run) => {
    if (run.section !== filters.section) {
      return false
    }
    if (!filters.family) {
      return false
    }
    if (filters.family !== 'all' && run.family !== filters.family) {
      return false
    }
    if (
      filters.runConfiguration !== 'all' &&
      !run.runConfiguration.some(
        (item) => getRunConfigurationOptionValue(item) === filters.runConfiguration,
      )
    ) {
      return false
    }
    if (!normalizedQuery) {
      return true
    }

    const details = [
      ...run.modelConfiguration,
      ...run.runConfiguration,
      ...run.embeddingConfiguration,
    ]
      .map((item) => `${item.label} ${item.value}`)
      .join(' ')

    const searchable = [
      run.relativePath,
      run.displayName,
      run.runId,
      run.family,
      details,
    ]
      .join(' ')
      .toLowerCase()

    return searchable.includes(normalizedQuery)
  })
}

export function getFamilyOptions(runs: RunSummary[], section: DashboardSection) {
  return [...new Set(runs.filter((run) => run.section === section).map((run) => run.family))].sort()
}

function getRunConfigurationOptionValue(item: { label: string; value: string }) {
  return `${item.label}::${item.value}`
}

export function getRunConfigurationOptions(
  runs: RunSummary[],
  section: DashboardSection,
  family: string,
) {
  const allowedLabels =
    section === 'embeddings'
      ? new Set(['output_vector_dimension'])
      : RUN_CONFIGURATION_FILTER_LABELS
  const options = new Map<string, { label: string; value: string }>()

  runs
    .filter((run) => run.section === section)
    .filter((run) => family === 'all' || run.family === family)
    .flatMap((run) => run.runConfiguration)
    .filter((item) => allowedLabels.has(item.label))
    .forEach((item) => {
      const value = getRunConfigurationOptionValue(item)
      options.set(value, {
        label: `${formatLabel(item.label)}: ${item.value}`,
        value,
      })
    })

  return [...options.values()].sort((first, second) =>
    first.label.localeCompare(second.label),
  )
}

export function countSection(runs: RunSummary[], section: DashboardSection) {
  return runs.filter((run) => run.section === section).length
}
