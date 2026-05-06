import type { RunSummary } from '../types/runSummary'

function safeFileName(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

export function exportRunConfiguration(run: RunSummary) {
  const payload = {
    displayName: run.displayName,
    section: run.section,
    family: run.family,
    runId: run.runId,
    status: run.status,
    generatedAt: run.generatedAt,
    sourceMetadataPath: run.relativePath,
    evaluationMetrics: run.metrics,
    modelConfiguration: run.modelConfiguration,
    runConfiguration: run.runConfiguration,
    embeddingConfiguration: run.embeddingConfiguration,
  }

  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${safeFileName(run.displayName || run.runId)}-configuration.json`
  anchor.click()
  URL.revokeObjectURL(url)
}
