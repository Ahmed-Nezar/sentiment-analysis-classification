import type { RunSummariesResponse } from '../types/runSummary'

export async function fetchRunSummaries(): Promise<RunSummariesResponse> {
  const response = await fetch('/api/run-summaries')

  if (!response.ok) {
    throw new Error(`Run summary request failed with ${response.status}`)
  }

  return response.json() as Promise<RunSummariesResponse>
}
