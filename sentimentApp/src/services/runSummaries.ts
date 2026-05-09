import type { RunSummariesResponse } from '../types/runSummary'

export async function fetchRunSummaries(): Promise<RunSummariesResponse> {
  const summariesUrl = import.meta.env.DEV
    ? '/api/run-summaries'
    : `${import.meta.env.BASE_URL}run-summaries.json`
  const response = await fetch(summariesUrl)

  if (!response.ok) {
    throw new Error(`Run summary request failed with ${response.status}`)
  }

  return response.json() as Promise<RunSummariesResponse>
}
