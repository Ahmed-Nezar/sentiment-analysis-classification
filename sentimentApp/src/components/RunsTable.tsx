import type { EvaluationMetricSet, RunSummary } from '../types/runSummary'
import {
  extractPrimaryMetrics,
  formatLabel,
  formatMetric,
  getRunMetrics,
} from '../utils/runSummary'

type RunsTableProps = {
  runs: RunSummary[]
  selectedPath: string
  evaluationMetricSet: EvaluationMetricSet
  onSelectRun: (relativePath: string) => void
}

export function RunsTable({
  runs,
  selectedPath,
  evaluationMetricSet,
  onSelectRun,
}: RunsTableProps) {
  if (runs.length === 0) {
    return (
      <section className="table-wrap table-empty">
        <h2>No matching runs</h2>
      </section>
    )
  }

  return (
    <section className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Family</th>
            <th>Run ID</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const primaryMetrics = extractPrimaryMetrics(
              getRunMetrics(run, evaluationMetricSet),
            )
            return (
              <tr
                key={run.relativePath}
                className={run.relativePath === selectedPath ? 'selected' : ''}
                onClick={() => onSelectRun(run.relativePath)}
              >
                <td>
                  <button
                    type="button"
                    className="table-run-button"
                    onClick={() => onSelectRun(run.relativePath)}
                  >
                    <strong>{run.displayName}</strong>
                    <span>{run.relativePath}</span>
                    {run.datasetName && <span>Dataset: {run.datasetName}</span>}
                  </button>
                </td>
                <td>{formatLabel(run.family)}</td>
                <td>{run.runId}</td>
                <td>
                  <div className="metric-pills">
                    {Object.keys(primaryMetrics).length === 0 && (
                      <span className="muted">No evaluation metrics</span>
                    )}
                    {Object.entries(primaryMetrics)
                      .slice(0, 3)
                      .map(([key, value]) => (
                        <span className="pill" key={key}>
                          {formatLabel(key)} {formatMetric(value)}
                        </span>
                      ))}
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}
