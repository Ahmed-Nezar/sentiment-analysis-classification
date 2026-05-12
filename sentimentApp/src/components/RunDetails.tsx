import { useState } from 'react'
import type {
  DetailItem,
  EvaluationDetails,
  EvaluationMetricSet,
  RunSummary,
} from '../types/runSummary'
import { exportRunConfiguration } from '../utils/exportRunConfiguration'
import {
  formatLabel,
  formatMetric,
  getRunEvaluationDetails,
  getRunMetrics,
  hasNoiseRemovedEvaluation,
} from '../utils/runSummary'

type RunDetailsProps = {
  run: RunSummary | null
  evaluationMetricSet: EvaluationMetricSet
}

function DetailSection({
  title,
  items,
}: {
  title: string
  items: DetailItem[]
}) {
  if (items.length === 0) {
    return null
  }

  return (
    <section className="detail-card">
      <h3>{title}</h3>
      <dl className="detail-list">
        {items.map((item) => (
          <div key={`${item.label}-${item.value}`}>
            <dt>{formatLabel(item.label)}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

function EvaluationDiagnostics({
  details,
}: {
  details: EvaluationDetails
}) {
  const maxCell = Math.max(...details.confusionMatrix.flat(), 1)

  return (
    <section className="detail-card diagnostics-card">
      <h3>Confusion Matrix</h3>
      <div className="confusion-shell">
        <table className="confusion-matrix">
          <thead>
            <tr>
              <th scope="col">True Class</th>
              {details.labels.map((label) => (
                <th key={label} scope="col">
                  Predicted Class {formatLabel(label)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {details.confusionMatrix.map((row, rowIndex) => (
              <tr key={details.labels[rowIndex] ?? rowIndex}>
                <th scope="row">{formatLabel(details.labels[rowIndex] ?? `class ${rowIndex}`)}</th>
                {row.map((value, columnIndex) => {
                  const intensity = value / maxCell

                  return (
                    <td
                      key={`${rowIndex}-${columnIndex}`}
                      style={{
                        backgroundColor: `rgba(49, 104, 142, ${0.08 + intensity * 0.72})`,
                        color: intensity > 0.56 ? '#fff' : undefined,
                      }}
                    >
                      {value}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Scores Per Class</h3>
      <div className="class-score-list">
        {details.classScores.map((score) => (
          <article key={score.label} className="class-score-card">
            <div>
              <span>{formatLabel(score.label)}</span>
              <strong>{formatPercent(score.f1)}</strong>
            </div>
            <dl>
              <div>
                <dt>Precision</dt>
                <dd>{formatPercent(score.precision)}</dd>
              </div>
              <div>
                <dt>Recall</dt>
                <dd>{formatPercent(score.recall)}</dd>
              </div>
              <div>
                <dt>F1</dt>
                <dd>{formatPercent(score.f1)}</dd>
              </div>
              <div>
                <dt>Support</dt>
                <dd>{score.support}</dd>
              </div>
            </dl>
            <meter min="0" max="1" value={score.f1} />
          </article>
        ))}
      </div>
    </section>
  )
}

export function RunDetails({ run, evaluationMetricSet }: RunDetailsProps) {
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState(false)

  if (!run) {
    return (
      <aside className="details-panel">
        <h2>Select a run</h2>
      </aside>
    )
  }
  const activeMetrics = getRunMetrics(run, evaluationMetricSet)
  const evaluationDetails = getRunEvaluationDetails(run, evaluationMetricSet)
  const isNoiseRemovedEvaluation =
    evaluationMetricSet === 'without_noise' && hasNoiseRemovedEvaluation(run)

  return (
    <aside className="details-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{formatLabel(run.family)}</p>
          <h2>{run.displayName}</h2>
        </div>
        <div className="panel-actions">
          {run.status && <span className="kind-badge">{formatLabel(run.status)}</span>}
          <button
            type="button"
            className="export-button"
            onClick={() => exportRunConfiguration(run, evaluationMetricSet)}
          >
            Export JSON
          </button>
        </div>
      </div>

      {evaluationDetails && (
        <div className="detail-view-toggle">
          <button
            type="button"
            className="is-active"
            onClick={() => setIsDiagnosticsOpen(true)}
          >
            Open Diagnostics
          </button>
        </div>
      )}

      {isDiagnosticsOpen && evaluationDetails && (
        <div
          className="diagnostics-modal-backdrop"
          role="presentation"
          onClick={() => setIsDiagnosticsOpen(false)}
        >
          <section
            className="diagnostics-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="diagnostics-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="diagnostics-modal-header">
              <div>
                <p className="eyebrow">{formatLabel(run.family)}</p>
                <h2 id="diagnostics-title">Diagnostics</h2>
                <p>{run.displayName}</p>
              </div>
              <button
                type="button"
                className="modal-close-button"
                aria-label="Close diagnostics"
                onClick={() => setIsDiagnosticsOpen(false)}
              >
                Close
              </button>
            </div>
            <EvaluationDiagnostics details={evaluationDetails} />
          </section>
        </div>
      )}

      {Object.keys(activeMetrics).length > 0 && (
        <section className="detail-card">
          <h3>
            {isNoiseRemovedEvaluation
              ? 'Evaluation Metrics Without Noise'
              : 'Evaluation Metrics'}
          </h3>
          {isNoiseRemovedEvaluation && run.trainedOnNoisyData && (
            <p className="detail-note">
              Scores exclude noisy evaluation rows. This model was trained on
              {run.datasetName ? ` ${run.datasetName}` : ' a noisy dataset'}.
            </p>
          )}
          {isNoiseRemovedEvaluation && run.trainedOnNoisyData === false && (
            <p className="detail-note">
              Scores come from the metadata evaluation for
              {run.datasetName ? ` ${run.datasetName}` : ' a noise removed dataset'}.
            </p>
          )}
          <div className="metric-grid">
            {Object.entries(activeMetrics).map(([key, value]) => (
              <div key={key}>
                <span>{formatLabel(key)}</span>
                <strong>{formatMetric(value)}</strong>
              </div>
            ))}
          </div>
        </section>
      )}

      <DetailSection title="Model Configuration" items={run.modelConfiguration} />
      <DetailSection title="Run Configuration" items={run.runConfiguration} />
      <DetailSection
        title={
          run.section === 'embeddings'
            ? 'Embedding Configuration'
            : 'Used Embedding'
        }
        items={run.embeddingConfiguration}
      />
    </aside>
  )
}
