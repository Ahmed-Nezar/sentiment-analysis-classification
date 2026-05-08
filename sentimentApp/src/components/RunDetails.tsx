import type {
  DetailItem,
  EvaluationMetricSet,
  RunSummary,
} from '../types/runSummary'
import { exportRunConfiguration } from '../utils/exportRunConfiguration'
import {
  formatLabel,
  formatMetric,
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

export function RunDetails({ run, evaluationMetricSet }: RunDetailsProps) {
  if (!run) {
    return (
      <aside className="details-panel">
        <h2>Select a run</h2>
      </aside>
    )
  }
  const activeMetrics = getRunMetrics(run, evaluationMetricSet)
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
