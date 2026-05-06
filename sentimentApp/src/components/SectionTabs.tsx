import type { DashboardSection } from '../types/runSummary'

type SectionTabsProps = {
  activeSection: DashboardSection
  embeddingCount: number
  modelCount: number
  onChange: (section: DashboardSection) => void
}

export function SectionTabs({
  activeSection,
  embeddingCount,
  modelCount,
  onChange,
}: SectionTabsProps) {
  return (
    <div className="section-tabs" role="tablist" aria-label="Dashboard sections">
      <button
        type="button"
        className={activeSection === 'models' ? 'active' : ''}
        onClick={() => onChange('models')}
      >
        <span>Models</span>
        <strong>{modelCount}</strong>
      </button>
      <button
        type="button"
        className={activeSection === 'embeddings' ? 'active' : ''}
        onClick={() => onChange('embeddings')}
      >
        <span>Embedding Runs</span>
        <strong>{embeddingCount}</strong>
      </button>
    </div>
  )
}
