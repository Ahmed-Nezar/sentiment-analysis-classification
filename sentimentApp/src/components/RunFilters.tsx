import type { DashboardSection, RunSortKey } from '../types/runSummary'
import { formatLabel } from '../utils/runSummary'

type RunFiltersProps = {
  section: DashboardSection
  families: string[]
  family: string
  runConfiguration: string
  runConfigurationOptions: { label: string; value: string }[]
  query: string
  sortKey: RunSortKey
  onFamilyChange: (value: string) => void
  onRunConfigurationChange: (value: string) => void
  onQueryChange: (value: string) => void
  onSortKeyChange: (value: RunSortKey) => void
}

export function RunFilters({
  section,
  families,
  family,
  runConfiguration,
  runConfigurationOptions,
  query,
  sortKey,
  onFamilyChange,
  onRunConfigurationChange,
  onQueryChange,
  onSortKeyChange,
}: RunFiltersProps) {
  return (
    <section className="filter-bar" aria-label="Run filters">
      <label>
        <span>Search</span>
        <input
          type="search"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Model, embedding, metric, configuration"
        />
      </label>

      <label>
        <span>Run family</span>
        <select
          value={family}
          onChange={(event) => onFamilyChange(event.target.value)}
          disabled={section === 'embeddings'}
        >
          {section === 'models' && (
            <option value="">Select model family</option>
          )}
          {families.map((option) => (
            <option key={option} value={option}>
              {formatLabel(option)}
            </option>
          ))}
        </select>
      </label>

      <label>
        <span>Run configuration</span>
        <select
          value={runConfiguration}
          onChange={(event) => onRunConfigurationChange(event.target.value)}
        >
          <option value="all">All configurations</option>
          {runConfigurationOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      <label>
        <span>Sort by</span>
        <select
          value={sortKey}
          onChange={(event) => onSortKeyChange(event.target.value as RunSortKey)}
        >
          <option value="newest">Newest</option>
          <option value="accuracy_desc">Accuracy high to low</option>
          <option value="accuracy_asc">Accuracy low to high</option>
          <option value="name">Name</option>
        </select>
      </label>
    </section>
  )
}
