import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { EmptyState } from './components/EmptyState'
import { RunDetails } from './components/RunDetails'
import { RunFilters } from './components/RunFilters'
import { RunOverview } from './components/RunOverview'
import { RunsTable } from './components/RunsTable'
import { SectionTabs } from './components/SectionTabs'
import { fetchRunSummaries } from './services/runSummaries'
import type {
  DashboardSection,
  EvaluationMetricSet,
  RunSortKey,
  RunSummary,
} from './types/runSummary'
import {
  countSection,
  filterRuns,
  getFamilyOptions,
  getRunConfigurationOptions,
  sortRuns,
  sortVisibleRuns,
} from './utils/runSummary'

function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedPath, setSelectedPath] = useState<string>('')
  const [section, setSection] = useState<DashboardSection>('models')
  const [family, setFamily] = useState('')
  const [runConfiguration, setRunConfiguration] = useState('all')
  const [evaluationMetricSet, setEvaluationMetricSet] =
    useState<EvaluationMetricSet>('original')
  const [query, setQuery] = useState('')
  const [sortKey, setSortKey] = useState<RunSortKey>('newest')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isActive = true

    fetchRunSummaries()
      .then((response) => {
        if (!isActive) {
          return
        }
        const sortedRuns = sortRuns(response.runs)
        setRuns(sortedRuns)
        setSelectedPath('')
        setError(null)
      })
      .catch((unknownError: unknown) => {
        if (!isActive) {
          return
        }
        setError(
          unknownError instanceof Error
            ? unknownError.message
            : 'Unable to load model run summaries.',
        )
      })
      .finally(() => {
        if (isActive) {
          setIsLoading(false)
        }
      })

    return () => {
      isActive = false
    }
  }, [])

  const sectionTotal = useMemo(() => countSection(runs, section), [runs, section])
  const familyOptions = useMemo(
    () => getFamilyOptions(runs, section),
    [runs, section],
  )
  const runConfigurationOptions = useMemo(
    () => getRunConfigurationOptions(runs, section, family),
    [family, runs, section],
  )
  const filteredRuns = useMemo(
    () =>
      sortVisibleRuns(
        filterRuns(runs, {
          section,
          family,
          runConfiguration,
          evaluationMetricSet,
          query,
        }),
        sortKey,
        evaluationMetricSet,
      ),
    [evaluationMetricSet, family, query, runConfiguration, runs, section, sortKey],
  )
  const selectedRun = useMemo(
    () =>
      filteredRuns.find((run) => run.relativePath === selectedPath) ??
      filteredRuns[0] ??
      null,
    [filteredRuns, selectedPath],
  )

  function handleSectionChange(nextSection: DashboardSection) {
    setSection(nextSection)
    setRunConfiguration('all')
    const nextFamily = nextSection === 'embeddings' ? 'embeddings_runs' : ''
    setFamily(nextFamily)
    const firstRun = runs.find(
      (run) => run.section === nextSection && run.family === nextFamily,
    )
    setSelectedPath(firstRun?.relativePath ?? '')
  }

  const needsModelFamilySelection = section === 'models' && !family

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Sentiment run analytics</p>
          <h1>Run Dashboard</h1>
        </div>
        <div className="header-stat">
          <span>{runs.length}</span>
          <small>metadata runs</small>
        </div>
      </header>

      <SectionTabs
        activeSection={section}
        embeddingCount={countSection(runs, 'embeddings')}
        modelCount={countSection(runs, 'models')}
        onChange={handleSectionChange}
      />

      <RunFilters
        section={section}
        families={familyOptions}
        family={family}
        runConfiguration={runConfiguration}
        runConfigurationOptions={runConfigurationOptions}
        query={query}
        sortKey={sortKey}
        evaluationMetricSet={evaluationMetricSet}
        onFamilyChange={(value) => {
          setFamily(value)
          setRunConfiguration('all')
        }}
        onRunConfigurationChange={setRunConfiguration}
        onEvaluationMetricSetChange={setEvaluationMetricSet}
        onQueryChange={setQuery}
        onSortKeyChange={setSortKey}
      />

      {isLoading && <EmptyState title="Loading run metadata" />}
      {error && <EmptyState title="Could not load runs" detail={error} />}
      {!isLoading && !error && needsModelFamilySelection && (
        <EmptyState
          title="Select a model family"
          detail="Choose ML Models, DL Models, or Fine Tuned Models to render the table and details."
        />
      )}

      {!isLoading && !error && !needsModelFamilySelection && (
        <section className="dashboard-grid">
          <div className="runs-column">
            <RunOverview
              runs={filteredRuns}
              section={section}
              totalRuns={sectionTotal}
            />
            <RunsTable
              runs={filteredRuns}
              selectedPath={selectedRun?.relativePath ?? ''}
              evaluationMetricSet={evaluationMetricSet}
              onSelectRun={setSelectedPath}
            />
          </div>
          <RunDetails run={selectedRun} evaluationMetricSet={evaluationMetricSet} />
        </section>
      )}
    </main>
  )
}

export default App
