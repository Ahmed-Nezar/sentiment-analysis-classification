import { useEffect, useMemo, useState, type FormEvent } from 'react'
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

type AppView = 'landing' | 'dashboard' | 'inference'
type SentimentLabel = 'negative' | 'neutral' | 'positive'
type EncoderClassificationResponse = {
  predicted_class_id: number
  probability?: number[]
  probablity?: number[]
}
type DecoderClassificationResponse = {
  text: string
  label: SentimentLabel
  raw_output: string
}
type InferenceModelId = 'bge-m3' | 'qwen-4b'
type InferenceModel = {
  id: InferenceModelId
  name: string
  runId: string
  apiTarget: 'api1' | 'api2'
  isAvailable: boolean
}
type InferenceResult = {
  modelId: InferenceModelId
  label: SentimentLabel
  text?: string
  classId?: number
  probability?: number[]
  rawOutput?: string
}

const TEXT_CLASSIFICATION_URL = __TEXT_CLASSIFICATION_URL__
function buildTextClassificationUrl(apiTarget: InferenceModel['apiTarget']) {
  if (!TEXT_CLASSIFICATION_URL) {
    return ''
  }

  const url = new URL(TEXT_CLASSIFICATION_URL, window.location.origin)
  url.searchParams.set('api', apiTarget)
  return url.toString()
}

const SENTIMENT_LABELS: Record<number, SentimentLabel> = {
  0: 'negative',
  1: 'neutral',
  2: 'positive',
}
const SENTIMENT_LABELS_LIST = Object.values(SENTIMENT_LABELS)
const INFERENCE_MODELS: InferenceModel[] = [
  {
    id: 'bge-m3',
    name: 'BAAI/bge-m3',
    runId: 'baai_bge_m3_20260509_002022',
    apiTarget: 'api1',
    isAvailable: true,
  },
  {
    id: 'qwen-4b',
    name: 'Qwen/Qwen3-4B-Instruct-2507',
    runId: 'Qwen_Qwen3-4B-Instruct-2507_20260509_161431',
    apiTarget: 'api2',
    isAvailable: true,
  },
]

function LandingPage({ onNavigate }: { onNavigate: (view: AppView) => void }) {
  return (
    <main className="landing-shell">
      <section className="landing-hero">
        <div className="landing-copy">
          <p className="eyebrow">SentimentFlow</p>
          <h1>Explore the experiment results or test the classifier.</h1>
          <p className="landing-lede">
            Open the model run dashboard for training metrics and metadata, or
            move into the inference workspace when you are ready to try samples.
          </p>
          <div className="landing-actions" aria-label="Primary navigation">
            <button type="button" onClick={() => onNavigate('dashboard')}>
              Open Dashboard
            </button>
            <button type="button" onClick={() => onNavigate('inference')}>
              Test Inference
            </button>
          </div>
        </div>
        <div className="landing-panel" aria-label="Available workspaces">
          <button
            type="button"
            className="workspace-card"
            onClick={() => onNavigate('dashboard')}
          >
            <span>Dashboard</span>
            <strong>Compare runs, metrics, families, and configurations.</strong>
            <small>Best for model selection and experiment review.</small>
          </button>
          <button
            type="button"
            className="workspace-card"
            onClick={() => onNavigate('inference')}
          >
            <span>Inference</span>
            <strong>Send text through the sentiment model test surface.</strong>
            <small>Implementation details can plug in here next.</small>
          </button>
        </div>
      </section>
    </main>
  )
}

function InferencePage({ onBack }: { onBack: () => void }) {
  const [selectedModelId, setSelectedModelId] =
    useState<InferenceModelId>('bge-m3')
  const [text, setText] = useState('this is a good movie!')
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const selectedModel =
    INFERENCE_MODELS.find((model) => model.id === selectedModelId) ??
    INFERENCE_MODELS[0]

  function handleModelChange(modelId: InferenceModelId) {
    setSelectedModelId(modelId)
    setResult(null)
    setError(null)
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmedText = text.trim()

    if (!selectedModel.isAvailable) {
      setResult(null)
      setError('This model is listed, but its inference API is not wired yet.')
      return
    }

    const endpoint = buildTextClassificationUrl(selectedModel.apiTarget)

    if (!endpoint) {
      setResult(null)
      setError('TEXT_CLASSIFICATION_URL is not configured for this build.')
      return
    }

    if (!trimmedText) {
      setResult(null)
      setError('Enter text to classify.')
      return
    }

    setIsPredicting(true)
    setError(null)

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: trimmedText }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`)
      }

      if (selectedModel.id === 'qwen-4b') {
        const payload = (await response.json()) as DecoderClassificationResponse

        if (!SENTIMENT_LABELS_LIST.includes(payload.label) || !payload.raw_output) {
          throw new Error('The Qwen model returned an unexpected response.')
        }

        setResult({
          modelId: selectedModel.id,
          label: payload.label,
          text: payload.text,
          rawOutput: payload.raw_output,
        })
        return
      }

      const payload = (await response.json()) as EncoderClassificationResponse
      const label = SENTIMENT_LABELS[payload.predicted_class_id]
      const probability = payload.probability ?? payload.probablity

      if (!label || !Array.isArray(probability)) {
        throw new Error('The BGE-m3 model returned an unexpected response.')
      }

      setResult({
        modelId: selectedModel.id,
        label,
        classId: payload.predicted_class_id,
        probability,
      })
    } catch (unknownError) {
      setResult(null)
      setError(
        unknownError instanceof Error
          ? unknownError.message
          : 'Unable to classify this text.',
      )
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Sentiment inference</p>
          <h1>Testing Workspace</h1>
        </div>
        <button type="button" className="nav-button" onClick={onBack}>
          Back Home
        </button>
      </header>
      <section className="inference-grid">
        <form className="inference-form" onSubmit={handleSubmit}>
          <div>
            <p className="eyebrow">Selected model</p>
            <h2>Text Classification</h2>
            <p>
              {selectedModel.name}
              <br />
              Run ID: <strong>{selectedModel.runId}</strong>
            </p>
          </div>
          <fieldset className="model-selector">
            <legend>Model</legend>
            {INFERENCE_MODELS.map((model) => (
              <label key={model.id}>
                <input
                  type="radio"
                  name="inference-model"
                  value={model.id}
                  checked={selectedModelId === model.id}
                  onChange={() => handleModelChange(model.id)}
                />
                <span>
                  <strong>{model.name}</strong>
                  <small>{model.runId}</small>
                </span>
              </label>
            ))}
          </fieldset>
          <label>
            Text to classify
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={8}
              placeholder="Type a review, comment, or sentence..."
            />
          </label>
          <button type="submit" disabled={isPredicting}>
            {isPredicting ? 'Classifying...' : 'Classify Text'}
          </button>
          {error && <p className="form-error">{error}</p>}
        </form>

        <aside className="inference-result">
          <div className="result-heading">
            <span>Prediction</span>
            <strong>{result ? result.label : 'Waiting for input'}</strong>
          </div>
          {result ? (
            <>
              {result.modelId === 'qwen-4b' ? (
                <div className="decoder-output">
                  <div>
                    <span>Input text</span>
                    <p>{result.text}</p>
                  </div>
                  <div>
                    <span>Raw output</span>
                    <pre>{result.rawOutput}</pre>
                  </div>
                </div>
              ) : (
                <>
                  <p className="muted">Predicted class ID: {result.classId}</p>
                  <div className="probability-list">
                    {result.probability?.map((score, index) => {
                      const label = SENTIMENT_LABELS[index] ?? `class ${index}`

                      return (
                        <div key={label}>
                          <span>{label}</span>
                          <strong>{(score * 100).toFixed(2)}%</strong>
                          <meter min="0" max="1" value={score} />
                        </div>
                      )
                    })}
                  </div>
                </>
              )}
            </>
          ) : (
            <p className="muted">
              Submit text to see the predicted sentiment and probability scores.
            </p>
          )}
        </aside>
      </section>
    </main>
  )
}

function DashboardPage({ onBack }: { onBack: () => void }) {
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
        <div className="header-actions">
          <button type="button" className="nav-button" onClick={onBack}>
            Back Home
          </button>
          <div className="header-stat">
            <span>{runs.length}</span>
            <small>metadata runs</small>
          </div>
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

function App() {
  const [view, setView] = useState<AppView>('landing')

  if (view === 'dashboard') {
    return <DashboardPage onBack={() => setView('landing')} />
  }

  if (view === 'inference') {
    return <InferencePage onBack={() => setView('landing')} />
  }

  return <LandingPage onNavigate={setView} />
}

export default App
