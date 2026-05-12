import { promises as fs } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Plugin } from 'vite'

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

type JsonObject = { [key: string]: JsonValue }
type DetailSource = { [key: string]: JsonValue | undefined }
type DetailItem = { label: string; value: string }
type EvaluationDetails = {
  labels: string[]
  confusionMatrix: number[][]
  classScores: {
    label: string
    precision: number
    recall: number
    f1: number
    support: number
  }[]
}

const appDir = path.dirname(fileURLToPath(import.meta.url))
const modelsRoot = path.resolve(appDir, '../models')
const envRoot = path.resolve(appDir, '..')
const OMITTED_KEYS = new Set([
  'metadata_path',
  'model_path',
  'model_dir',
  'tokenizer_dir',
  'predictions_path',
  'targets_path',
  'output_dir',
  'embedding_object_path',
  'x_train_path',
  'x_test_path',
  'y_train_path',
  'y_test_path',
  'train_split_path',
  'test_split_path',
  'configuration_hash_payload',
  'history',
  'metrics',
  'final_metrics',
  'metrics_without_noise',
])

function isJsonObject(value: JsonValue | undefined): value is JsonObject {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function readString(data: JsonObject, key: string): string | undefined {
  const value = data[key]
  return typeof value === 'string' ? value : undefined
}

function compactPath(value: string): string {
  return value.replaceAll('\\', '/').split('/').slice(-2).join('/')
}

function inferBasePath(env: Record<string, string | undefined>): string {
  if (env.VITE_BASE_PATH) {
    return env.VITE_BASE_PATH
  }

  const repository = process.env.GITHUB_REPOSITORY
  if (!process.env.GITHUB_ACTIONS || !repository) {
    return '/'
  }

  const repositoryName = repository.split('/').at(-1) ?? ''
  return repositoryName.endsWith('.github.io') ? '/' : `/${repositoryName}/`
}

function baseName(value: string | undefined): string | undefined {
  if (!value) {
    return undefined
  }
  const normalized = value.replaceAll('\\', '/')
  return normalized.split('/').filter(Boolean).at(-1)
}

function formatValue(value: JsonValue): string {
  if (value === null) {
    return 'None'
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No'
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)))
  }
  if (typeof value === 'string') {
    return value.includes('\\') || value.includes('/') ? compactPath(value) : value
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatValue(item)).join(', ')
  }
  return Object.entries(value)
    .filter(([key]) => !OMITTED_KEYS.has(key))
    .map(([key, item]) => `${key}: ${formatValue(item)}`)
    .join(', ')
}

function detailsFromObject(data: DetailSource | undefined, preferredKeys?: string[]): DetailItem[] {
  if (!data) {
    return []
  }

  const entries = preferredKeys
    ? preferredKeys
        .filter((key) => data[key] !== undefined && !OMITTED_KEYS.has(key))
        .map((key) => [key, data[key]] as [string, JsonValue])
    : Object.entries(data).filter(([key]) => !OMITTED_KEYS.has(key))

  return entries
    .filter((entry): entry is [string, JsonValue] => entry[1] !== undefined && entry[1] !== null)
    .map(([key, value]) => ({
      label: key,
      value: formatValue(value),
    }))
    .filter((item) => item.value !== '')
}

function readMetrics(
  data: JsonObject,
  keys = ['metrics', 'final_metrics'],
): Record<string, number | string> {
  const metrics = keys
    .map((key) => data[key])
    .find((value) => isJsonObject(value))
  if (!isJsonObject(metrics)) {
    return {}
  }

  const entries = Object.entries(metrics).filter(
    (entry): entry is [string, number | string] =>
      typeof entry[1] === 'number' || typeof entry[1] === 'string',
  )
  return Object.fromEntries(entries)
}

function readNumberMatrix(value: JsonValue | undefined): number[][] | undefined {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    !value.every(
      (row) =>
        Array.isArray(row) &&
        row.length > 0 &&
        row.every((cell) => typeof cell === 'number'),
    )
  ) {
    return undefined
  }

  return value as number[][]
}

function normalizeLabel(label: string): string {
  if (label === '0') {
    return 'negative'
  }
  if (label === '1') {
    return 'neutral'
  }
  if (label === '2') {
    return 'positive'
  }
  return label
}

function readLabels(data: JsonObject, matrixSize: number): string[] {
  const labels = Array.isArray(data.labels) ? data.labels : []
  const normalizedLabels = labels
    .filter((label): label is string | number => typeof label === 'string' || typeof label === 'number')
    .map((label) => String(label))

  if (normalizedLabels.length === matrixSize) {
    return normalizedLabels.map(normalizeLabel)
  }

  return Array.from({ length: matrixSize }, (_item, index) => {
    if (index === 0) {
      return 'negative'
    }
    if (index === 1) {
      return 'neutral'
    }
    if (index === 2) {
      return 'positive'
    }
    return `class ${index}`
  })
}

function readSupport(
  source: JsonObject,
  labels: string[],
  matrix: number[][],
): number[] {
  const supportByLabel = isJsonObject(source.support_by_label)
    ? source.support_by_label
    : {}

  return labels.map((label, index) => {
    const labelValue = supportByLabel[label] ?? supportByLabel[String(index)]
    if (typeof labelValue === 'number') {
      return labelValue
    }
    return matrix[index]?.reduce((total, value) => total + value, 0) ?? 0
  })
}

function displayIndexesForMatrix(labels: string[], support: number[]): number[] {
  const indexes = labels
    .map((label, index) => ({ index, label: label.trim().toLowerCase() }))
    .filter((item) => item.label !== '-1' && item.label !== 'invalid')
    .map((item) => item.index)

  return indexes.length > 0 ? indexes : support
    .map((value, index) => ({ value, index }))
    .filter((item) => item.value > 0)
    .map((item) => item.index)
}

function evaluationDetailsFromMetrics(
  source: JsonObject | undefined,
  labelsSource: JsonObject | undefined,
): EvaluationDetails | undefined {
  if (!source) {
    return undefined
  }

  const confusionMatrix = readNumberMatrix(source.confusion_matrix)
  if (!confusionMatrix) {
    return undefined
  }

  const labels = readLabels(labelsSource ?? source, confusionMatrix.length)
  const support = readSupport(source, labels, confusionMatrix)
  const displayIndexes = displayIndexesForMatrix(labels, support)
  const displayLabels = displayIndexes.map((index) => labels[index])
  const displayMatrix = displayIndexes.map((rowIndex) =>
    displayIndexes.map((columnIndex) => confusionMatrix[rowIndex]?.[columnIndex] ?? 0),
  )
  const classScores = displayIndexes.map((index) => {
    const label = labels[index]
    const truePositive = confusionMatrix[index]?.[index] ?? 0
    const predictedTotal = confusionMatrix.reduce(
      (total, row) => total + (row[index] ?? 0),
      0,
    )
    const actualTotal = support[index] ?? 0
    const precision = predictedTotal > 0 ? truePositive / predictedTotal : 0
    const recall = actualTotal > 0 ? truePositive / actualTotal : 0
    const f1 =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0

    return {
      label,
      precision,
      recall,
      f1,
      support: actualTotal,
    }
  })

  return {
    labels: displayLabels,
    confusionMatrix: displayMatrix,
    classScores,
  }
}

function readEvaluationDetails(
  data: JsonObject,
  sidecarData: JsonObject | undefined,
  key: 'original' | 'without_noise',
): EvaluationDetails | undefined {
  if (key === 'without_noise') {
    const sidecarMetrics = sidecarData?.metrics_without_noise
    const metadataMetrics = data.metrics_without_noise
    return evaluationDetailsFromMetrics(
      isJsonObject(sidecarMetrics)
        ? sidecarMetrics
        : isJsonObject(metadataMetrics)
          ? metadataMetrics
          : undefined,
      sidecarData ?? data,
    )
  }

  const recomputedMetrics = sidecarData?.recomputed_original_metrics
  const metadataMetrics = data.metrics ?? data.final_metrics
  return evaluationDetailsFromMetrics(
    isJsonObject(recomputedMetrics)
      ? recomputedMetrics
      : isJsonObject(metadataMetrics)
        ? metadataMetrics
        : undefined,
    sidecarData ?? data,
  )
}

function inferFamily(parts: string[]): string {
  if (parts[0] === 'fine_tuned_bert') {
    return 'fine_tuned_models'
  }
  return parts[0] ?? 'models'
}

function isEncoderOrDecoderFamily(family: string): boolean {
  return family === 'encoder_models' || family === 'decoder_models'
}

function inferSection(parts: string[]): 'models' | 'embeddings' {
  return parts[0] === 'embeddings_runs' ? 'embeddings' : 'models'
}

function inferDisplayName(data: JsonObject, section: 'models' | 'embeddings'): string {
  if (section === 'embeddings') {
    return readString(data, 'name') ?? readString(data, 'type') ?? 'Embedding run'
  }

  const embeddingName = readString(data, 'embedding_name')
  const modelName = readString(data, 'model_name') ?? readString(data, 'task')
  if (embeddingName && modelName) {
    return `${embeddingName} / ${modelName}`
  }
  return modelName ?? 'Model run'
}

function inferRunId(parts: string[], data: JsonObject): string {
  return (
    readString(data, 'dl_run_id') ??
    readString(data, 'ml_run_id') ??
    readString(data, 'embedding_run_id') ??
    readString(data, 'configuration_hash') ??
    parts[1] ??
    'root'
  )
}

function shapeDetails(data: JsonObject): DetailItem[] {
  const details: DetailItem[] = []
  if (Array.isArray(data.train_shape)) {
    details.push({ label: 'train_shape', value: formatValue(data.train_shape) })
  }
  if (Array.isArray(data.test_shape)) {
    details.push({ label: 'test_shape', value: formatValue(data.test_shape) })
  }
  if (Array.isArray(data.classes) || Array.isArray(data.labels)) {
    details.push({
      label: 'labels',
      value: formatValue((data.classes ?? data.labels) as JsonValue),
    })
  }
  return details
}

function outputVectorDimension(data: JsonObject): number | undefined {
  const shape = Array.isArray(data.train_shape) ? data.train_shape : data.test_shape
  const dimension = Array.isArray(shape) ? shape[1] : undefined
  return typeof dimension === 'number' ? dimension : undefined
}

function embeddingConfiguration(data: JsonObject): DetailItem[] {
  const parameters = isJsonObject(data.parameters) ? data.parameters : {}
  const configured = isJsonObject(data.configured_parameters)
    ? data.configured_parameters
    : {}
  return [
    ...detailsFromObject(
      {
      name: data.name,
      type: data.type,
      output_vector_dimension: outputVectorDimension(data),
      vector_dimension: configured.vector_dimension ?? parameters.vector_dimension,
      batch_size: configured.batch_size ?? parameters.batch_size,
        max_length: configured.max_length ?? parameters.max_length,
        ngram_range: configured.ngram_range ?? parameters.ngram_range,
        max_features: configured.max_features ?? parameters.max_features,
        min_df: configured.min_df ?? parameters.min_df,
        max_df: configured.max_df ?? parameters.max_df,
        window: configured.window ?? parameters.window,
        min_count: configured.min_count ?? parameters.min_count,
        sg: configured.sg ?? parameters.sg,
        workers: configured.workers ?? parameters.workers,
        epochs: configured.epochs ?? parameters.epochs,
        model_name: parameters.model_name,
        kind: parameters.kind,
        device: parameters.device,
      },
    ),
  ]
}

function modelConfiguration(data: JsonObject): DetailItem[] {
  const parameters = isJsonObject(data.parameters) ? data.parameters : {}
  const configured = isJsonObject(data.configured_parameters)
    ? data.configured_parameters
    : {}
  return detailsFromObject(
    {
      model_name: data.model_name,
      model_type: data.model_type ?? data.task,
      parameter_source: data.parameter_source,
      ...configured,
      ...parameters,
    },
    [
      'model_name',
      'model_type',
      'type',
      'parameter_source',
      'solver',
      'C',
      'c',
      'kernel',
      'gamma',
      'max_iter',
      'n_estimators',
      'max_depth',
      'learning_rate',
      'weight_decay',
      'dropout',
      'hidden_dims',
      'hidden_layers',
      'activation_functions',
      'bidirectional',
      'num_hidden_states',
      'token_embedding_dim',
      'max_sequence_length',
      'max_vocab_size',
      'max_length',
      'batch_size',
      'warmup_ratio',
      'device',
    ],
  )
}

function modelRunConfiguration(data: JsonObject): DetailItem[] {
  const hyperparameterOptimization = isJsonObject(data.hyperparameter_optimization)
    ? data.hyperparameter_optimization
    : undefined
  const earlyStoppedEpoch = Array.isArray(data.history)
    ? data.history.find((epoch) => isJsonObject(epoch) && epoch.early_stopped === true)
    : undefined
  const earlyStopped = Array.isArray(data.history)
    ? data.history.some((epoch) => isJsonObject(epoch) && epoch.early_stopped === true)
    : false
  const parameters = isJsonObject(data.parameters) ? data.parameters : {}

  return [
    ...detailsFromObject({
      status: data.status,
      run_id: data.dl_run_id ?? data.ml_run_id,
      generated_at_utc: data.generated_at_utc,
      epochs: data.epochs ?? parameters.epochs,
      early_stopping: earlyStopped,
      early_stopped_at_epoch: isJsonObject(earlyStoppedEpoch)
        ? earlyStoppedEpoch.epoch
        : undefined,
      early_stopping_reason: isJsonObject(earlyStoppedEpoch)
        ? earlyStoppedEpoch.early_stopping_reason
        : undefined,
      early_stopping_patience: parameters.early_stopping_patience,
      hyperparameter_optimization: hyperparameterOptimization?.enabled,
      hyperparameter_trials: hyperparameterOptimization?.n_trials,
      hyperparameter_metric: hyperparameterOptimization?.metric,
      embedding_mode: data.embedding_mode,
      embedding_name: data.embedding_name ?? 'No embedding used',
      embedding_type: data.embedding_type,
      embedding_run_id: data.embedding_run_id,
      dataset: baseName(readString(data, 'dataset_path')),
      test_size: data.test_size,
      random_state: data.random_state,
      stratify: data.use_stratify,
    }),
    ...shapeDetails(data),
  ]
}

function embeddingRunConfiguration(data: JsonObject): DetailItem[] {
  const splitConfig = isJsonObject(data.configuration_hash_payload)
    && isJsonObject(data.configuration_hash_payload.split_config)
    ? data.configuration_hash_payload.split_config
    : {}

  return [
    ...detailsFromObject({
      status: data.status,
      run_hash: data.run_hash ?? data.configuration_hash,
      dataset: datasetNameFromMetadata(data),
      output_vector_dimension: outputVectorDimension(data),
      text_column: splitConfig.text_column,
      label_column: splitConfig.label_column,
      test_size: splitConfig.test_size,
      random_state: splitConfig.random_state,
      stratify: splitConfig.use_stratify,
    }),
    ...shapeDetails(data),
  ]
}

function fineTunedRunConfiguration(data: JsonObject): DetailItem[] {
  return [
    ...detailsFromObject({
      generated_at_utc: data.generated_at_utc,
      dataset: baseName(readString(data, 'dataset_path')),
      text_column: data.text_column,
      label_column: data.label_column,
      test_size: data.test_size,
      random_state: data.random_state,
      use_stratify: data.use_stratify,
      epochs: data.epochs,
      early_stopping: false,
      hyperparameter_optimization: false,
      embedding_name: 'No embedding used',
      final_validation_loss: data.final_validation_loss,
    }),
    ...shapeDetails(data),
  ]
}

function transformerModelConfiguration(data: JsonObject): DetailItem[] {
  return detailsFromObject(data, [
    'model_name',
    'model_slug',
    'task',
    'approach',
    'tuning_mode',
    'classification_tuning_mode',
    'fine_tune_model',
    'use_chat_template',
    'trust_remote_code',
    'max_length',
    'batch_size',
    'epochs',
    'learning_rate',
    'use_classifier_learning_rate',
    'classifier_learning_rate',
    'weight_decay',
    'warmup_ratio',
    'gradient_clip_norm',
    'generation_max_new_tokens',
    'generation_do_sample',
    'generation_temperature',
    'use_peft',
    'peft_r',
    'peft_alpha',
    'peft_dropout',
    'peft_target_modules',
    'device',
    'cuda_device_name',
  ])
}

function transformerRunConfiguration(data: JsonObject): DetailItem[] {
  return [
    ...detailsFromObject({
      generated_at_utc: data.generated_at_utc,
      dataset: baseName(readString(data, 'dataset_path')),
      text_column: data.text_column,
      label_column: data.label_column,
      test_size: data.test_size,
      random_state: data.random_state,
      use_stratify: data.use_stratify,
      epochs: data.epochs,
      embedding_name: 'No embedding used',
      final_validation_loss: data.final_validation_loss,
    }),
    ...shapeDetails(data),
  ]
}

async function readOptionalJson(filePath: string): Promise<JsonObject | undefined> {
  try {
    const rawJson = await fs.readFile(filePath, 'utf-8')
    return JSON.parse(rawJson) as JsonObject
  } catch {
    return undefined
  }
}

async function findEmbeddingMetadata(data: JsonObject): Promise<JsonObject | undefined> {
  const embeddingName = readString(data, 'embedding_name')
  if (!embeddingName) {
    return undefined
  }

  const runIds = [
    readString(data, 'embedding_run_id'),
    readString(data, 'ml_run_id'),
    readString(data, 'dl_run_id'),
  ].filter((value): value is string => Boolean(value))

  for (const runId of runIds) {
    const metadata = await readOptionalJson(
      path.join(modelsRoot, 'embeddings_runs', runId, embeddingName, 'metadata.json'),
    )
    if (metadata) {
      return metadata
    }
  }

  try {
    const embeddingRootEntries = await fs.readdir(
      path.join(modelsRoot, 'embeddings_runs'),
      { withFileTypes: true },
    )
    for (const entry of embeddingRootEntries) {
      if (!entry.isDirectory()) {
        continue
      }
      const metadata = await readOptionalJson(
        path.join(modelsRoot, 'embeddings_runs', entry.name, embeddingName, 'metadata.json'),
      )
      if (metadata) {
        return metadata
      }
    }
  } catch {
    return undefined
  }

  return undefined
}

function datasetNameFromMetadata(data: JsonObject, sidecarData?: JsonObject): string | undefined {
  const payload = isJsonObject(data.configuration_hash_payload)
    ? data.configuration_hash_payload
    : undefined
  const datasetPayload = payload && isJsonObject(payload.dataset)
    ? payload.dataset
    : undefined
  const datasetPath =
    readString(data, 'dataset_path') ??
    readString(sidecarData ?? {}, 'dataset_path') ??
    (datasetPayload ? readString(datasetPayload, 'dataset_path') : undefined)

  return baseName(datasetPath)
}

function trainedOnNoisyData(datasetName: string | undefined): boolean | undefined {
  if (!datasetName) {
    return undefined
  }
  return !datasetName.toLowerCase().includes('noise_removed')
}

async function listMetadataFiles(directory: string): Promise<string[]> {
  const entries = await fs.readdir(directory, { withFileTypes: true })
  const nestedFiles = await Promise.all(
    entries.map(async (entry) => {
      const entryPath = path.join(directory, entry.name)
      if (entry.isDirectory()) {
        return listMetadataFiles(entryPath)
      }
      return entry.isFile() && entry.name === 'metadata.json' ? [entryPath] : []
    }),
  )

  return nestedFiles.flat()
}

function sendJson(
  response: { setHeader: (key: string, value: string) => void; end: (body?: string) => void },
  payload: unknown,
) {
  response.setHeader('Content-Type', 'application/json')
  response.end(JSON.stringify(payload))
}

async function loadRunSummaries() {
  try {
    await fs.access(modelsRoot)
  } catch {
    return { modelsRoot: 'models', runs: [] }
  }

  const metadataFiles = await listMetadataFiles(modelsRoot)
  const runs = await Promise.all(
    metadataFiles.map(async (filePath) => {
      const rawJson = await fs.readFile(filePath, 'utf-8')
      const data = JSON.parse(rawJson) as JsonObject
      const metricsWithoutNoiseData = await readOptionalJson(
        path.join(path.dirname(filePath), 'metrics_without_noise.json'),
      )
      const embeddingMetadata = await findEmbeddingMetadata(data)
      const relativePath = path.relative(modelsRoot, filePath).replaceAll(path.sep, '/')
      const parts = relativePath.split('/')
      const section = inferSection(parts)
      const family = inferFamily(parts)
      const datasetName =
        datasetNameFromMetadata(data, metricsWithoutNoiseData) ??
        (embeddingMetadata ? datasetNameFromMetadata(embeddingMetadata) : undefined)

      return {
        relativePath,
        section,
        family,
        runId: inferRunId(parts, data),
        displayName: inferDisplayName(data, section),
        generatedAt: readString(data, 'generated_at_utc'),
        status: readString(data, 'status'),
        metrics: readMetrics(data),
        metricsWithoutNoise: metricsWithoutNoiseData
          ? readMetrics(metricsWithoutNoiseData, ['metrics_without_noise'])
          : {},
        evaluationDetails: readEvaluationDetails(
          data,
          metricsWithoutNoiseData,
          'original',
        ),
        evaluationDetailsWithoutNoise: readEvaluationDetails(
          data,
          metricsWithoutNoiseData,
          'without_noise',
        ),
        datasetName,
        trainedOnNoisyData: trainedOnNoisyData(datasetName),
        modelConfiguration:
          isEncoderOrDecoderFamily(family) || family === 'fine_tuned_models'
            ? transformerModelConfiguration(data)
            : section === 'models'
              ? modelConfiguration(data)
              : [],
        runConfiguration:
          isEncoderOrDecoderFamily(family)
            ? transformerRunConfiguration(data)
            : family === 'fine_tuned_models'
              ? fineTunedRunConfiguration(data)
              : section === 'models'
                ? modelRunConfiguration({
                    ...data,
                    dataset_path: datasetName ?? data.dataset_path,
                  })
                : embeddingRunConfiguration(data),
        embeddingConfiguration:
          section === 'embeddings'
            ? embeddingConfiguration(data)
            : detailsFromObject({
                embedding_name: data.embedding_name ?? 'No embedding used',
                embedding_type: data.embedding_type,
                embedding_mode: data.embedding_mode,
                embedding_run_id: data.embedding_run_id,
              }),
      }
    }),
  )

  return { modelsRoot: 'models', runs }
}

function sendError(
  response: { statusCode: number; end: (body?: string) => void },
  error: unknown,
) {
  response.statusCode = 500
  response.end(
    JSON.stringify({
      error: error instanceof Error ? error.message : 'Unknown error',
    }),
  )
}

function readRequestBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []

    request.on('data', (chunk: Buffer) => {
      chunks.push(chunk)
    })
    request.on('end', () => {
      resolve(Buffer.concat(chunks).toString('utf-8'))
    })
    request.on('error', reject)
  })
}

function runSummaryApiPlugin(): Plugin {
  return {
    name: 'run-summary-api',
    async generateBundle() {
      const payload = await loadRunSummaries()
      this.emitFile({
        type: 'asset',
        fileName: 'run-summaries.json',
        source: JSON.stringify(payload),
      })
    },
    configureServer(server) {
      server.middlewares.use('/api/run-summaries', async (_request, response) => {
        try {
          const payload = await loadRunSummaries()
          sendJson(response, payload)
        } catch (error) {
          sendError(response, error)
        }
      })
    },
    configurePreviewServer(server) {
      server.middlewares.use('/api/run-summaries', async (_request, response) => {
        try {
          const payload = await loadRunSummaries()
          sendJson(response, payload)
        } catch (error) {
          sendError(response, error)
        }
      })
    },
  }
}

function textClassificationProxyPlugin(textClassificationUrl: string): Plugin {
  async function handleProxy(request: IncomingMessage, response: ServerResponse) {
    if (request.method !== 'POST') {
      response.statusCode = 405
      sendJson(response, { error: 'Method not allowed' })
      return
    }

    if (!textClassificationUrl) {
      response.statusCode = 500
      sendJson(response, { error: 'TEXT_CLASSIFICATION_URL is not configured.' })
      return
    }

    try {
      const body = await readRequestBody(request)
      const upstreamResponse = await fetch(textClassificationUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body,
      })
      const payload = await upstreamResponse.text()

      response.statusCode = upstreamResponse.status
      response.setHeader(
        'Content-Type',
        upstreamResponse.headers.get('content-type') ?? 'application/json',
      )
      response.end(payload)
    } catch (error) {
      sendError(response, error)
    }
  }

  return {
    name: 'text-classification-proxy',
    configureServer(server) {
      server.middlewares.use('/api/text-classification/predict', handleProxy)
    },
    configurePreviewServer(server) {
      server.middlewares.use('/api/text-classification/predict', handleProxy)
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = {
    ...loadEnv(mode, envRoot, ''),
    ...process.env,
  }
  const textClassificationUrl =
    env.TEXT_CLASSIFICATION_URL ??
    env.VITE_TEXT_CLASSIFICATION_URL ??
    env['TEXT-CLASSIFICATION-URL'] ??
    ''

  return {
    base: inferBasePath(env),
    envDir: envRoot,
    define: {
      __TEXT_CLASSIFICATION_URL__: JSON.stringify(textClassificationUrl),
    },
    plugins: [
      react(),
      runSummaryApiPlugin(),
      textClassificationProxyPlugin(textClassificationUrl),
    ],
  }
})
