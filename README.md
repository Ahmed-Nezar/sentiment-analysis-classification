# Sentiment Analysis Classification

SentimentFlow is an end-to-end sentiment classification project for comparing classical machine learning, deep learning, encoder fine-tuning, and decoder/instruction-model fine-tuning approaches on a three-class sentiment task: negative, neutral, and positive.

The repository contains the reproducible project code, configuration files, model metadata, evaluation summaries, and a React dashboard/inference UI. Large or transient artifacts such as raw datasets, notebooks, Kaggle export archives, embedding tensors, fitted model binaries, and full model weights are intentionally not committed because they are not needed to understand or reproduce the full flow from the public code and configuration.

## Best Published Models

The strongest final models are published on Hugging Face:

- BGE-M3 sentiment classifier: https://huggingface.co/Nezar1/bge-m3-sentiment-classifier
- Qwen3-4B Instruct sentiment classifier: https://huggingface.co/Nezar1/Qwen3-4B-Instruct-2507-sentiment-classifier

These models were selected after running multiple experiment families and comparing the original metrics with the metrics after removing noisy or ambiguous evaluation rows.

## Project Flow

The project is organized as a repeatable pipeline:

1. Dataset acquisition and exploration
2. Data cleaning and noise investigation
3. Embedding generation
4. Classical machine learning trials
5. Deep learning trials
6. Transformer encoder fine-tuning
7. Decoder/instruction-model fine-tuning
8. Noise-aware evaluation
9. Best model publication to Hugging Face
10. Frontend deployment and inference wiring

The starting dataset is:

- https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset

The GloVe embedding source used for the GloVe experiments is:

- https://nlp.stanford.edu/data/glove.6B.zip

## Repository Structure

```text
.
|-- sentimentFlow/              # Python package for preprocessing, embeddings, training, and evaluation
|   `-- src/
|       |-- extraction/         # Dataset preprocessing and noisy-row filtering
|       |-- embedding/          # BoW, TF-IDF, Word2Vec, GloVe, FastText, transformer embeddings
|       |-- train/              # Classical ML, DL, encoder, and decoder training pipelines
|       |-- evaluation/         # Noise-aware metrics generation
|       |-- usage/              # Script-style entry points
|       `-- utils/              # Configuration loading helpers
|-- sentimentApp/               # React + Vite dashboard and inference UI
|-- models/                     # Lightweight metadata and evaluation summaries kept for the dashboard
|-- config.yaml                 # Shared dataset path configuration
|-- embedding_config.yaml       # Embedding run configuration
|-- ml_config.yaml              # Classical ML configuration
|-- dl_config.yaml              # Deep learning configuration
|-- encoder_config.yaml         # Encoder fine-tuning configuration
|-- decoder_config.yaml         # Decoder/instruction tuning configuration
`-- .github/workflows/          # GitHub Pages deployment workflow
```

## What Is Not Pushed

Some files are intentionally excluded from version control:

- `datasets/`: raw, cleaned, and noise-removed CSV files.
- `notebooks/` and `*.ipynb`: exploratory notebooks and training notebooks used during experimentation.
- `kaggle/`: Kaggle output archives, model zip files, prediction bundles, and exported run artifacts.
- `embeddings/`: downloaded GloVe files and generated embedding assets.
- Heavy model artifacts such as `*.pt`, `*.joblib`, `*.pkl`, `*.tensor`, `*.safetensors`, `*.model`, and `*.npy`.
- Local environment files such as `.env`, `.venv/`, Jupyter runtime data, logs, and frontend build output.

The committed `models/` content is mainly metadata and evaluation JSON used by the dashboard. The actual model tensors and large intermediate arrays are not required for the repository-level flow and would make the repository unnecessarily large.

## Data Cleaning and Noise Removal

The project first normalizes and cleans the dataset into a text/label format. A second pass identifies rows that are noisy, ambiguous, mislabeled, or not useful for fair evaluation. The criteria are documented in:

- `Noise_Removal_Criteria.md`

Noise-aware evaluation is important in this project because the original public dataset contains examples that can make model comparison misleading. For this reason, many summaries include both normal metrics and `metrics_without_noise.json`.

Main preprocessing commands:

```bash
python sentimentFlow/src/usage/usage.py
python sentimentFlow/src/usage/usage.py --no-noise
python sentimentFlow/src/usage/usage.py --no-noise --clean
```

## Embedding Trials

Embedding runs are controlled by `embedding_config.yaml`. The project tested sparse, static, and transformer-based representations:

- Bag of Words: unigram, bigram, unigram + bigram
- TF-IDF: unigram, bigram, unigram + bigram
- Word2Vec: skip-gram and CBOW
- FastText: skip-gram and CBOW
- GloVe 300d
- BERT base and BERT large embeddings
- BGE small/base/M3
- GTE small/base/large
- MiniLM L6/L12
- Qwen3 embedding models

Run embeddings with:

```bash
python sentimentFlow/src/usage/embedding_usage.py
```

The embedding pipeline stores run metadata under `models/embeddings_runs/`. Large generated embedding arrays are ignored by Git.

## Classical Machine Learning Trials

Classical ML experiments are configured in `ml_config.yaml`. The pipeline combines selected embedding runs with:

- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost
- Decision Tree

The configuration also supports Optuna hyperparameter optimization through:

```yaml
hyperparameter_optimization: [false, 10]
```

The first value enables or disables Optuna. The second value is the number of trials.

Run classical ML experiments with:

```bash
python sentimentFlow/src/usage/ml_usage.py
```

The best classical ML runs were competitive, especially when SVM or logistic regression were paired with strong sentence embeddings such as BGE-M3.

## Deep Learning Trials

Deep learning experiments are configured in `dl_config.yaml`. The project tested:

- Feed-forward neural networks over predefined embeddings
- RNN
- LSTM
- GRU
- HMM-style sequence classifiers
- Network-learned token embeddings
- Predefined transformer embedding inputs

The deep learning configs include early stopping, dropout, weight decay, batch size, sequence length, and different hidden-layer configurations. CUDA can be required for these runs through:

```yaml
require_cuda: true
```

Run deep learning experiments with:

```bash
python sentimentFlow/src/usage/dl_usage.py
```

## Encoder Fine-Tuning

Encoder fine-tuning is configured in `encoder_config.yaml`. These experiments fine-tune encoder-style transformer models directly for classification. The pipeline supports:

- Stratified train/test splits
- Mixed precision
- Optional data parallelism
- Optional PEFT/LoRA settings
- Separate classifier learning rate
- Warmup and gradient clipping

Run encoder fine-tuning with:

```bash
python sentimentFlow/src/usage/encoder_usage.py
```

The BGE-M3 fine-tuned model became one of the final published models:

- https://huggingface.co/Nezar1/bge-m3-sentiment-classifier

## Decoder and Instruction-Model Fine-Tuning

Decoder and instruction-model runs are configured in `decoder_config.yaml`. The project tested decoder models in two broad modes:

- Classification-head tuning
- Instruction-style sentiment prediction

The decoder pipeline supports full decoder tuning, instruction prompts, chat templates, generation settings, classifier learning rates, mixed precision, and optional PEFT/LoRA.

Run decoder fine-tuning with:

```bash
python sentimentFlow/src/usage/decoder_usage.py
```

The Qwen3-4B Instruct model produced the strongest final decoder result and was published here:

- https://huggingface.co/Nezar1/Qwen3-4B-Instruct-2507-sentiment-classifier

## Kaggle Usage

Kaggle was used for GPU-backed experimentation, especially for larger fine-tuning runs that were too heavy or slow for a local environment. The notebooks under `notebooks/` were used as working notebooks for:

- BERT and encoder fine-tuning
- Decoder and instruction-model experiments
- Metrics and prediction export
- Hugging Face publishing checks
- Inference smoke tests

Kaggle-generated archives are stored locally under `kaggle/` during development, but they are ignored in Git. These archives can be several hundred megabytes or multiple gigabytes and contain generated model artifacts, predictions, or zipped training outputs. They are useful as run outputs, not as source files.

## Lightning.ai Usage

Lightning.ai was used as a training and deployment environment for GPU-backed model work and service hosting. In the project flow, Lightning.ai served two purposes:

- Training and validating selected model candidates in a managed GPU environment.
- Hosting or exposing the best inference service so the deployed frontend could call it through a stable HTTP endpoint.

The frontend does not require the training notebooks or model tensors to be present. It only needs the deployed inference URL and the static run-summary data generated from metadata.

## Cloudflare Worker Proxy

The deployed frontend uses a Cloudflare Worker proxy in front of the model inference service. This keeps the public React app from depending directly on a raw training or hosting endpoint and gives the project a stable prediction URL.

The expected request shape is:

```json
{
  "text": "this is a good movie!"
}
```

The expected response shape is:

```json
{
  "predicted_class_id": 2,
  "probability": [0.01, 0.04, 0.95]
}
```

The frontend maps class IDs as:

```text
0 -> negative
1 -> neutral
2 -> positive
```

For local Vite development, `sentimentApp/vite.config.ts` also provides a dev/preview proxy route at:

```text
/api/text-classification/predict
```

For the production GitHub Pages build, the app uses `TEXT_CLASSIFICATION_URL`, which should point to the public Cloudflare Worker URL or another compatible inference endpoint.

## Frontend Dashboard and Inference App

The `sentimentApp/` app has three main views:

- Landing page: entry point for the dashboard and inference workspace.
- Run dashboard: compares embedding, ML, DL, encoder, and decoder run metadata.
- Inference workspace: sends text to the configured sentiment classifier endpoint.

During development, Vite reads metadata from the local `models/` folder through a dev server route. During production build, the Vite plugin emits a static `run-summaries.json` asset, so GitHub Pages can serve the dashboard without a backend.

Run the frontend locally:

```bash
cd sentimentApp
npm install
npm run dev
```

Build the frontend:

```bash
cd sentimentApp
npm run build
```

## Deployment

The React app is deployed to GitHub Pages through:

```text
.github/workflows/deploy-pages.yml
```

The workflow:

1. Runs on pushes to `main` or `master`, or manually through `workflow_dispatch`.
2. Checks out the repository.
3. Installs Node.js 22.
4. Installs frontend dependencies in `sentimentApp/`.
5. Builds the Vite app.
6. Uploads `sentimentApp/dist` as a GitHub Pages artifact.
7. Deploys the artifact to GitHub Pages.

Deployment environment variables:

```text
TEXT_CLASSIFICATION_URL
VITE_BASE_PATH
```

`TEXT_CLASSIFICATION_URL` should point to the Cloudflare Worker proxy or another endpoint compatible with the inference response format. `VITE_BASE_PATH` can be used when GitHub Pages needs a repository-specific base path.

## Environment Variables

Create `.env` from `.env.example` when running locally:

```text
HF_TOKEN="<HF_TOKEN>"
HF_HOME="<HF_HOME_DIR>"
TEXT_CLASSIFICATION_URL="<BGE_M3_OR_WORKER_URL>"
```

`HF_TOKEN` is used for Hugging Face access and publishing workflows. `HF_HOME` can be used to control model cache location. `TEXT_CLASSIFICATION_URL` is used by the frontend and local proxy path for inference.

## Python Setup

This project targets Python 3.11 or newer. Dependencies are declared in `pyproject.toml`.

Using `uv`:

```bash
uv sync
```

Using pip:

```bash
python -m venv .venv
pip install -e .[dev,notebook]
```

The project uses CUDA-enabled PyTorch settings in `pyproject.toml` through the PyTorch CUDA 12.6 index for GPU runs.

## Running the Full Pipeline

A typical local or remote workflow is:

```bash
python sentimentFlow/src/usage/usage.py
python sentimentFlow/src/usage/usage.py --no-noise --clean
python sentimentFlow/src/usage/embedding_usage.py
python sentimentFlow/src/usage/ml_usage.py
python sentimentFlow/src/usage/dl_usage.py
python sentimentFlow/src/usage/encoder_usage.py
python sentimentFlow/src/usage/decoder_usage.py
```

For heavier runs, the same pipeline can be transferred into Kaggle or Lightning.ai notebooks and executed with GPU acceleration. Final model artifacts are published to Hugging Face instead of being committed to this repository.

## Evaluation

Evaluation artifacts are stored as JSON summaries under `models/`. The dashboard reads these summaries to compare:

- Accuracy
- Macro F1
- Original metrics
- Metrics after noisy-row removal
- Dataset kind
- Run configuration
- Embedding configuration
- Model family
- Generated run metadata

The strongest observed model families were the fine-tuned transformer models, especially BGE-M3 and Qwen3-4B Instruct. Classical and deep learning models remain in the repository because they document the comparison path and justify the final model choices.

## Notes for Future Work

- Wire the second published Qwen3-4B inference endpoint into the frontend once the deployment endpoint is finalized.
- Keep heavy tensors and generated datasets out of Git; publish final deployable artifacts to Hugging Face instead.
- Regenerate `run-summaries.json` through the Vite build whenever model metadata changes.
- Keep Cloudflare Worker responses aligned with the frontend inference contract.
