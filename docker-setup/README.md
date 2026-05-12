### Full Description / README

# HF Decoder Text Classifier API

A generic Dockerized FastAPI inference server for Hugging Face decoder-based text classification models.

Docker Hub image:

```text
nezar1/hf-decoder-text-classifier-api:latest
```

Docker Hub page:

```text
https://hub.docker.com/repository/docker/nezar1/hf-decoder-text-classifier-api/general
```

## Features

- Hugging Face decoder model support
- GPU acceleration with CUDA PyTorch
- Configurable prompt templates
- Optional chat-template inference
- Configurable labels
- FastAPI REST API
- Docker deployment ready
- Lightning.ai compatible
- CORS enabled

## Run From Docker Hub

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_ID=Nezar1/Qwen3-4B-Instruct-2507-sentiment-classifier \
  -e HF_TOKEN=<HF_TOKEN> \
  -e LABELS=negative,neutral,positive \
  nezar1/hf-decoder-text-classifier-api:latest
```

The API is then available at:

```text
http://localhost:8000/predict
```

## API Endpoint

### POST `/predict`

Request:

```json
{
  "text": "I love this product"
}
```

Response:

```json
{
  "text": "I love this product",
  "label": "positive",
  "raw_output": "positive"
}
```

## Environment Variables

| Variable          | Description            |
| ----------------- | ---------------------- |
| MODEL_ID          | Hugging Face model ID  |
| HF_TOKEN          | Hugging Face token     |
| PROMPT_TEMPLATE   | Prompt template        |
| USE_CHAT_TEMPLATE | true/false             |
| SYSTEM_PROMPT     | Optional system prompt |
| LABELS            | Comma-separated labels |
| MAX_NEW_TOKENS    | Generation length      |

## Example

```txt
MODEL_ID=Nezar1/Qwen3-4B-Instruct-2507-sentiment-classifier
PROMPT_TEMPLATE=Classify the sentiment. Labels: negative, neutral, positive\nReview: {text}\nOutput:
USE_CHAT_TEMPLATE=false
LABELS=negative,neutral,positive
```

## Technologies

* FastAPI
* Transformers
* PyTorch
* CUDA
* Docker
* Lightning.ai


