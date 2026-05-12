### Full Description / README

# HF Decoder Text Classifier API

A generic Dockerized FastAPI inference server for Hugging Face decoder-based text classification models.

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

## API Endpoint

### POST `/predict`

Request:

```json
{
  "text": "I love this product"
}
````

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


