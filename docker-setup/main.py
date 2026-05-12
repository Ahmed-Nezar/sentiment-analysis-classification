import os
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

PROMPT_TEMPLATE = os.getenv(
    "PROMPT_TEMPLATE",
    "Classify the sentiment. Labels: negative, neutral, positive\nReview: {text}\nOutput:"
)

USE_CHAT_TEMPLATE = os.getenv("USE_CHAT_TEMPLATE", "true").lower() == "true"

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a text classification model. Return only the label."
)

LABELS = [
    label.strip().lower()
    for label in os.getenv("LABELS", "negative,neutral,positive").split(",")
]

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "3"))

if not MODEL_ID:
    raise RuntimeError("MODEL_ID environment variable is required")

app = FastAPI(title="HF Decoder Text Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.eval()

@app.get("/")
def root():
    return {
        "status": "running",
        "model": MODEL_ID,
        "use_chat_template": USE_CHAT_TEMPLATE,
        "labels": LABELS,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

def build_prompt(text: str):
    user_prompt = PROMPT_TEMPLATE.format(text=text)

    if USE_CHAT_TEMPLATE:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return user_prompt

def extract_label(raw_output: str):
    cleaned = raw_output.strip().lower()

    for label in LABELS:
        if cleaned.startswith(label):
            return label

    for label in LABELS:
        if label in cleaned:
            return label

    return "UNKNOWN"

@app.post("/predict")
def predict(req: PredictRequest):
    prompt = build_prompt(req.text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    label = extract_label(raw_output)

    return {
        "text": req.text,
        "label": label,
        "raw_output": raw_output,
    }