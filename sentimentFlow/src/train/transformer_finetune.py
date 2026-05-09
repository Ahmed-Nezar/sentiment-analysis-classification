from __future__ import annotations

from contextlib import nullcontext
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ..utils import load_config
from .classical.utils import compute_metrics, json_compatible, resolve_path, to_bool, to_int


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return default


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return to_int(value, 0)


def _to_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    if not stripped:
        return None
    return [item.strip() for item in stripped.strip("[]").split(",") if item.strip()]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _seed_everything(seed: int, device: torch.device) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _resolve_device(require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA is required. Set require_cuda=false to allow CPU fallback.")
    return torch.device("cpu")


def _import_peft() -> tuple[Any, Any, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT/LoRA was requested, but 'peft' is not installed. "
            "Install peft or set use_peft=false."
        ) from exc
    return LoraConfig, TaskType, get_peft_model


def _get_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def _load_split(
    config: dict[str, Any],
    *,
    prefix: str,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    dataset_path = resolve_path(
        config.get(f"{prefix}_DATASET_PATH") or config.get("CLEANED_DATASET_PATH") or config.get("DATASET_PATH"),
        "datasets/cleaned_dataset.csv",
    )
    text_column = str(config.get(f"{prefix}_TEXT_COLUMN", "text"))
    label_column = str(config.get(f"{prefix}_LABEL_COLUMN", "label"))
    test_size = _to_float(config.get(f"{prefix}_TEST_SIZE"), 0.2)
    random_state = to_int(config.get(f"{prefix}_RANDOM_STATE"), 42)
    use_stratify = to_bool(config.get(f"{prefix}_USE_STRATIFY"), True)
    max_train_samples = _to_optional_int(config.get(f"{prefix}_MAX_TRAIN_SAMPLES"))
    max_test_samples = _to_optional_int(config.get(f"{prefix}_MAX_TEST_SAMPLES"))

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_df = pd.read_csv(dataset_path)
    if text_column not in dataset_df.columns:
        raise KeyError(f"Missing text column '{text_column}' in dataset")
    if label_column not in dataset_df.columns:
        raise KeyError(f"Missing label column '{label_column}' in dataset")

    texts = dataset_df[text_column].astype(str)
    labels = dataset_df[label_column]
    stratify_values = labels if use_stratify else None
    try:
        split = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_values,
        )
    except ValueError:
        split = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    x_train, x_test, y_train_raw, y_test_raw = split
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train_raw = y_train_raw.reset_index(drop=True)
    y_test_raw = y_test_raw.reset_index(drop=True)

    if max_train_samples is not None and max_train_samples > 0:
        x_train = x_train.iloc[:max_train_samples].reset_index(drop=True)
        y_train_raw = y_train_raw.iloc[:max_train_samples].reset_index(drop=True)
    if max_test_samples is not None and max_test_samples > 0:
        x_test = x_test.iloc[:max_test_samples].reset_index(drop=True)
        y_test_raw = y_test_raw.iloc[:max_test_samples].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    label_names = [str(label) for label in label_encoder.classes_]
    return x_train, x_test, y_train, y_test, label_encoder, label_names


class EncoderTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: Any,
        max_length: int,
        *,
        use_zero_token_type_ids: bool,
    ) -> None:
        self.texts = [str(text) for text in texts]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_zero_token_type_ids = use_zero_token_type_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }
        if self.use_zero_token_type_ids:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])
        elif "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        return item


class DecoderClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer: Any, max_length: int) -> None:
        self.texts = [str(text) for text in texts]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class DecoderInstructionDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: Any,
        max_length: int,
        *,
        label_names: list[str],
        use_chat_template: bool,
    ) -> None:
        self.texts = [str(text) for text in texts]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = label_names
        self.use_chat_template = use_chat_template

    def __len__(self) -> int:
        return len(self.labels)

    def _prompt(self, text: str) -> str:
        options = ", ".join(self.label_names)
        content = f"Classify the sentiment. Labels: {options}\nReview: {text}\nOutput:"
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"Input: Classify the sentiment.\nReview: {text}\nOutput:"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        label_text = self.label_names[int(self.labels[index])]
        full_text = f"{self._prompt(self.texts[index])} {label_text}"
        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[encoded["attention_mask"].squeeze(0) == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "class_labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class ManualEncoderClassifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, num_labels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.num_labels = num_labels
        hidden_size = int(getattr(self.config, "hidden_size"))
        dropout_prob = float(
            getattr(self.config, "classifier_dropout", None)
            or getattr(self.config, "hidden_dropout_prob", 0.1)
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        try:
            outputs = self.encoder(**kwargs)
        except TypeError:
            kwargs.pop("token_type_ids", None)
            outputs = self.encoder(**kwargs)
        last_hidden_state = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        pooled_output = last_hidden_state[:, 0].to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(pooled_output))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)


class DecoderHiddenStateClassifier(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module, num_labels: int, dropout_prob: float) -> None:
        super().__init__()
        self.decoder = decoder
        self.config = decoder.config
        self.num_labels = num_labels
        hidden_size = int(getattr(self.config, "hidden_size", None) or getattr(self.config, "n_embd"))
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        last_hidden_state = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if attention_mask is None:
            pooled = last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            pooled = last_hidden_state[torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), sequence_lengths]
        logits = self.classifier(self.dropout(pooled.to(dtype=self.classifier.weight.dtype)))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)


def _infer_lora_target_modules(model: torch.nn.Module, *, decoder: bool) -> list[str]:
    candidate_suffixes = (
        ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value", "dense")
        if decoder
        else ("query", "key", "value", "query_proj", "key_proj", "value_proj", "q_proj", "k_proj", "v_proj", "o_proj")
    )
    found = set()
    for module_name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        if not isinstance(module, torch.nn.Linear) and module_class_name != "Conv1D":
            continue
        short_name = module_name.rsplit(".", 1)[-1]
        if short_name in candidate_suffixes:
            found.add(short_name)
    if not found:
        raise ValueError("Could not infer LoRA target modules. Set peft_target_modules manually.")
    return sorted(found)


def _freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _build_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    classifier_learning_rate: float | None,
) -> torch.optim.Optimizer:
    model_for_params = model.module if isinstance(model, torch.nn.DataParallel) else model
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln_", "norm.weight")
    classifier_params = []
    decay_params = []
    no_decay_params = []
    for name, parameter in model_for_params.named_parameters():
        if not parameter.requires_grad:
            continue
        if classifier_learning_rate is not None and name.startswith("classifier"):
            classifier_params.append(parameter)
        elif any(token in name for token in no_decay):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    param_groups = [
        {"params": decay_params, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": learning_rate, "weight_decay": 0.0},
    ]
    if classifier_params:
        param_groups.append(
            {"params": classifier_params, "lr": classifier_learning_rate, "weight_decay": weight_decay}
        )
    return torch.optim.AdamW([group for group in param_groups if group["params"]])


def _evaluate_classifier(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.mean()
            logits = outputs.logits
            total_loss += float(loss.detach().cpu()) * batch["labels"].size(0)
            total_examples += batch["labels"].size(0)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(batch["labels"].cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return total_loss / max(1, total_examples), y_true, y_pred, compute_metrics(y_true, y_pred)


def _train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict[str, Any],
    *,
    prefix: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, float], float]:
    epochs = to_int(config.get(f"{prefix}_EPOCHS"), 1)
    learning_rate = _to_float(config.get(f"{prefix}_LEARNING_RATE"), 1e-5)
    weight_decay = _to_float(config.get(f"{prefix}_WEIGHT_DECAY"), 0.01)
    warmup_ratio = _to_float(config.get(f"{prefix}_WARMUP_RATIO"), 0.1)
    gradient_accumulation_steps = max(1, to_int(config.get(f"{prefix}_GRADIENT_ACCUMULATION_STEPS"), 1))
    use_mixed_precision = to_bool(config.get(f"{prefix}_USE_MIXED_PRECISION"), True)
    mixed_precision_dtype = str(config.get(f"{prefix}_MIXED_PRECISION_DTYPE", "float16"))
    gradient_clip_norm = _to_float(config.get(f"{prefix}_GRADIENT_CLIP_NORM"), 1.0)
    use_classifier_lr = to_bool(config.get(f"{prefix}_USE_CLASSIFIER_LEARNING_RATE"), False)
    classifier_lr = (
        _to_float(config.get(f"{prefix}_CLASSIFIER_LEARNING_RATE"), 1e-4)
        if use_classifier_lr
        else None
    )

    optimizer = _build_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        classifier_learning_rate=classifier_lr,
    )
    num_training_steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
    amp_dtype = _get_torch_dtype(mixed_precision_dtype)
    use_amp = use_mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with context:
                outputs = model(**batch)
                loss = outputs.loss.mean()
                scaled_loss = loss / gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = step % gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_train_loss += float(loss.detach().cpu()) * batch["labels"].size(0)
            total_train_examples += batch["labels"].size(0)
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        validation_loss, y_true, y_pred, metrics = _evaluate_classifier(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_train_loss / max(1, total_train_examples),
                "validation_loss": validation_loss,
                **metrics,
            }
        )
        print(history[-1])

    validation_loss, y_true, y_pred, final_metrics = _evaluate_classifier(model, test_loader, device)
    return history, y_true, y_pred, final_metrics, validation_loss


def _normalize_generated_label(text: str, label_names: list[str]) -> int:
    cleaned = str(text).strip().lower()
    for index, label_name in enumerate(label_names):
        label = label_name.lower()
        if cleaned.startswith(label) or label in cleaned:
            return index
    return -1


def _instruction_prompts(
    texts: list[str],
    *,
    tokenizer: Any,
    label_names: list[str],
    use_chat_template: bool,
) -> list[str]:
    prompts = []
    options = ", ".join(label_names)
    for text in texts:
        content = f"Classify the sentiment. Labels: {options}\nReview: {text}\nOutput:"
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            prompts.append(f"Input: Classify the sentiment.\nReview: {text}\nOutput:")
    return prompts


def _evaluate_instruction_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating instruction loss", unit="batch"):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "class_labels"
            }
            outputs = model(**batch)
            loss = outputs.loss.mean()
            total_loss += float(loss.detach().cpu()) * batch["input_ids"].size(0)
            total_examples += batch["input_ids"].size(0)
    return total_loss / max(1, total_examples)


def _evaluate_instruction_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    labels: np.ndarray,
    *,
    label_names: list[str],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], pd.DataFrame]:
    model.eval()
    batch_size = to_int(config.get("DECODER_BATCH_SIZE"), 8)
    max_length = to_int(config.get("DECODER_MAX_LENGTH"), 256)
    use_chat_template = to_bool(config.get("DECODER_USE_CHAT_TEMPLATE"), True)
    max_new_tokens = to_int(config.get("DECODER_GENERATION_MAX_NEW_TOKENS"), 8)
    do_sample = to_bool(config.get("DECODER_GENERATION_DO_SAMPLE"), False)
    temperature = _to_float(config.get("DECODER_GENERATION_TEMPERATURE"), 0.1)
    y_pred = []
    decoded_outputs = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    terminators = []
    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Generating labels", unit="batch"):
            batch_texts = texts[start : start + batch_size]
            prompts = _instruction_prompts(
                batch_texts,
                tokenizer=tokenizer,
                label_names=label_names,
                use_chat_template=use_chat_template,
            )
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": terminators if terminators else tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
            generated = model.generate(**encoded, **generation_kwargs)
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            decoded_outputs.extend(decoded)
            y_pred.extend([_normalize_generated_label(output, label_names) for output in decoded])

    tokenizer.padding_side = original_padding_side
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred_array)
    prediction_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": [label_names[int(label_id)] for label_id in y_true],
            "generated_output": decoded_outputs,
            "predicted_label": [
                label_names[int(label_id)] if 0 <= int(label_id) < len(label_names) else "UNKNOWN"
                for label_id in y_pred_array
            ],
        }
    )
    return y_true, y_pred_array, metrics, prediction_df


def _train_instruction(
    model: torch.nn.Module,
    tokenizer: Any,
    train_loader: DataLoader,
    test_loader: DataLoader,
    texts_for_eval: list[str],
    labels_for_eval: np.ndarray,
    config: dict[str, Any],
    *,
    label_names: list[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, float], float, pd.DataFrame]:
    epochs = to_int(config.get("DECODER_EPOCHS"), 1)
    learning_rate = _to_float(config.get("DECODER_LEARNING_RATE"), 2e-5)
    weight_decay = _to_float(config.get("DECODER_WEIGHT_DECAY"), 0.01)
    warmup_ratio = _to_float(config.get("DECODER_WARMUP_RATIO"), 0.1)
    gradient_clip_norm = _to_float(config.get("DECODER_GRADIENT_CLIP_NORM"), 1.0)
    optimizer = _build_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        classifier_learning_rate=None,
    )
    num_training_steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in progress:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "class_labels"
            }
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm, foreach=False)
            optimizer.step()
            scheduler.step()
            total_train_loss += float(loss.detach().cpu()) * batch["input_ids"].size(0)
            total_train_examples += batch["input_ids"].size(0)
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        validation_loss = _evaluate_instruction_loss(model, test_loader, device)
        y_true, y_pred, metrics, prediction_df = _evaluate_instruction_generation(
            model,
            tokenizer,
            texts_for_eval,
            labels_for_eval,
            label_names=label_names,
            config=config,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_train_loss / max(1, total_train_examples),
                "validation_loss": validation_loss,
                **metrics,
            }
        )
        print(history[-1])
    validation_loss = _evaluate_instruction_loss(model, test_loader, device)
    y_true, y_pred, final_metrics, prediction_df = _evaluate_instruction_generation(
        model,
        tokenizer,
        texts_for_eval,
        labels_for_eval,
        label_names=label_names,
        config=config,
        device=device,
    )
    return history, y_true, y_pred, final_metrics, validation_loss, prediction_df


class EncoderFineTunePipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        self.runs = self.config.get("ENCODER_RUNS") or [{"name": "default"}]

    def run(self) -> dict[str, Any]:
        results = [self._run_one(dict(run_config)) for run_config in self.runs]
        return {"runs": results, "summary_path": str(self._write_summary(results))}

    def _run_one(self, run_config: dict[str, Any]) -> dict[str, Any]:
        config = {**self.config, **{f"ENCODER_{key.upper()}": value for key, value in run_config.items()}}
        device = _resolve_device(to_bool(config.get("ENCODER_REQUIRE_CUDA"), False))
        random_state = to_int(config.get("ENCODER_RANDOM_STATE"), 42)
        _seed_everything(random_state, device)

        x_train, x_test, y_train, y_test, _, label_names = _load_split(config, prefix="ENCODER")
        model_name = str(config.get("ENCODER_MODEL_NAME", "bert-base-uncased"))
        model_slug = _slug(model_name)
        trust_remote_code = to_bool(config.get("ENCODER_TRUST_REMOTE_CODE"), True)
        max_length = to_int(config.get("ENCODER_MAX_LENGTH"), 64)
        batch_size = to_int(config.get("ENCODER_BATCH_SIZE"), 32)
        use_zero_token_type_ids = to_bool(config.get("ENCODER_USE_ZERO_TOKEN_TYPE_IDS"), False)
        use_peft = to_bool(config.get("ENCODER_USE_PEFT"), False)
        manual_head = to_bool(config.get("ENCODER_ACCELERATOR_ERROR"), False)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer_added_pad_token = False
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                tokenizer_added_pad_token = True

        train_loader = DataLoader(
            EncoderTextDataset(
                x_train.tolist(),
                y_train,
                tokenizer,
                max_length,
                use_zero_token_type_ids=use_zero_token_type_ids,
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            EncoderTextDataset(
                x_test.tolist(),
                y_test,
                tokenizer,
                max_length,
                use_zero_token_type_ids=use_zero_token_type_ids,
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        target_modules = None
        if manual_head:
            encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            if tokenizer_added_pad_token:
                encoder.resize_token_embeddings(len(tokenizer))
            if use_peft:
                LoraConfig, TaskType, get_peft_model = _import_peft()
                target_modules = _to_string_list(config.get("ENCODER_PEFT_TARGET_MODULES")) or _infer_lora_target_modules(encoder, decoder=False)
                encoder = get_peft_model(
                    encoder,
                    LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        inference_mode=False,
                        r=to_int(config.get("ENCODER_PEFT_R"), 16),
                        lora_alpha=to_int(config.get("ENCODER_PEFT_ALPHA"), 32),
                        lora_dropout=_to_float(config.get("ENCODER_PEFT_DROPOUT"), 0.05),
                        target_modules=target_modules,
                    ),
                )
            model = ManualEncoderClassifier(encoder, num_labels=len(label_names))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(label_names),
                id2label={index: label for index, label in enumerate(label_names)},
                label2id={label: index for index, label in enumerate(label_names)},
                trust_remote_code=trust_remote_code,
            )
            if tokenizer_added_pad_token:
                model.resize_token_embeddings(len(tokenizer))
            if use_peft:
                LoraConfig, TaskType, get_peft_model = _import_peft()
                target_modules = _to_string_list(config.get("ENCODER_PEFT_TARGET_MODULES")) or _infer_lora_target_modules(model, decoder=False)
                model = get_peft_model(
                    model,
                    LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        inference_mode=False,
                        r=to_int(config.get("ENCODER_PEFT_R"), 16),
                        lora_alpha=to_int(config.get("ENCODER_PEFT_ALPHA"), 32),
                        lora_dropout=_to_float(config.get("ENCODER_PEFT_DROPOUT"), 0.05),
                        target_modules=target_modules,
                        modules_to_save=_to_string_list(config.get("ENCODER_PEFT_MODULES_TO_SAVE")),
                    ),
                )

        use_dataparallel = to_bool(config.get("ENCODER_USE_DATAPARALLEL"), True)
        if use_dataparallel and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        history, y_true, y_pred, final_metrics, validation_loss = _train_classifier(
            model,
            train_loader,
            test_loader,
            config,
            prefix="ENCODER",
            device=device,
        )

        output_root = resolve_path(config.get("ENCODER_OUTPUT_DIR"), "models/fine_tuned_encoder_models")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tuning_mode = f"peft_lora_r{to_int(config.get('ENCODER_PEFT_R'), 16)}" if use_peft else "full"
        head_mode = "manual_head" if manual_head else "auto_seq_cls"
        run_id = _slug(
            f"{model_slug}_{head_mode}_{tuning_mode}_len{max_length}_bs{batch_size}_"
            f"ep{to_int(config.get('ENCODER_EPOCHS'), 1)}_lr{_to_float(config.get('ENCODER_LEARNING_RATE'), 1e-5)}_{timestamp}"
        )
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        if manual_head:
            model_dir = output_dir / "model"
            encoder_dir = model_dir / "encoder"
            encoder_dir.mkdir(parents=True, exist_ok=True)
            model_to_save.encoder.save_pretrained(encoder_dir)
            torch.save(model_to_save.classifier.state_dict(), model_dir / "classifier_head.pt")
            (model_dir / "manual_classifier_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name": model_name,
                        "num_labels": len(label_names),
                        "labels": label_names,
                        "hidden_size": int(getattr(model_to_save.config, "hidden_size")),
                        "pooling": "cls",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            model_to_save.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        self._write_outputs(
            output_dir,
            y_true=y_true,
            y_pred=y_pred,
            texts=x_test.tolist(),
            label_names=label_names,
            metadata={
                **self._common_metadata(config, "ENCODER"),
                "model_name": model_name,
                "model_slug": model_slug,
                "trust_remote_code": trust_remote_code,
                "tokenizer_pad_token": tokenizer.pad_token,
                "tokenizer_pad_token_id": tokenizer.pad_token_id,
                "tokenizer_added_pad_token": tokenizer_added_pad_token,
                "accelerator_error_manual_head": manual_head,
                "use_zero_token_type_ids": use_zero_token_type_ids,
                "task": "sequence_classification",
                "model_head_mode": head_mode,
                "tuning_mode": tuning_mode,
                "use_peft": use_peft,
                "peft_target_modules": target_modules if use_peft else None,
                "history": history,
                "validation_loss": validation_loss,
                "metrics": final_metrics,
                "label_names": label_names,
                "device": str(device),
            },
        )
        return {"run_id": run_id, "output_dir": str(output_dir), "metrics": final_metrics}

    def _write_summary(self, results: list[dict[str, Any]]) -> Path:
        output_root = resolve_path(self.config.get("ENCODER_OUTPUT_DIR"), "models/fine_tuned_encoder_models")
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "encoder_summary.json"
        summary_path.write_text(json.dumps(json_compatible({"runs": results}), indent=2), encoding="utf-8")
        return summary_path

    def _common_metadata(self, config: dict[str, Any], prefix: str) -> dict[str, Any]:
        return _common_metadata(config, prefix)

    def _write_outputs(self, *args, **kwargs) -> None:
        _write_outputs(*args, **kwargs)


class DecoderFineTunePipeline(EncoderFineTunePipeline):
    def run(self) -> dict[str, Any]:
        self.runs = self.config.get("DECODER_RUNS") or [{"name": "default"}]
        results = [self._run_decoder_one(dict(run_config)) for run_config in self.runs]
        output_root = resolve_path(self.config.get("DECODER_OUTPUT_DIR"), "models/fine_tuned_decoder_models")
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "decoder_summary.json"
        summary_path.write_text(json.dumps(json_compatible({"runs": results}), indent=2), encoding="utf-8")
        return {"runs": results, "summary_path": str(summary_path)}

    def _run_decoder_one(self, run_config: dict[str, Any]) -> dict[str, Any]:
        config = {**self.config, **{f"DECODER_{key.upper()}": value for key, value in run_config.items()}}
        device = _resolve_device(to_bool(config.get("DECODER_REQUIRE_CUDA"), False))
        random_state = to_int(config.get("DECODER_RANDOM_STATE"), 42)
        _seed_everything(random_state, device)
        x_train, x_test, y_train, y_test, _, label_names = _load_split(config, prefix="DECODER")

        model_name = str(config.get("DECODER_MODEL_NAME", "openai-community/gpt2"))
        model_slug = _slug(model_name)
        trust_remote_code = to_bool(config.get("DECODER_TRUST_REMOTE_CODE"), True)
        approach = str(config.get("DECODER_APPROACH", "classification_head")).strip().lower()
        fine_tune_model = to_bool(config.get("DECODER_FINE_TUNE_MODEL"), True)
        max_length = to_int(config.get("DECODER_MAX_LENGTH"), 256)
        batch_size = to_int(config.get("DECODER_BATCH_SIZE"), 8)
        use_peft = to_bool(config.get("DECODER_USE_PEFT"), False)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer_added_pad_token = False
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                tokenizer_added_pad_token = True

        target_modules = None
        if approach == "instruction":
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            if tokenizer_added_pad_token:
                model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.use_cache = False
            if use_peft:
                LoraConfig, TaskType, get_peft_model = _import_peft()
                target_modules = _to_string_list(config.get("DECODER_PEFT_TARGET_MODULES")) or _infer_lora_target_modules(model, decoder=True)
                model = get_peft_model(
                    model,
                    LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=to_int(config.get("DECODER_PEFT_R"), 16),
                        lora_alpha=to_int(config.get("DECODER_PEFT_ALPHA"), 32),
                        lora_dropout=_to_float(config.get("DECODER_PEFT_DROPOUT"), 0.05),
                        target_modules=target_modules,
                    ),
                )
            train_dataset = DecoderInstructionDataset(
                x_train.tolist(),
                y_train,
                tokenizer,
                max_length,
                label_names=label_names,
                use_chat_template=to_bool(config.get("DECODER_USE_CHAT_TEMPLATE"), True),
            )
            test_dataset = DecoderInstructionDataset(
                x_test.tolist(),
                y_test,
                tokenizer,
                max_length,
                label_names=label_names,
                use_chat_template=to_bool(config.get("DECODER_USE_CHAT_TEMPLATE"), True),
            )
        elif approach == "classification_head":
            decoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            if tokenizer_added_pad_token:
                decoder.resize_token_embeddings(len(tokenizer))
            decoder.config.pad_token_id = tokenizer.pad_token_id
            decoder.config.use_cache = False
            if use_peft:
                LoraConfig, TaskType, get_peft_model = _import_peft()
                target_modules = _to_string_list(config.get("DECODER_PEFT_TARGET_MODULES")) or _infer_lora_target_modules(decoder, decoder=True)
                decoder = get_peft_model(
                    decoder,
                    LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        inference_mode=False,
                        r=to_int(config.get("DECODER_PEFT_R"), 16),
                        lora_alpha=to_int(config.get("DECODER_PEFT_ALPHA"), 32),
                        lora_dropout=_to_float(config.get("DECODER_PEFT_DROPOUT"), 0.05),
                        target_modules=target_modules,
                    ),
                )
            elif str(config.get("DECODER_CLASSIFICATION_TUNING_MODE", "full_decoder")).lower() == "classifier_only":
                _freeze_module(decoder)
            model = DecoderHiddenStateClassifier(
                decoder,
                num_labels=len(label_names),
                dropout_prob=_to_float(config.get("DECODER_CLASSIFIER_DROPOUT"), 0.1),
            )
            train_dataset = DecoderClassificationDataset(x_train.tolist(), y_train, tokenizer, max_length)
            test_dataset = DecoderClassificationDataset(x_test.tolist(), y_test, tokenizer, max_length)
        else:
            raise ValueError("decoder approach must be 'instruction' or 'classification_head'")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if to_bool(config.get("DECODER_USE_DATAPARALLEL"), True) and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        instruction_prediction_df = None
        if approach == "instruction":
            if fine_tune_model:
                (
                    history,
                    y_true,
                    y_pred,
                    final_metrics,
                    validation_loss,
                    instruction_prediction_df,
                ) = _train_instruction(
                    model,
                    tokenizer,
                    train_loader,
                    test_loader,
                    x_test.tolist(),
                    y_test,
                    config,
                    label_names=label_names,
                    device=device,
                )
            else:
                history = []
                validation_loss = _evaluate_instruction_loss(model, test_loader, device)
                y_true, y_pred, final_metrics, instruction_prediction_df = _evaluate_instruction_generation(
                    model,
                    tokenizer,
                    x_test.tolist(),
                    y_test,
                    label_names=label_names,
                    config=config,
                    device=device,
                )
        elif fine_tune_model:
            history, y_true, y_pred, final_metrics, validation_loss = _train_classifier(
                model,
                train_loader,
                test_loader,
                config,
                prefix="DECODER",
                device=device,
            )
        else:
            history = []
            validation_loss, y_true, y_pred, final_metrics = _evaluate_classifier(model, test_loader, device)

        output_root = resolve_path(config.get("DECODER_OUTPUT_DIR"), "models/fine_tuned_decoder_models")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tuning_mode = "inference_only" if not fine_tune_model else ("peft_lora" if use_peft else str(config.get("DECODER_CLASSIFICATION_TUNING_MODE", "full")))
        run_id = _slug(
            f"{model_slug}_{approach}_{tuning_mode}_len{max_length}_bs{batch_size}_"
            f"ep{to_int(config.get('DECODER_EPOCHS'), 1)}_lr{_to_float(config.get('DECODER_LEARNING_RATE'), 1e-5)}_{timestamp}"
        )
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        if fine_tune_model and approach == "instruction":
            model_to_save.save_pretrained(output_dir / "model")
        elif approach == "classification_head":
            model_dir = output_dir / "model"
            decoder_dir = model_dir / "decoder"
            decoder_dir.mkdir(parents=True, exist_ok=True)
            model_to_save.decoder.save_pretrained(decoder_dir)
            torch.save(model_to_save.classifier.state_dict(), model_dir / "classifier_head.pt")
            (model_dir / "decoder_classifier_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name": model_name,
                        "num_labels": len(label_names),
                        "labels": label_names,
                        "hidden_size": int(getattr(model_to_save.config, "hidden_size", None) or getattr(model_to_save.config, "n_embd")),
                        "pooling": "last_non_padding_token",
                        "dropout": _to_float(config.get("DECODER_CLASSIFIER_DROPOUT"), 0.1),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        tokenizer.save_pretrained(output_dir / "tokenizer")
        _write_outputs(
            output_dir,
            y_true=y_true,
            y_pred=y_pred,
            texts=x_test.tolist(),
            label_names=label_names,
            prediction_df=instruction_prediction_df,
            metadata={
                **_common_metadata(config, "DECODER"),
                "model_name": model_name,
                "model_slug": model_slug,
                "trust_remote_code": trust_remote_code,
                "tokenizer_pad_token": tokenizer.pad_token,
                "tokenizer_pad_token_id": tokenizer.pad_token_id,
                "tokenizer_added_pad_token": tokenizer_added_pad_token,
                "approach": approach,
                "fine_tune_model": fine_tune_model,
                "task": "decoder_sentiment_classification",
                "tuning_mode": tuning_mode,
                "classification_tuning_mode": config.get("DECODER_CLASSIFICATION_TUNING_MODE") if approach == "classification_head" else None,
                "use_peft": use_peft,
                "peft_target_modules": target_modules if use_peft else None,
                "history": history,
                "validation_loss": validation_loss,
                "metrics": final_metrics,
                "label_names": label_names,
                "device": str(device),
            },
        )
        return {"run_id": run_id, "output_dir": str(output_dir), "metrics": final_metrics}


def _common_metadata(config: dict[str, Any], prefix: str) -> dict[str, Any]:
    dataset_path = resolve_path(
        config.get(f"{prefix}_DATASET_PATH") or config.get("CLEANED_DATASET_PATH") or config.get("DATASET_PATH"),
        "datasets/cleaned_dataset.csv",
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "text_column": str(config.get(f"{prefix}_TEXT_COLUMN", "text")),
        "label_column": str(config.get(f"{prefix}_LABEL_COLUMN", "label")),
        "test_size": _to_float(config.get(f"{prefix}_TEST_SIZE"), 0.2),
        "random_state": to_int(config.get(f"{prefix}_RANDOM_STATE"), 42),
        "use_stratify": to_bool(config.get(f"{prefix}_USE_STRATIFY"), True),
        "max_length": to_int(config.get(f"{prefix}_MAX_LENGTH"), 64),
        "batch_size": to_int(config.get(f"{prefix}_BATCH_SIZE"), 32),
        "epochs": to_int(config.get(f"{prefix}_EPOCHS"), 1),
        "learning_rate": _to_float(config.get(f"{prefix}_LEARNING_RATE"), 1e-5),
        "weight_decay": _to_float(config.get(f"{prefix}_WEIGHT_DECAY"), 0.01),
        "warmup_ratio": _to_float(config.get(f"{prefix}_WARMUP_RATIO"), 0.1),
    }


def _write_outputs(
    output_dir: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: list[str],
    label_names: list[str],
    prediction_df: pd.DataFrame | None = None,
    metadata: dict[str, Any],
) -> None:
    joblib.dump(y_pred, output_dir / "y_pred.joblib")
    joblib.dump(y_true, output_dir / "y_true.joblib")
    if prediction_df is None:
        prediction_df = pd.DataFrame(
            {
                "text": texts,
                "true_label": [label_names[int(label_id)] for label_id in y_true],
                "predicted_label": [
                    label_names[int(label_id)] if 0 <= int(label_id) < len(label_names) else "UNKNOWN"
                    for label_id in y_pred
                ],
            }
        )
    prediction_df.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(json_compatible(metadata), indent=2),
        encoding="utf-8",
    )


def run_encoder_pipeline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return EncoderFineTunePipeline(config).run()


def run_decoder_pipeline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return DecoderFineTunePipeline(config).run()
