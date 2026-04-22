from __future__ import annotations

from pathlib import Path
import re
from typing import Final

import pandas as pd


URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"http\S+|www\S+|https\S+")
SPECIAL_CHAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9\s]")

def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
	return pd.read_csv(dataset_path)


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
	if "id" in df.columns:
		return df.drop(columns=["id"])
	return df.copy()


def remove_url_rows(df: pd.DataFrame) -> pd.DataFrame:
	return df[~df["text"].astype(str).str.contains(URL_PATTERN, na=False)].copy()


def remove_special_characters(df: pd.DataFrame) -> pd.DataFrame:
	cleaned_df = df.copy()
	cleaned_df["text"] = cleaned_df["text"].astype(str).apply(
		lambda value: re.sub(SPECIAL_CHAR_PATTERN, "", value)
	)
	return cleaned_df


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
	normalized_df = df.copy()
	normalized_df["text"] = normalized_df["text"].astype(str).str.lower()
	return normalized_df


def remove_empty_text_rows(df: pd.DataFrame) -> pd.DataFrame:
	filtered_df = df[df["text"].astype(str).str.strip() != ""].copy()
	filtered_df.reset_index(drop=True, inplace=True)
	return filtered_df


def drop_duplicate_text_rows(df: pd.DataFrame) -> pd.DataFrame:
	deduplicated_df = df.drop_duplicates(subset=["text"]).copy()
	deduplicated_df.reset_index(drop=True, inplace=True)
	return deduplicated_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
	cleaned_df = drop_id_column(df)
	cleaned_df = remove_url_rows(cleaned_df)
	cleaned_df = remove_special_characters(cleaned_df)
	cleaned_df = normalize_text(cleaned_df)
	cleaned_df = remove_empty_text_rows(cleaned_df)
	cleaned_df = drop_duplicate_text_rows(cleaned_df)
	return cleaned_df


def save_cleaned_dataset(
	df: pd.DataFrame,
	output_path: str | Path | None = None,
) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	return output_path


def preprocess_and_save(
	dataset_path: str | Path | None = None,
	output_path: str | Path | None = None,
) -> Path:
	dataset_df = load_dataset(dataset_path)
	cleaned_df = clean_dataset(dataset_df)
	return save_cleaned_dataset(cleaned_df, output_path)
