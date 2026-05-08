# Noise Removal Criteria

This project keeps the original datasets unchanged and creates derived noise-removed datasets:

- `datasets/dataset_noise_removed.csv`
- `datasets/cleaned_dataset_noise_removed.csv`

Rows are removed by 1-based CSV data-row number, where row 1 means the first record after the header. The row numbers are hardcoded in `sentimentFlow/src/extraction/noise_rows.py` for reproducibility.

## How To Generate

Generate the raw dataset without noisy/ambiguous rows:

```powershell
python sentimentFlow/src/usage/usage.py --no-noise
```

Generate the cleaned dataset without noisy/ambiguous rows:

```powershell
python sentimentFlow/src/usage/usage.py --no-noise --clean
```

Optional custom paths:

```powershell
python sentimentFlow/src/usage/usage.py --no-noise --dataset-path datasets/dataset.csv --output-path datasets/dataset_noise_removed.csv
python sentimentFlow/src/usage/usage.py --no-noise --clean --dataset-path datasets/cleaned_dataset.csv --output-path datasets/cleaned_dataset_noise_removed.csv
```

## Selection Criteria

The strict noise-removal pass removes rows matching any of these categories:

| Criterion | Meaning |
|---|---|
| `previously_confirmed_noisy` | Rows manually/subagent-reviewed earlier and confirmed as clear noisy labels, corrupted text, or sentiment-impossible rows. |
| `too_short_or_sentiment_impossible` | Text is too short or vague to infer sentiment reliably, such as one-word fragments or unsupported tokens. |
| `mostly_symbols_or_nonlinguistic` | Text is mostly symbols, masked content, emoji-only content, or otherwise non-linguistic. |
| `corrupted_or_spreadsheet_error_text` | Text contains corrupted artifacts or spreadsheet-style errors such as `#NAME?`. |
| `positive_label_with_negative_language` | The label is positive, but the text contains strong negative sentiment cues. |
| `negative_label_with_positive_language` | The label is negative, but the text contains strong positive sentiment cues. |
| `neutral_label_with_positive_language` | The label is neutral, but the text contains strong positive sentiment cues. |
| `neutral_label_with_negative_language` | The label is neutral, but the text contains strong negative sentiment cues. |
| `mixed_positive_negative_sentiment` | The text contains both positive and negative cues, making the row ambiguous for a single clean sentiment label. |

## Previously Confirmed Noisy

`previously_confirmed_noisy` means the row was part of the earlier human/subagent review before the stricter automated pass. Those rows were selected conservatively because they were obvious label contradictions or unusable samples.

Examples include:

- Positive text labeled negative, such as `Nice good`.
- Negative text labeled positive, such as `Reminder doesn't work`.
- Neutral rows with clearly positive or negative sentiment, such as `Best app` or `It is not working`.
- Very short or unsupported rows, such as `apk`, `tv`, or symbol-only text.
- Corrupted or garbled text where sentiment could not be reliably inferred.

## Removal Counts

Current strict removal counts:

| Dataset | Original Rows | Removed Rows | Remaining Rows |
|---|---:|---:|---:|
| `datasets/dataset.csv` | 31,232 | 7,354 | 23,878 |
| `datasets/cleaned_dataset.csv` | 29,725 | 8,076 | 21,649 |

## Important Note

The derived datasets are stricter and cleaner, but sentiment labeling is partly subjective. This process removes rows likely to confuse training or evaluation; it should not be read as a mathematical guarantee that every remaining row is perfectly labeled.
