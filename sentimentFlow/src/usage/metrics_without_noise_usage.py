from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sentimentFlow.src.evaluation import run_noise_metrics_pipeline
    from sentimentFlow.src.utils import PROJECT_ROOT
else:
    from ..evaluation import run_noise_metrics_pipeline
    from ..utils import PROJECT_ROOT


def _path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute saved model evaluation metrics after removing hardcoded "
            "strict noisy/ambiguous evaluation rows. This does not retrain models."
        )
    )
    parser.add_argument(
        "--model-root",
        action="append",
        type=_path,
        default=None,
        help=(
            "Model root to scan, such as models/ml_models. Can be provided more "
            "than once. Defaults to ml_models, dl_models, encoder_models, and decoder_models."
        ),
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=_path,
        default=None,
        help="Specific model directory containing metadata.json. Can be provided more than once.",
    )
    parser.add_argument(
        "--output-name",
        default="metrics_without_noise.json",
        help="Output filename written inside each model directory.",
    )
    return parser


def run_metrics_without_noise(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_noise_metrics_pipeline(
        model_roots=args.model_root,
        model_dirs=args.model_dir,
        output_name=args.output_name,
    )
    print(
        "Metrics without noise generation finished. "
        f"Generated: {summary['generated_count']}, skipped: {summary['skipped_count']}."
    )
    if summary["summary_paths"]:
        print("Summary files:")
        for summary_path in summary["summary_paths"]:
            print(f"- {summary_path}")


if __name__ == "__main__":
    run_metrics_without_noise()
