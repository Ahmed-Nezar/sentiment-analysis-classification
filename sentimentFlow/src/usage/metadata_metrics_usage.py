from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sentimentFlow.src.evaluation import run_metadata_metrics_pipeline
    from sentimentFlow.src.utils import PROJECT_ROOT
else:
    from ..evaluation import run_metadata_metrics_pipeline
    from ..utils import PROJECT_ROOT


def _path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find model metadata.json files and backfill confusion matrices, "
            "support counts, and per-class precision/recall/F1 scores from saved predictions."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=_path,
        default=PROJECT_ROOT / "models",
        help=(
            "Root folder to scan for metadata.json files. Defaults to models/. "
            "You can pass a model folder, such as models/ml_models/..."
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
        "--dry-run",
        action="store_true",
        help="Resolve metadata files and metrics without writing changes.",
    )
    return parser


def run_metadata_metrics(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_dirs = args.model_dir
    model_roots = None if model_dirs else [args.root]
    summary = run_metadata_metrics_pipeline(
        model_roots=model_roots,
        model_dirs=model_dirs,
        dry_run=args.dry_run,
    )
    action = "Checked" if args.dry_run else "Updated"
    print(
        f"{action} metadata metrics. "
        f"Updated: {summary['updated_count']}, skipped: {summary['skipped_count']}."
    )
    skipped = [item for item in summary["results"] if item.get("status") == "skipped"]
    if skipped:
        print("Skipped files:")
        for item in skipped[:20]:
            print(f"- {item['model_dir']}: {item['reason']}")
        if len(skipped) > 20:
            print(f"- ... {len(skipped) - 20} more")


if __name__ == "__main__":
    run_metadata_metrics()
