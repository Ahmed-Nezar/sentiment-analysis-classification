from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sentimentFlow.src.utils import CONFIG, load_config_paths
    from sentimentFlow.src.extraction import preprocess_and_save, save_noise_removed_dataset
else:
    from ..utils import CONFIG, load_config_paths
    from ..extraction import preprocess_and_save, save_noise_removed_dataset

def load_config():
    load_config_paths()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess sentiment datasets or generate noise-removed datasets."
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Generate a noise-removed dataset using the hardcoded strict noisy/ambiguous row list.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Use the cleaned dataset row list. With --no-noise, writes cleaned_dataset_noise_removed.csv.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional input CSV path. Defaults to config paths or datasets/cleaned_dataset.csv with --clean --no-noise.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    return parser


def run_preprocessing(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    load_config()
    if args.no_noise:
        if args.clean:
            dataset_path = args.dataset_path or Path("datasets/cleaned_dataset.csv")
            output_path = args.output_path or Path("datasets/cleaned_dataset_noise_removed.csv")
        else:
            dataset_path = args.dataset_path or Path(CONFIG.get("DATASET_PATH", "datasets/dataset.csv"))
            output_path = args.output_path or Path("datasets/dataset_noise_removed.csv")
        saved_path = save_noise_removed_dataset(dataset_path, output_path, clean=args.clean)
        print(f"Noise-removed dataset saved to: {saved_path}")
        return

    dataset_path = args.dataset_path or CONFIG.get("DATASET_PATH")
    output_path = args.output_path or CONFIG.get("CLEANED_DATASET_PATH")
    saved_path = preprocess_and_save(dataset_path, output_path)
    print(f"Preprocessing completed. Cleaned dataset saved to: {saved_path}")

if __name__ == "__main__":
    run_preprocessing()

