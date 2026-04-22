from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sentimentFlow.src.utils import CONFIG, load_config_paths
    from sentimentFlow.src.extraction import preprocess_and_save
else:
    from ..utils import CONFIG, load_config_paths
    from ..extraction import preprocess_and_save

def load_config():
    load_config_paths()

def run_preprocessing():
    load_config()
    dataset_path = CONFIG.get("DATASET_PATH")
    output_path = CONFIG.get("CLEANED_DATASET_PATH")
    preprocess_and_save(dataset_path, output_path)

if __name__ == "__main__":
    run_preprocessing()

