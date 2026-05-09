from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from sentimentFlow.src.train import run_decoder_pipeline
    from sentimentFlow.src.utils import load_config
else:
    from ..train import run_decoder_pipeline
    from ..utils import load_config


def run_decoder_models() -> None:
    config = load_config()
    summary = run_decoder_pipeline(config)
    print(
        "Decoder fine-tuning pipeline finished. "
        f"Summary saved at: {summary.get('summary_path', 'unknown')}"
    )


if __name__ == "__main__":
    run_decoder_models()
