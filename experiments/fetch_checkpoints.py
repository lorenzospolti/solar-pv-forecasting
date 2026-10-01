"""Pre-download the foundation-model checkpoints into the Hugging Face cache.

Overnight runs should never stall on (or fail because of) a first-use download,
so fetch the weights once, up front. The repos are the ones the darts wrappers
load (see the ``hub_model_name`` defaults in darts and configs/experiment.yaml).

Usage:
    python experiments/fetch_checkpoints.py
"""
from __future__ import annotations

from huggingface_hub import snapshot_download

CHECKPOINTS = {
    "chronos2": "autogluon/chronos-2-small",       # 28M; zero-shot + fine-tuned rows
    "timesfm": "google/timesfm-2.5-200m-pytorch",  # 200M; zero-shot
    "tirex": "NX-AI/TiRex",                        # 35M; zero-shot (needs tirex-ts)
}


def main() -> None:
    for arch, repo in CHECKPOINTS.items():
        path = snapshot_download(repo_id=repo)
        print(f"{arch:9s} {repo:34s} -> {path}")


if __name__ == "__main__":
    main()
