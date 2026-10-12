"""Pre-download the foundation-model checkpoints into the Hugging Face cache.

Overnight runs should never stall on (or fail because of) a first-use download,
so fetch the weights once, up front. The repos are the ones the darts wrappers
load (see the ``hub_model_name`` defaults in darts and configs/experiment.yaml).

Usage:
    python experiments/fetch_checkpoints.py
    python experiments/fetch_checkpoints.py --all    # + the extras' and covariate runs' checkpoints
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

CHECKPOINTS = {
    "chronos2": "autogluon/chronos-2-small",       # 28M; zero-shot + fine-tuned rows
    "timesfm": "google/timesfm-2.5-200m-pytorch",  # 200M; zero-shot
    "tirex": "NX-AI/TiRex",                        # 35M; zero-shot (needs tirex-ts)
}

#: configs/extras.yaml and configs/covariates.yaml (COVARIATES.md); fetched with --all
EXTRA_CHECKPOINTS = {
    "chronos2_base": "amazon/chronos-2",           # 120M; Chronos-B rows
    "timesfm3": "google/timesfm-3.0-pytorch",      # ~330M; TimesFM non-commercial licence v1.0
    "tirex2": "NX-AI/TiRex-2",                     # 38M; gated: accept on the model page and
}                                                  #   `hf auth login` first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also the extras' and covariate runs' checkpoints")
    args = ap.parse_args()
    wanted = {**CHECKPOINTS, **(EXTRA_CHECKPOINTS if args.all else {})}
    for arch, repo in wanted.items():
        path = snapshot_download(repo_id=repo)
        print(f"{arch:9s} {repo:34s} -> {path}")


if __name__ == "__main__":
    main()
