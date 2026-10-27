"""Pre-download the foundation-model checkpoints into the Hugging Face cache.

Overnight runs should never stall on (or fail because of) a first-use download,
so fetch the weights once, up front; the run then holds them fixed with
``HF_HUB_OFFLINE=1``. The repos are the ones the wrappers load (the ``hub_model_name``
and ``revision`` params in configs/experiment.yaml, DEFINITIVE_RUN.md section 3).
All four are fetched by commit hash, which writes no ``refs/main``:

- TimesFM-3 and TiRex-2: the wrapper passes the same ``revision`` to the loader.
- Chronos-2 base and small: darts loads ``hub_model_name`` at ``refs/main``, so
  ``refs/main`` must name the pin. It is written when missing; a different one (an
  online load moved it) stops the script instead of changing the run silently.

Usage:
    python experiments/fetch_checkpoints.py
    python experiments/fetch_checkpoints.py --all    # + TimesFM-2.5 and TiRex (older configs)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

#: the definitive run's checkpoints: arch label -> (repo, snapshot_download kwargs)
CHECKPOINTS: dict[str, tuple[str, dict]] = {
    "chronos2": ("amazon/chronos-2",                       # 120M; Chronos-B
                 {"revision": "29ec3766d36d6f73f0696f85560a422f50e8498c"}),
    "chronos2_small": ("autogluon/chronos-2-small",        # 28M; Chronos-S, Chronos-S-FT
                       {"revision": "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"}),
    "timesfm3": ("google/timesfm-3.0-pytorch",             # 331M; TimesFM non-commercial v1.0
                 {"revision": "43046b85ec22d584a13f8098c2ed39c889e129c2"}),
    "tirex2": ("NX-AI/TiRex-2",                            # 82.5M; Apache-2.0, not gated
               {"revision": "30788a311699acba7e6393ad9700ea7b7cb9bb2d",
                "allow_patterns": ["model-config.yaml", "model.ckpt"]}),
}
#: loaded by darts at refs/main (no revision param): refs/main must name the pin
LOADED_AT_MAIN = ("chronos2", "chronos2_small")

#: TimesFM-2.5 and TiRex: panel v2, configs/extras.yaml and configs/covariates.yaml; with --all
EXTRA_CHECKPOINTS: dict[str, tuple[str, dict]] = {
    "timesfm": ("google/timesfm-2.5-200m-pytorch", {}),    # 200M
    "tirex": ("NX-AI/TiRex", {}),                          # 35M (needs tirex-ts)
}


def refs_main(repo: str) -> Path:
    """The cache file that maps ``main`` to a commit for a model repo."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"


def pin_main(repo: str, revision: str) -> str:
    """Make ``refs/main`` name ``revision``: write it when missing, refuse a different one."""
    ref = refs_main(repo)
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision)
        return "refs/main written"
    got = ref.read_text().strip()
    if got != revision:
        raise SystemExit(f"{repo}: refs/main is {got}, not the pinned {revision}; darts would "
                         f"load {got[:7]}. Restore the pin in {ref} before any run.")
    return "refs/main ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also TimesFM-2.5 and TiRex")
    args = ap.parse_args()
    wanted = {**CHECKPOINTS, **(EXTRA_CHECKPOINTS if args.all else {})}
    for arch, (repo, kwargs) in wanted.items():
        path = snapshot_download(repo_id=repo, **kwargs)
        note = f" ({pin_main(repo, kwargs['revision'])})" if arch in LOADED_AT_MAIN else ""
        print(f"{arch:14s} {repo:34s} -> {path}{note}")


if __name__ == "__main__":
    main()
