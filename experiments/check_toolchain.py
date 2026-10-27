"""Week-1 toolchain smoke test (plan §Verify step 1).

Verifies the environment is sane on Apple Silicon: imports the stack, reports
versions, checks the torch device (MPS with CPU fallback), runs a tiny pvlib
clear-sky call, and a one-line LightGBM fit. Exits non-zero on any hard failure.

Run:  python experiments/check_toolchain.py
"""
from __future__ import annotations

import sys


def check() -> int:
    ok = True
    print("== pvfc toolchain check ==")

    # --- core imports + versions ---
    from pvfc.utils import env_fingerprint, get_device, set_seed

    set_seed(0)
    for name, ver in env_fingerprint().items():
        flag = "  " if ver != "missing" else "??"
        print(f"{flag} {name:12s} {ver}")

    # --- torch device (MPS gotchas) ---
    try:
        import torch

        dev = get_device("mps")
        built = torch.backends.mps.is_built()
        avail = torch.backends.mps.is_available()
        print(f"   torch device -> {dev}  (mps built={built}, available={avail})")
        x = torch.randn(8, 8, device=dev) @ torch.randn(8, 8, device=dev)
        print(f"   matmul on {dev} OK, sum={float(x.sum()):.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"!! torch/MPS check failed: {exc}")
        ok = False

    # --- pvlib clear-sky ---
    try:
        import pandas as pd

        from pvfc import clearsky as cs

        idx = pd.date_range("2022-06-21", periods=24, freq="1h", tz="UTC")
        ghi = cs.clearsky_ghi(idx, 22.337, 114.263, 60.0)
        print(f"   pvlib clear-sky GHI OK, midday peak={ghi.max():.0f} W/m^2")
    except Exception as exc:  # noqa: BLE001
        print(f"!! pvlib check failed: {exc}")
        ok = False

    # --- lightgbm one-liner ---
    try:
        import lightgbm as lgb
        import numpy as np

        X = np.random.rand(200, 4)
        y = X @ np.array([1, 2, 3, 4]) + np.random.rand(200)
        m = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
        print(f"   lightgbm fit OK, train R2={m.score(X, y):.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"!! lightgbm check failed: {exc}")
        ok = False

    # --- darts (optional at week 1) ---
    try:
        import darts

        print(f"   darts {darts.__version__} import OK")
    except Exception as exc:  # noqa: BLE001
        print(f"   (darts not yet installed: {exc})")

    print("== RESULT:", "PASS ==" if ok else "FAIL ==")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check())
