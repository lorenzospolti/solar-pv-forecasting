"""Turn the tidy results store into thesis tables and figures.

Reads ``results/results.parquet`` (the config-hash-keyed store the runner writes)
and emits, into ``thesis/tables/`` and ``thesis/figures/``:

  * ``master_results.{csv,md}`` — metrics aggregated over folds (and sites), per
    model x weather_mode x daytime mask, with mean±std dispersion.
  * ``skill_by_model.png``      — skill over smart persistence (the headline bar).
  * ``nrmse_by_model.png``      — capacity-normalized RMSE per model.
  * ``skill_vs_compute.png``    — skill vs wall-clock (the "at what cost?" figure).
  * ``calibration_picp80.png``  — 80%-interval coverage vs the 0.8 nominal line
                                  (only models that produced probabilistic output).

Robust to the schema growing over time: it only uses columns that are present
and silently skips a figure whose inputs are entirely missing.

Usage:
    python experiments/report.py
    python experiments/report.py --results results/results.parquet \
        --tables-dir thesis/tables --figures-dir thesis/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# metrics to aggregate (kept if present); (column, nicer label, lower_is_better)
METRICS = [
    ("skill_rmse", "Skill vs SP", False),
    ("nrmse", "nRMSE", True),
    ("nmae", "nMAE", True),
    ("rmse", "RMSE [kW]", True),
    ("mae", "MAE [kW]", True),
    ("mbe", "MBE [kW]", True),
    ("r2", "R²", False),
    ("crps", "CRPS", True),
    ("pinball", "Pinball", True),
    ("picp_80", "PICP-80", False),
    ("width_80", "Width-80", True),
    ("fit_predict_s", "Fit+pred [s]", True),
]
GROUP_CANDIDATES = ["model", "weather_mode", "daytime_only"]


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"No results at {path}. Run experiments/run.py first.")
    df = pd.read_parquet(path)
    if df.empty:
        raise SystemExit(f"Results store {path} is empty.")
    return df


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Mean (+std) of each present metric over folds/sites, grouped by what we have."""
    groups = [c for c in GROUP_CANDIDATES if c in df.columns]
    present = [m for (m, _, _) in METRICS if m in df.columns]
    agg = df.groupby(groups, dropna=False)[present].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index(), groups


def _to_md(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub pipe table (no `tabulate` dependency)."""
    cols = list(frame.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def write_tables(agg: pd.DataFrame, groups: list[str], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(tables_dir / "master_results.csv", index=False)

    # human-readable markdown: "mean±std" per metric
    show = agg.copy()
    cols = list(groups)
    for m, label, _ in METRICS:
        mean_c, std_c = f"{m}_mean", f"{m}_std"
        if mean_c in show.columns:
            show[label] = [
                "—" if pd.isna(mu) else (f"{mu:.3f}±{sd:.3f}" if pd.notna(sd) else f"{mu:.3f}")
                for mu, sd in zip(show[mean_c], show.get(std_c, [float("nan")] * len(show)))
            ]
            cols.append(label)
    md = _to_md(show[cols])
    n_folds = int(agg[[c for c in agg.columns if c.endswith("_count")][0]].max()) if any(
        c.endswith("_count") for c in agg.columns) else 0
    out = tables_dir / "master_results.md"
    out.write_text(
        "# Master results\n\n"
        f"Metrics aggregated over folds/sites (up to {n_folds} per group). "
        "Skill is over smart persistence; nRMSE/nMAE/CRPS are capacity-normalized.\n\n"
        + md + "\n"
    )
    return out


def _headline_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer the daytime-only rows for headline bars when that column exists."""
    if "daytime_only" in df.columns and df["daytime_only"].any():
        return df[df["daytime_only"] == True]  # noqa: E712
    return df


def _model_label(row, groups) -> str:
    parts = [str(row["model"])]
    if "weather_mode" in groups and pd.notna(row.get("weather_mode")):
        parts.append(str(row["weather_mode"]))
    return "\n".join(parts)


def bar_figure(agg, groups, metric, label, figures_dir, fname, nominal=None):
    mean_c = f"{metric}_mean"
    if mean_c not in agg.columns:
        return None
    data = _headline_slice(agg).dropna(subset=[mean_c])
    if data.empty:
        return None
    data = data.sort_values(mean_c)
    labels = [_model_label(r, groups) for _, r in data.iterrows()]
    yerr = data[f"{metric}_std"] if f"{metric}_std" in data.columns else None
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(data)), 4.2))
    ax.bar(range(len(data)), data[mean_c], yerr=yerr, capsize=3, color="#3b6ea5")
    if nominal is not None:
        ax.axhline(nominal, ls="--", c="crimson", lw=1, label=f"nominal {nominal}")
        ax.legend()
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(label + " by model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = figures_dir / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def scatter_skill_vs_compute(agg, groups, figures_dir):
    if not {"skill_rmse_mean", "fit_predict_s_mean"} <= set(agg.columns):
        return None
    data = _headline_slice(agg).dropna(subset=["skill_rmse_mean", "fit_predict_s_mean"])
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(data["fit_predict_s_mean"], data["skill_rmse_mean"], s=45, color="#3b6ea5")
    for _, r in data.iterrows():
        ax.annotate(_model_label(r, groups).replace("\n", " "),
                    (r["fit_predict_s_mean"], r["skill_rmse_mean"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, ls="--", c="grey", lw=1)
    if (data["fit_predict_s_mean"] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel("Fit + predict time per fold [s] (log)")
    ax.set_ylabel("Skill vs smart persistence")
    ax.set_title("Skill vs compute cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "skill_vs_compute.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "results.parquet"))
    ap.add_argument("--tables-dir", default=str(ROOT / "thesis" / "tables"))
    ap.add_argument("--figures-dir", default=str(ROOT / "thesis" / "figures"))
    args = ap.parse_args()

    df = load_results(Path(args.results))
    agg, groups = aggregate(df)
    tables_dir, figures_dir = Path(args.tables_dir), Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    written = [write_tables(agg, groups, tables_dir)]
    written.append(bar_figure(agg, groups, "skill_rmse", "Skill vs SP", figures_dir,
                              "skill_by_model.png"))
    written.append(bar_figure(agg, groups, "nrmse", "nRMSE", figures_dir, "nrmse_by_model.png"))
    written.append(bar_figure(agg, groups, "picp_80", "PICP-80", figures_dir,
                              "calibration_picp80.png", nominal=0.8))
    written.append(scatter_skill_vs_compute(agg, groups, figures_dir))

    print(f"Grouped by: {groups}")
    print(f"Models: {sorted(df['model'].unique())}")
    for p in written:
        if p is not None:
            print(f"  wrote {p}")
    skipped = [n for p, n in zip(written, ["tables", "skill", "nrmse", "picp", "scatter"])
               if p is None]
    if skipped:
        print(f"  skipped (no data): {skipped}")


if __name__ == "__main__":
    main()
