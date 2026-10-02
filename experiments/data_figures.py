"""Figures for the Data chapter: thesis/figures/data_*.{pdf,png}.

  data_coverage  share of daytime hours with missing power, per panel station and
                 local day (2021-2023), with the nine test windows, the first test
                 day and a marker on each day with outage hours (pvfc.qc) that the
                 ETL set missing
  data_clearsky  one week of measured power against the clear-sky template P^cs,
                 and the clear-sky index k, for LSK North (5-11 Jul 2022, the
                 first week of the first test window)
  data_k         distribution of daytime k per panel station: the
                 cloud-enhancement tail above 1 and the values clipped at 1.3
  data_weather   monthly midday irradiance from the campus tower and from the
                 Open-Meteo archive, and the panel's apparent performance ratio
                 (mean capacity factor per kW/m2 of each irradiance series)

Inputs (read-only): the processed station frames (``pvfc.data.load_station``,
weather mode "none", cached under data/processed/ for the current ETL_VERSION), the
raw site-level PV files (only to locate the outage hours), the raw 1-min tower
irradiance files and the cached Open-Meteo archive
(``pvfc.weather.fetch_open_meteo``). The panel and the fold calendar come from
configs/experiment.yaml. Writes only thesis/figures/data_*.

Usage:
    python experiments/data_figures.py                   # all four figures
    python experiments/data_figures.py --only k,weather  # a subset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from pvfc import data as D  # noqa: E402
from pvfc import weather as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "thesis" / "figures"
CONFIG = ROOT / "configs" / "experiment.yaml"
IRRADIANCE_DIR = (D.RAW_DIR / "Dataset" / "Time series dataset" / "Meteorological dataset"
                  / "Irradiance")
TZ = D.HKUST_TZ

# ---------------------------------------------------------------- style
FULL_W = 6.3                      # text width in inches
INK, INK2 = "#0b0b0b", "#52514e"  # primary / secondary text
GRID, AXIS = "#e6e5e0", "#c3c2b7"
SURFACE = "#ffffff"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", facecolor=SURFACE)  # exact size, no crop
    plt.close(fig)
    print(f"wrote {FIG_DIR / name}.pdf/.png")


# ---------------------------------------------------------------- inputs
def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def _panel() -> list[str]:
    return list(_config()["data"]["panel"])


def _test_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first, last) local test day of each fold, as the calendar-anchored splitter
    lays them out: windows start every ``step_days`` from ``first_test_day``."""
    sp = _config()["splits"]
    first = pd.Timestamp(sp["first_test_day"])
    out = []
    for i in range(int(sp["max_folds"])):
        start = first + pd.Timedelta(days=i * int(sp["step_days"]))
        out.append((start, start + pd.Timedelta(days=int(sp["test_days"]) - 1)))
    return out


def _frame(site: str) -> tuple[pd.DataFrame, float]:
    """Processed frame (weather mode "none") in local time, and the nameplate capacity."""
    s = D.load_station(site, weather_mode="none")
    return s.frame.tz_convert(TZ), s.capacity


def _label(site: str) -> str:
    """'<station> · <native resolution>', from the QC log the ETL writes."""
    minutes = int(pd.read_csv(D.QC_LOG).set_index("site_id").loc[site, "resolution_min"])
    return f"{site} · {'1 h' if minutes >= 60 else f'{minutes} min'}"


def _outage_days(site: str, frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Local days (naive midnights) holding outage hours that the ETL set missing.

    Replays the ETL steps that precede the outage rule (``pvfc.data.load_station``:
    site-level power from ``_read_power``, meter-error QC, hourly means, night QC,
    short-gap bridging)
    and returns the days of the hours that have a value there but are missing in the
    processed ``frame`` (local time): only the outage rule (``pvfc.qc``) removes
    values after the bridging step.
    """
    raw, _ = D._read_power(D.list_stations()[site])
    cap = D.station_capacity(site)
    raw, _ = D._qc_power(raw, cap)
    hourly, _ = D._qc_night(raw.resample("1h").mean(), cap)
    hourly, _ = D._bridge_short_gaps(hourly)
    before = hourly.tz_convert(TZ).reindex(frame.index)
    flagged = before.notna() & frame["power"].isna()
    return pd.DatetimeIndex(frame.index[flagged].normalize().tz_localize(None).unique())


# ---------------------------------------------------------------- (a) coverage
def fig_coverage() -> None:
    panel = _panel()
    days = pd.date_range("2021-03-01", "2023-12-31", freq="D")
    grid = np.full((len(panel), len(days)), np.nan)
    labels, outages = [], []
    for r, site in enumerate(panel):
        f, _ = _frame(site)
        day = f["is_day"].astype(bool)
        g = pd.DataFrame({"day": day, "miss": day & f["power"].isna()}).groupby(
            f.index.normalize().tz_localize(None))
        n_day = g["day"].sum()
        share = (g["miss"].sum() / n_day.where(n_day > 0)).reindex(days)
        grid[r] = share.to_numpy()
        labels.append(_label(site))
        outages.append(_outage_days(site, f))

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cmap.set_bad((0, 0, 0, 0))
    x = mdates.date2num(pd.date_range(days[0], periods=len(days) + 1, freq="D").to_pydatetime())

    fig = plt.figure(figsize=(FULL_W, 2.6))
    ax_t = fig.add_axes([0.175, 0.825, 0.775, 0.065])   # test-window track
    ax = fig.add_axes([0.175, 0.30, 0.775, 0.52], sharex=ax_t)
    cax = fig.add_axes([0.66, 0.115, 0.25, 0.035])

    n = len(panel)
    pad = 0.14                                   # white gap between station rows
    for r in range(n):
        mesh = ax.pcolormesh(x, [r + pad, r + 1 - pad], np.ma.masked_invalid(grid[r:r + 1]),
                             cmap=cmap, vmin=0, vmax=1, shading="flat", linewidth=0)
        # record extent: hairline outline, so "no record" reads as empty
        valid = np.where(np.isfinite(grid[r]))[0]
        ax.add_patch(Rectangle((x[valid[0]], r + pad), x[valid[-1] + 1] - x[valid[0]],
                               1 - 2 * pad, fill=False, edgecolor=AXIS, linewidth=0.5))
        # outage days: a small ink triangle in the gap above the strip, at mid-day
        if len(outages[r]):
            xo = mdates.date2num((outages[r] + pd.Timedelta(hours=12)).to_pydatetime())
            ax.plot(xo, np.full(len(xo), r + 0.5 * pad), linestyle="none", marker="v",
                    markersize=3.2, color=INK, markeredgewidth=0, zorder=6, clip_on=False)
    ax.set_ylim(n, 0)
    ax.set_yticks(np.arange(n) + 0.5, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    windows = _test_windows()
    for a, b in windows:
        x0 = mdates.date2num(a.to_pydatetime())
        x1 = mdates.date2num((b + pd.Timedelta(days=1)).to_pydatetime())
        ax_t.add_patch(Rectangle((x0, 0.15), x1 - x0, 0.7, color=CAT[1], linewidth=0))
        # light band behind the station rows, so each window lines up with the strips
        ax.axvspan(x0, x1, color=CAT[1], alpha=0.12, linewidth=0, zorder=0)
    ax_t.set_ylim(0, 1)
    ax_t.set_yticks([0.5], ["test windows"])
    ax_t.tick_params(axis="both", which="both", length=0)
    ax_t.tick_params(axis="x", labelbottom=False)
    ax_t.grid(False)
    for side in ("left", "bottom"):
        ax_t.spines[side].set_visible(False)

    first = mdates.date2num(windows[0][0].to_pydatetime())
    for a_ in (ax, ax_t):
        a_.axvline(first, color=INK, linewidth=0.8, zorder=5)
    ax_t.annotate("first test day, 5 Jul 2022", xy=(first, 1.0), xytext=(-4, 4),
                  textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK2,
                  annotation_clip=False)

    ax.set_xlim(x[0], x[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", which="minor", length=2, color=AXIS)

    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    cax.set_title("daytime hours missing, per day", fontsize=7.5, color=INK2, pad=3)
    handles = [Patch(facecolor=SURFACE, edgecolor=AXIS, linewidth=0.5, label="no record"),
               Patch(facecolor=CAT[1], label="test window (30 days)"),
               Line2D([], [], linestyle="none", marker="v", markersize=4, color=INK,
                      markeredgewidth=0, label="day with outage hours set missing")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.165, 0.0), ncol=1,
               handlelength=1.4, labelspacing=0.35)
    _save(fig, "data_coverage")


# ---------------------------------------------------------------- (b) clear-sky week
CLEARSKY_SITE = "LSK North"
CLEARSKY_WEEK = ("2022-07-05", "2022-07-11")   # first week of the first test window


def fig_clearsky() -> None:
    f, cap = _frame(CLEARSKY_SITE)
    w = f.loc[CLEARSKY_WEEK[0]:CLEARSKY_WEEK[1]]
    t = w.index.tz_localize(None) + pd.Timedelta(minutes=30)  # plot at interval midpoints
    k_day = w["k"].where(w["is_day"].astype(bool))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL_W, 3.0), sharex=True, layout="constrained",
                                   gridspec_kw={"height_ratios": [1.7, 1]})
    ax1.fill_between(t, 0, w["p_clear"], color=INK2, alpha=0.10, linewidth=0)
    ax1.plot(t, w["p_clear"], color=INK2, linewidth=1.0,
             label=r"clear-sky template $P^{\mathrm{cs}}_t$")
    ax1.plot(t, w["power"], color=CAT[0], linewidth=1.4, label=r"measured power $P_t$")
    ax1.axhline(cap, color=AXIS, linewidth=0.6)
    ax1.annotate(f"nameplate C = {cap:.0f} kW", xy=(t[0], cap), xytext=(2, 2),
                 textcoords="offset points", fontsize=7, color=INK2, va="bottom")
    ax1.set_ylabel("power (kW)")
    ax1.set_ylim(0, cap * 1.12)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.6,
               borderaxespad=0.2)

    ax2.plot(t, k_day, color=CAT[0], linewidth=1.2)
    clipped = (k_day >= 1.3).to_numpy()
    ax2.plot(t[clipped], k_day[clipped], linestyle="none", marker="o", markersize=3.5,
             markerfacecolor=CAT[1], markeredgecolor=SURFACE, markeredgewidth=0.6,
             label="clipped at 1.3")
    for level, text in ((1.0, "$k_t = 1$"), (1.3, "$k_t = 1.3$")):
        ax2.axhline(level, color=INK2, linewidth=0.6)
        ax2.annotate(text, xy=(1.0, level), xycoords=("axes fraction", "data"), xytext=(3, 0),
                     textcoords="offset points", fontsize=7, color=INK2, va="center")
    ax2.set_ylabel("clear-sky index $k_t$")
    ax2.set_ylim(0, 1.45)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.legend(loc="lower right", handlelength=1.0)
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=12))
    end = pd.Timestamp(CLEARSKY_WEEK[1]) + pd.Timedelta(days=1)
    for a in (ax1, ax2):
        a.set_xlim(pd.Timestamp(CLEARSKY_WEEK[0]), end)
        a.grid(axis="x", which="major", color=GRID)
    _save(fig, "data_clearsky")


# ---------------------------------------------------------------- (c) k distribution
def fig_k() -> None:
    panel = _panel()
    edges = np.arange(0.0, 1.3 + 1e-9, 0.05)
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.1), sharex=True, sharey=True,
                             layout="constrained")
    for ax, site in zip(axes.ravel(), panel):
        f, _ = _frame(site)
        day = f["is_day"].astype(bool)
        n_all = int(day.sum())                       # denominator as in the QC log
        k = f.loc[day, "k"]
        k = k[np.isfinite(k)]
        at_clip = k >= 1.3 - 1e-9
        hist, _ = np.histogram(k[~at_clip], bins=edges)
        ax.bar(edges[:-1], 100 * hist / n_all, width=0.05, align="edge", color=CAT[0],
               edgecolor=SURFACE, linewidth=0.3, label="daytime hours")
        ax.bar(1.32, 100 * at_clip.sum() / n_all, width=0.035, align="edge", color=CAT[1],
               linewidth=0, label="clipped at 1.3")
        ax.axvline(1.0, color=INK2, linewidth=0.6)
        share_gt1 = 100 * float((k > 1.0).sum()) / n_all
        share_clip = 100 * float(at_clip.sum()) / n_all
        ax.text(0.03, 0.96, f"$k_t > 1$: {share_gt1:.1f}%\nat 1.3: {share_clip:.1f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK2,
                linespacing=1.3)
        ax.set_title(site, loc="left", color=INK, pad=3)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("daytime clear-sky index $k_t$")
        ax.set_xticks([0, 0.5, 1.0, 1.3], ["0", "0.5", "1", "1.3"])
    for ax in axes[:, 0]:
        ax.set_ylabel("share of daytime\nhours (%)")
    axes[0, 0].set_xlim(0, 1.38)
    axes[0, 0].set_ylim(0, 13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=2, handlelength=1.2)
    _save(fig, "data_k")


# ---------------------------------------------------------------- (d) weather
def _tower_hourly() -> pd.Series:
    parts = []
    for year in (2021, 2022, 2023):
        df = D._read_ts_csv(IRRADIANCE_DIR / f"Irradiance_{year}.csv")
        parts.append(pd.to_numeric(df.iloc[:, 0], errors="coerce"))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.resample("1h").mean()           # interval-start hourly means, as the ETL


def fig_weather() -> None:
    tower = _tower_hourly().rename("tower")
    arch = W.fetch_open_meteo(D.HKUST_LAT, D.HKUST_LON, "2021-01-01", "2023-12-31")["ghi"]
    both = pd.concat([tower, arch.rename("archive")], axis=1).tz_convert(TZ)
    caps = [_frame(s) for s in _panel()]
    # panel capacity factor only where all six stations report (no partial panel)
    cf = pd.concat([f["power"] / c for f, c in caps], axis=1).mean(axis=1, skipna=False)
    both["cf"] = cf.reindex(both.index)
    mid = both[both.index.hour.isin([10, 11, 12, 13])]           # 10:00-14:00 local
    month = mid.index.tz_localize(None).to_period("M")
    m = mid.groupby(month)[["tower", "archive"]].mean()
    t = m.index.to_timestamp() + pd.Timedelta(days=14)
    # apparent performance ratio (P/C) / (GHI / 1 kW/m2), on hours where all inputs exist
    ok = mid.dropna(subset=["tower", "archive", "cf"])
    g = ok.groupby(ok.index.tz_localize(None).to_period("M"))
    pr = pd.DataFrame({"tower": g["cf"].mean() / (g["tower"].mean() / 1000),
                       "archive": g["cf"].mean() / (g["archive"].mean() / 1000)})
    pr = pr[g.size() >= 60]                  # at least ~half a month of midday hours
    t_pr = pr.index.to_timestamp() + pd.Timedelta(days=14)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL_W, 3.2), sharex=True, layout="constrained")
    style = {"linewidth": 1.3, "marker": "o", "markersize": 3.6, "markeredgecolor": SURFACE,
             "markeredgewidth": 0.6}
    ax1.plot(t, m["tower"], color=CAT[0], label="campus tower (measured)", **style)
    ax1.plot(t, m["archive"], color=CAT[1], label="Open-Meteo archive (forecast)", **style)
    ax1.set_ylabel("midday GHI (W/m²)")
    ax1.set_ylim(0, 950)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.6,
               borderaxespad=0.2)
    ax2.plot(t_pr, pr["tower"], color=CAT[0], **style)
    ax2.plot(t_pr, pr["archive"], color=CAT[1], **style)
    ax2.set_ylabel("apparent PR")
    ax2.set_ylim(0, 1.4)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 10)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"))
    for a in (ax1, ax2):
        a.axvline(pd.Timestamp("2023-01-01"), color=AXIS, linewidth=0.6, zorder=1)
    _save(fig, "data_weather")


FIGURES = {"coverage": fig_coverage, "clearsky": fig_clearsky, "k": fig_k,
           "weather": fig_weather}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--only", default=",".join(FIGURES),
                    help=f"comma-separated subset of {','.join(FIGURES)}")
    args = ap.parse_args()
    for name in [s.strip() for s in args.only.split(",") if s.strip()]:
        FIGURES[name]()


if __name__ == "__main__":
    main()
