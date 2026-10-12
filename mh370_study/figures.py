"""论文图形：图 3-1 数据覆盖热力图、图 4-1 CAAR 路径、DCC 路径、VaR 边界与覆盖率、功效森林图。
Figures: data-coverage heatmap (Fig. 3-1), CAAR paths (Fig. 4-1), DCC path, VaR boundaries and
coverage, and the power forest plot.
"""

from __future__ import annotations

import itertools
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .data_alignment import DataAligner  # noqa: E402
from .config import EVENT_DATE, MH17_DATE  # noqa: E402
from .contagion_tests import DccConfig, DccFdrEngine  # noqa: E402
from .garch_var import GarchVarEngine  # noqa: E402

AP = {"Australia", "China", "Hong Kong", "Japan", "Malaysia", "New Zealand", "Singapore", "South Korea", "Thailand"}
EU = {"Austria", "France", "Germany", "Germany/United Kingdom", "Ireland", "Norway", "Spain", "Switzerland", "Turkey", "United Kingdom"}


def coverage_heatmap(companies_dir: Path, out_png: Path) -> None:
    """图 3-1：公司 × 交易日可得性热力图，展示跨市场非同步交易。
    Figure 3-1: firm × trading-day availability heatmap illustrating non-synchronous trading.
    """

    def macro(region: str) -> str:
        # 三大市场区域 / three macro regions
        return "1_Asia-Pacific" if region in AP else ("2_Europe" if region in EU else "3_North America")

    rows = []
    for f in sorted(companies_dir.glob("*.csv")):
        d = pd.read_csv(f, usecols=["date", "ticker", "country_or_region"], parse_dates=["date"])
        region = d["country_or_region"].iloc[0]
        rows.append((macro(region), region, d["ticker"].iloc[0], set(d["date"])))
    all_dates = pd.DatetimeIndex(sorted(set().union(*[r[3] for r in rows])))
    pos = {dt: i for i, dt in enumerate(all_dates)}
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    M = np.zeros((len(rows), len(all_dates)), dtype=np.int8)
    for i, (mname, _r, _t, dates) in enumerate(rows):
        M[i, [pos[dt] for dt in dates if dt in pos]] = int(mname[0])

    def nearest(t):
        return int(np.argmin(np.abs(all_dates - pd.Timestamp(t))))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(M, aspect="auto", cmap=ListedColormap(["white", "#2c7fb8", "#d95f0e", "#31a354"]), vmin=0, vmax=3, interpolation="nearest")
    ax.axvline(nearest(EVENT_DATE), color="red", lw=1.3)
    ax.axvline(nearest(MH17_DATE), color="black", lw=1.0, ls="--")
    ax.text(nearest(EVENT_DATE) - 7, 1.5, "MH370", color="red", rotation=90, va="top", ha="right", fontsize=8)
    ax.text(nearest(MH17_DATE) + 7, 1.5, "MH17", color="black", rotation=90, va="top", ha="left", fontsize=8)
    years = pd.date_range(all_dates.min().normalize().replace(month=1, day=1), all_dates.max(), freq="YS")
    ax.set_xticks([nearest(y) for y in years])
    ax.set_xticklabels([y.year for y in years])
    start, blocks = 0, []
    for mname, grp in itertools.groupby(rows, key=lambda r: r[0]):
        n = len(list(grp))
        blocks.append((start, start + n, mname))
        start += n
    for s, _e, _m in blocks:
        if s > 0:
            ax.axhline(s - 0.5, color="black", lw=0.8)
    ax.set_yticks([(s + e) / 2 - 0.5 for s, e, _ in blocks])
    ax.set_yticklabels([m.split("_", 1)[1] for _, _, m in blocks], rotation=90, va="center")
    ax.set_ylabel(f"Firms (grouped by market region, n={len(rows)})")
    ax.set_xlabel("Trading calendar (2012-2016)")
    ax.set_title("Data coverage: firm x trading-day availability", pad=12)
    ax.legend(
        handles=[Patch(facecolor="#2c7fb8", label="Asia-Pacific"), Patch(facecolor="#d95f0e", label="Europe"), Patch(facecolor="#31a354", label="North America"), Patch(facecolor="white", edgecolor="gray", label="No trading / missing")],
        loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def caar_paths(caar_overall: pd.DataFrame, caar_groups: pd.DataFrame, out_png: Path) -> None:
    """图 4-1：事件时间 CAAR 与 95% 置信带，叠加高/低暴露组。
    Figure 4-1: event-time CAAR with a 95% band, plus high/low exposure groups.
    """
    if caar_overall.empty:
        return
    tau, mean, se = caar_overall["tau"], caar_overall["CAAR"], caar_overall["SE_CAAR"]
    plt.figure(figsize=(11, 5))
    plt.plot(tau, mean, color="black", linewidth=1.4, label=f"All firms (N={int(caar_overall['N'].iloc[0])})")
    plt.fill_between(tau, mean - 1.96 * se, mean + 1.96 * se, color="black", alpha=0.15, label="95% band")
    for group, color in (("treated_high_exposure", "firebrick"), ("control_lower_exposure", "steelblue")):
        sub = caar_groups[caar_groups["group"] == group] if not caar_groups.empty else pd.DataFrame()
        if len(sub):
            plt.plot(sub["tau"], sub["CAAR"], color=color, linewidth=1.1, label=f"{group} (N={int(sub['N'].iloc[0])})")
    plt.axvline(0, color="red", linestyle="--", linewidth=0.8, label="MH370")
    plt.axhline(0, color="gray", linewidth=0.6)
    plt.title("Event-time cumulative average abnormal return (CAAR)")
    plt.xlabel("Event time tau (trading days)")
    plt.ylabel("CAAR")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def var_boundary(asset_returns: pd.DataFrame, out_png: Path, engine: GarchVarEngine, ticker: str = "AIRPORT.KL") -> None:
    """图 7-1：已实现收益与 GARCH 样本外 VaR 边界（复用 GARCH 缓存）。
    Figure 7-1: realised returns versus GARCH out-of-sample VaR boundaries (reuses the GARCH cache).
    """
    t = ticker if ticker in asset_returns.columns else asset_returns.columns[0]
    x = asset_returns[t].dropna().astype(float)
    selected, _c, _r = engine.fit_asset(t, x)
    sigma_in = pd.Series(np.asarray(selected["res"].conditional_volatility, dtype=float) / 100.0, index=x.index)
    paths, _resid, oos_idx, _ = engine._rolling_oos_forecasts(x, (0.05, 0.01))
    v95, v99 = pd.Series(paths[0.05], index=oos_idx), pd.Series(paths[0.01], index=oos_idx)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x.index, x, label=f"{t} log return", linewidth=0.6, color="steelblue", alpha=0.7)
    ax.plot(x.index, -1.645 * sigma_in, label="In-sample 95% conditional band (GARCH)", linewidth=0.8, color="gray", alpha=0.55)
    ax.plot(v95.index, -v95, label="OOS 95% VaR (GARCH)", linewidth=1.1, color="darkorange")
    ax.plot(v99.index, -v99, label="OOS 99% VaR (GARCH)", linewidth=1.1, color="firebrick", linestyle="--")
    aligned = x.loc[oos_idx]
    br = aligned[aligned < -v95.reindex(aligned.index)]
    ax.scatter(br.index, br.values, color="darkorange", s=12, zorder=5, label=f"95% VaR breaches (n={len(br)})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Realized returns versus GARCH(1,1) OOS VaR boundaries\n(OOS expanding-window, burn-in = {engine.config.oos_burn_in} obs.)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily log return")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def var_coverage(table_b: pd.DataFrame, out_png: Path) -> None:
    """图 7-2：多水平 VaR 实际与名义突破率 / Figure 7-2: observed vs nominal exceedance rates."""
    data = table_b.dropna(subset=["OOS_T", "Failures"])
    levels = sorted(data["VaR_Level"].dropna().unique())
    if not levels:
        return
    expected = [1.0 - lv for lv in levels]
    observed = [data[data["VaR_Level"] == lv]["Failures"].sum() / data[data["VaR_Level"] == lv]["OOS_T"].sum() for lv in levels]
    x, w = np.arange(len(levels)), 0.38
    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, expected, w, color="gray", label="Expected (nominal tail)")
    plt.bar(x + w / 2, observed, w, color="firebrick", label="Observed (pooled)")
    plt.xticks(x, [f"VaR {lv:g}" for lv in levels])
    plt.ylabel("Exceedance rate")
    plt.title("Multi-level VaR coverage: observed vs expected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def power_forest(power: pd.DataFrame, out_png: Path) -> None:
    """图 8-1：MDE 与合理效应量对照 / Figure 8-1: MDE versus plausible effect sizes."""
    data = power.dropna(subset=["MDE_80pct", "plausible"]).head(12)
    if data.empty:
        return
    y = np.arange(len(data))
    plt.figure(figsize=(10, 6))
    for i, (m, p) in enumerate(zip(data["MDE_80pct"], data["plausible"])):
        plt.plot([min(m, p), max(m, p)], [i, i], color="gray", linewidth=0.8, zorder=1)
    plt.scatter(data["MDE_80pct"], y, color="firebrick", zorder=3, label="MDE@80%")
    plt.scatter(data["plausible"], y, color="steelblue", marker="s", zorder=3, label="Plausible effect")
    plt.yticks(y, [str(t)[:34] for t in data["target"]], fontsize=7)
    plt.xlabel("Effect size")
    plt.title("Detectability: MDE@80% vs plausible effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def dcc_path(residuals: pd.DataFrame, out_png: Path, event_date: str, pair: tuple[str, str] | None = None, alignment_for=None) -> None:
    """图 5-1：核心资产对的 DCC 条件相关路径（默认取第一个可拟合的资产对）。
    Figure 5-1: DCC conditional-correlation path of a core pair (first fittable pair by default).
    """
    if residuals.shape[1] < 2:
        return
    dcc = DccFdrEngine(DccConfig(event_date=event_date))
    rule = alignment_for or (lambda a, b: "overlap2")
    candidates = [pair] if pair else list(combinations(residuals.columns, 2))
    for p in candidates:
        try:
            aligned = DataAligner.pair_align(residuals[p[0]], residuals[p[1]], mode=rule(p[0], p[1]))
            if len(aligned) < 100:
                continue
            rho = dcc.fit_pair(aligned["left"], aligned["right"])[0]
        except Exception:
            continue
        plt.figure(figsize=(11, 5))
        plt.plot(aligned.index, rho, label=f"{p[0]}-{p[1]} DCC correlation", linewidth=1.0)
        plt.axvline(pd.Timestamp(event_date), color="red", linestyle="--", label="MH370 event")
        plt.title("Pairwise DCC conditional correlation")
        plt.xlabel("Date")
        plt.ylabel("Conditional correlation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return
