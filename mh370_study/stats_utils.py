"""跨模块共享的统计工具。
Statistical utilities shared across modules.

包括 BH-FDR、事件日定位、区域与暴露等级映射、安慰剂候选中心、置换检验与
Kolari-Pynnönen 截面相关膨胀因子。
Includes BH-FDR, event-date location, region and exposure-rank mappings,
placebo candidate centres, permutation tests and the Kolari-Pynnönen
cross-sectional-correlation inflation factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# 航空公司暴露等级（非航空渠道记为 0）
# Airline exposure ranks (non-airline channels are coded 0)
EXPOSURE_RANK = {
    "direct_malaysia_china_exposure": 3.0,
    "asia_pacific_airline_exposure": 2.0,
    "regional_airline_exposure": 1.0,
    "global_airline_exposure": 0.0,
}
# 处理组：直接暴露与亚太航空 / Treated group: direct and Asia-Pacific airline exposure
TREATED_GROUPS = ("direct_malaysia_china_exposure", "asia_pacific_airline_exposure")

_AMERICAS = {"United States", "Canada", "Mexico/United States ADR", "Panama/United States"}
_OCEANIA = {"Australia", "New Zealand"}
_ASIA = {"China", "Hong Kong", "Japan", "South Korea", "Malaysia", "Thailand", "Singapore"}

# 80% 功效、5% 双侧检验下的 MDE 乘数 ≈ 2.80
# MDE multiplier for 80% power and a 5% two-sided test ≈ 2.80
Z_80_POWER = float(stats.norm.ppf(0.975) + stats.norm.ppf(0.80))


def region_of(country: object) -> str:
    """宽口径区域：Americas / Oceania / Asia / Europe（其余国家含土耳其归入 Europe）。
    Broad region: Americas / Oceania / Asia / Europe (all others, incl. Turkey, map to Europe).
    """
    c = str(country)
    if c in _AMERICAS:
        return "Americas"
    if c in _OCEANIA:
        return "Oceania"
    if c in _ASIA:
        return "Asia"
    return "Europe"


def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg 调整 p 值；NaN 保持 NaN。
    Benjamini-Hochberg adjusted p-values; NaNs are preserved.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    if valid.size == 0:
        return q
    pv = p[valid]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    q[valid] = adjusted
    return q


def event_position(index: pd.Index, event_date) -> int:
    """事件日在交易日索引中的位置；非交易日取其后第一个交易日。
    Position of the event date in a trading-day index; a non-trading day maps to the next trading day.
    """
    pos = int(np.searchsorted(np.asarray(index), np.datetime64(pd.Timestamp(event_date)), side="left"))
    if pos >= len(index):
        raise ValueError(f"Event date {event_date} is outside the return index.")
    return pos


def positions_for_dates(index: pd.Index, dates) -> tuple[int, ...]:
    """一组日期在索引中的位置 / Positions of several dates in the index."""
    arr = np.asarray(index)
    out = []
    for d in dates:
        pos = int(np.searchsorted(arr, np.datetime64(pd.Timestamp(d)), side="left"))
        if pos < len(index):
            out.append(pos)
    return tuple(out)


def placebo_centers(
    *,
    n: int,
    event_center: int,
    lo: int,
    hi: int,
    valid_start: int,
    valid_stop: int,
    base_block: tuple[int, int],
    confounding_positions: tuple[int, ...],
    confounding_block: int,
) -> np.ndarray:
    """可用作安慰剂“伪事件中心”的位置（升序）。
    Positions usable as placebo pseudo-event centres (ascending).

    中心 c 的检验窗口为 [c+lo, c+hi]。以下中心被排除：
    The test window of centre c is [c+lo, c+hi]. A centre is excluded if
      1) 落在事件日屏蔽带 [event+base_block[0], event+base_block[1]) 内；
         it lies in the event band [event+base_block[0], event+base_block[1]);
      2) 落在混淆日期 ±k 内；
         it lies within ±k of a confounding date;
      3) 其窗口与真实事件窗口 [event+lo, event+hi] 重叠；
         its window overlaps the true event window [event+lo, event+hi];
      4) 其窗口覆盖任一混淆日期 ±k。
         its window covers any confounding date ±k.
    """
    c = np.arange(max(valid_start, 0), min(valid_stop, n))
    c = c[(c + lo >= 0) & (c + hi < n)]
    keep = ~((c >= max(0, event_center + base_block[0])) & (c < min(n, event_center + base_block[1])))
    k = int(confounding_block)
    keep &= np.abs(c - event_center) > (hi - lo)
    for pos in confounding_positions:
        keep &= ~((c >= max(0, pos - k)) & (c < min(n, pos + k + 1)))
        keep &= ~((c + lo <= pos + k) & (c + hi >= pos - k))
    return c[keep]


def permutation_mean_diff(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """两样本均值差的双侧置换检验 p 值 (count+1)/(n_perm+1)。
    Two-sided permutation p-value for a difference in means, (count+1)/(n_perm+1).
    """
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b]).astype(float)
    na = len(a)
    threshold = abs(observed) - 1.0e-15
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        count += abs(pooled[:na].mean() - pooled[na:].mean()) >= threshold
    return float((count + 1) / (n_perm + 1))


def kp_inflation(n: int, rbar: float) -> float:
    """Kolari-Pynnönen 方差膨胀因子 sqrt((1+(n-1)r̄)/(1-r̄))。
    Kolari-Pynnönen variance-inflation factor sqrt((1+(n-1)r̄)/(1-r̄)).
    """
    if not np.isfinite(rbar) or n < 2:
        return float("nan")
    num, den = 1.0 + (n - 1) * rbar, 1.0 - rbar
    return float(np.sqrt(num / den)) if num > 0 and den > 0 else float("nan")
