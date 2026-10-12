"""第 5 章：两两 DCC-GARCH 联动与 Forbes-Rigobon 传染检验。
Chapter 5: pairwise DCC-GARCH co-movement and Forbes-Rigobon contagion tests.

* DCC(1,1)：输入为 GARCH 全样本标准化残差，两两对齐后估计 (a, b)，比较事件前后条件相关均值，
  日历安慰剂 + 按窗口族 BH-FDR；
  DCC(1,1) on full-sample GARCH standardised residuals, aligned pairwise; compares mean conditional
  correlation before and after the event using calendar placebos and BH-FDR within window families;
* Forbes-Rigobon (2002)：原始收益上的异方差校正相关，区分“传染”与“相依”，单侧安慰剂 + BH-FDR。
  Forbes-Rigobon (2002) heteroskedasticity-adjusted correlation on raw returns, separating contagion
  from interdependence, with a one-sided placebo and BH-FDR.

对齐规则：同一宽口径区域用同步收益，跨区域用两日重叠收益（overlap2）。
Alignment rule: contemporaneous returns within a broad region, two-day overlapping returns across regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .data_alignment import DataAligner
from .stats_utils import bh_fdr, event_position, placebo_centers, positions_for_dates

AlignmentRule = Callable[[str, str], str]

DCC_COLUMNS = [
    "Pair", "Window_Type", "FDR_Family", "FDR_Family_Size", "Mean_Pre", "Mean_Post", "Diff",
    "Naive_ttest_p_uncorrected", "FDR_Adjusted_p", "Placebo_p", "Placebo_N_draws", "Placebo_Min_Resolvable_p",
    "FDR_Reject_5pct", "DCC_a", "DCC_a_SE_QMLE", "DCC_a_t_QMLE", "DCC_b", "DCC_b_SE_QMLE", "DCC_b_t_QMLE",
    "DCC_a_plus_b", "DCC_a_plus_b_SE_QMLE", "DCC_a_SE_boot", "DCC_b_SE_boot", "DCC_a_plus_b_SE_boot",
    "Correlation_Targeting", "Alignment_Mode", "Pair_N_obs", "Model_Status",
]


@dataclass(frozen=True)
class DccConfig:
    event_date: str
    # (名称, 前窗起, 前窗止, 后窗起, 后窗止) / (label, pre start, pre end, post start, post end)
    windows: tuple[tuple[str, int, int, int, int], ...] = (
        ("short", -20, -1, 0, 20),
        ("medium", -40, -1, 0, 40),
        ("long", -60, -1, 0, 60),
    )
    placebo_replications: int = 999
    bootstrap_se_reps: int = 0
    bootstrap_block: int = 20
    correlation_targeting: str = "full"  # full / pre_event
    alignment_mode: str = "region_aware"  # region_aware / contemporaneous / overlap2 / lag1
    random_seed: int = 3702014
    fdr_alpha: float = 0.05
    confounding_dates: tuple[str, ...] = ()
    confounding_block: int = 5
    placebo_band: int = 60


def _dcc_path(z, a: float, b: float, qbar: np.ndarray) -> np.ndarray:
    """双变量 DCC(1,1) 条件相关路径（标量递推）。
    Bivariate DCC(1,1) conditional-correlation path (scalar recursion).

    Q_t = (1-a-b) Q̄ + a z_{t-1} z_{t-1}' + b Q_{t-1};  ρ_t = Q12 / sqrt(Q11 Q22)
    """
    Q11, Q12, Q22 = float(qbar[0, 0]), float(qbar[0, 1]), float(qbar[1, 1])
    w = 1 - a - b
    q11, q12, q22 = Q11, Q12, Q22
    zl = z.tolist() if isinstance(z, np.ndarray) else z
    rho = np.empty(len(zl))
    prev = None
    for t, row in enumerate(zl):
        o11, o12, o22 = (Q11, Q12, Q22) if prev is None else (prev[0] * prev[0], prev[0] * prev[1], prev[1] * prev[1])
        q11 = w * Q11 + a * o11 + b * q11
        q12 = w * Q12 + a * o12 + b * q12
        q22 = w * Q22 + a * o22 + b * q22
        r = q12 / math.sqrt(q11 * q22)
        rho[t] = -0.999 if r < -0.999 else (0.999 if r > 0.999 else r)
        prev = row
    return rho


class DccFdrEngine:
    """两两 DCC(1,1) 筛查（非联合多元 DCC）/ Pairwise DCC(1,1) screening (not a joint multivariate DCC)."""

    bh_fdr = staticmethod(bh_fdr)

    def __init__(self, config: DccConfig) -> None:
        self.config = config
        placebo_seed, boot_seed = np.random.SeedSequence(config.random_seed).spawn(2)
        self._placebo_rng = np.random.default_rng(placebo_seed)
        self._boot_rng = np.random.default_rng(boot_seed)

    @staticmethod
    def _dcc_filter(z: np.ndarray, a: float, b: float, qbar: np.ndarray | None = None) -> np.ndarray:
        return _dcc_path(z, a, b, np.cov(z.T) if qbar is None else qbar)

    @classmethod
    def _dcc_negloglik(cls, theta: np.ndarray, z: np.ndarray, qbar: np.ndarray | None = None) -> float:
        """相关部分的负对数似然（省略常数项）/ Correlation-part negative log-likelihood (constants omitted)."""
        a, b = float(theta[0]), float(theta[1])
        if a <= 0 or b <= 0 or a + b >= 0.999:
            return 1.0e10
        rho = cls._dcc_filter(z, a, b, qbar)
        ll = 0.0
        log = np.log
        for t, r in enumerate(rho):
            det = max(1 - r * r, 1.0e-10)
            z0, z1 = z[t, 0], z[t, 1]
            ll += 0.5 * (log(det) + (z0**2 - 2 * r * z0 * z1 + z1**2) / det)
        return float(ll)

    @staticmethod
    def _standardize_with_fit_sample(z_raw: np.ndarray, fit_slice: slice | None) -> tuple[np.ndarray, np.ndarray]:
        """用拟合样本的均值与标准差标准化 / Standardise with fit-sample mean and SD."""
        z_fit_raw = z_raw if fit_slice is None else z_raw[fit_slice]
        mu, sd = z_fit_raw.mean(axis=0), z_fit_raw.std(axis=0, ddof=1)
        if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
            raise ValueError("DCC standardization encountered non-positive fit-sample standard deviation.")
        z = (z_raw - mu) / sd
        return z, (z if fit_slice is None else z[fit_slice])

    def _optimize(self, z_fit: np.ndarray, qbar: np.ndarray):
        return optimize.minimize(lambda th: self._dcc_negloglik(th, z_fit, qbar), x0=np.array([0.03, 0.94]), bounds=((1.0e-6, 0.4), (1.0e-6, 0.998)), method="L-BFGS-B")

    def fit_pair(self, x: pd.Series, y: pd.Series, fit_slice: slice | None = None):
        """返回 (ρ 路径, a, b, se_a, se_b, a+b, se_{a+b})；QMLE 标准误不含第一阶段不确定性。
        Returns (ρ path, a, b, se_a, se_b, a+b, se_{a+b}); QMLE SEs exclude first-stage uncertainty.
        """
        df = pd.concat([x, y], axis=1).dropna()
        if len(df) < 100:
            raise ValueError("DCC pair has fewer than 100 synchronized residual observations.")
        z, z_fit = self._standardize_with_fit_sample(df.to_numpy(dtype=float), fit_slice)
        if len(z_fit) < 100:
            raise ValueError("DCC fit sample has fewer than 100 synchronized residual observations.")
        qbar = np.cov(z_fit.T)
        res = self._optimize(z_fit, qbar)
        if not res.success:
            raise RuntimeError(res.message)
        a, b = map(float, res.x)
        try:
            h = 1.0e-5
            f = lambda th: self._dcc_negloglik(th, z_fit, qbar)  # noqa: E731
            hess = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    ei, ej = np.zeros(2), np.zeros(2)
                    ei[i], ej[j] = h, h
                    hess[i, j] = (f(res.x + ei + ej) - f(res.x + ei - ej) - f(res.x - ei + ej) + f(res.x - ei - ej)) / (4 * h * h)
            cov = np.linalg.inv(hess)
            a_se, b_se = float(np.sqrt(max(cov[0, 0], 0.0))), float(np.sqrt(max(cov[1, 1], 0.0)))
            ab_se = float(np.sqrt(max(float(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1]), 0.0)))
        except Exception:
            a_se = b_se = ab_se = float("nan")
        return self._dcc_filter(z, a, b, qbar), a, b, a_se, b_se, a + b, ab_se

    def block_bootstrap_se(self, x: pd.Series, y: pd.Series, fit_slice: slice | None = None) -> tuple[float, float, float]:
        """移动块自助法标准误（默认关闭）/ Moving-block bootstrap SEs (off by default)."""
        nan3 = (float("nan"),) * 3
        z_raw = pd.concat([x, y], axis=1).dropna().to_numpy(dtype=float)
        if len(z_raw) < 100 or self.config.bootstrap_se_reps <= 0:
            return nan3
        _z, z_fit = self._standardize_with_fit_sample(z_raw, fit_slice)
        n, block = len(z_fit), int(self.config.bootstrap_block)
        if n < 100 or block <= 1 or n <= block:
            return nan3
        n_blocks = int(np.ceil(n / block))
        draws = []
        for _ in range(self.config.bootstrap_se_reps):
            starts = self._boot_rng.integers(0, n - block + 1, size=n_blocks)
            zb = z_fit[np.concatenate([np.arange(s, s + block) for s in starts])[:n]]
            res = self._optimize(zb, np.cov(zb.T))
            if res.success:
                draws.append((float(res.x[0]), float(res.x[1]), float(res.x[0] + res.x[1])))
        if len(draws) < 30:
            return nan3
        return tuple(float(v) for v in np.asarray(draws).std(axis=0, ddof=1))  # type: ignore[return-value]

    def _placebo_pvalue(self, rho, event_center, pre_len, post_len, actual_diff, confounding_positions=()) -> tuple[float, int]:
        """双侧日历安慰剂 p 值 / Two-sided calendar-placebo p-value."""
        n = len(rho)
        cands = placebo_centers(
            n=n, event_center=event_center, lo=-pre_len, hi=post_len - 1, valid_start=pre_len, valid_stop=n - post_len,
            base_block=(-self.config.placebo_band, self.config.placebo_band + 1),
            confounding_positions=confounding_positions, confounding_block=self.config.confounding_block,
        )
        if len(cands) == 0:
            return float("nan"), 0
        centers = self._placebo_rng.choice(cands, size=min(self.config.placebo_replications, len(cands)), replace=False)
        diffs = np.array([rho[c : c + post_len].mean() - rho[c - pre_len : c].mean() for c in centers])
        return float((np.sum(np.abs(diffs) >= abs(actual_diff)) + 1) / (len(diffs) + 1)), int(len(diffs))

    @staticmethod
    def pairwise_psd_diagnostic(dcc_table: pd.DataFrame, value_col: str = "Mean_Post") -> pd.DataFrame:
        """两两相关拼成的矩阵是否半正定 / Whether the matrix assembled from pairwise correlations is PSD."""
        if dcc_table.empty or "Pair" not in dcc_table.columns:
            return pd.DataFrame(columns=["Window_Type", "N_Assets", "Min_Eigenvalue", "Is_PSD"])
        rows = []
        for window_type, sub in dcc_table.groupby("Window_Type"):
            tickers = sorted({t for p in sub["Pair"].dropna() for t in str(p).split("|") if t})
            pos = {t: i for i, t in enumerate(tickers)}
            mat = np.eye(len(tickers))
            for _i, row in sub.iterrows():
                parts = str(row["Pair"]).split("|")
                val = row.get(value_col, np.nan)
                if len(parts) == 2 and parts[0] in pos and parts[1] in pos and pd.notna(val):
                    mat[pos[parts[0]], pos[parts[1]]] = mat[pos[parts[1]], pos[parts[0]]] = float(np.clip(val, -0.999, 0.999))
            eig = np.linalg.eigvalsh(mat) if len(tickers) > 1 else np.array([1.0])
            rows.append({"Window_Type": window_type, "N_Assets": len(tickers), "Min_Eigenvalue": float(eig.min()), "Is_PSD": bool(eig.min() >= -1.0e-8)})
        return pd.DataFrame(rows)

    def run(self, residuals: pd.DataFrame, pairs: list[tuple[str, str]] | None = None, alignment_for: AlignmentRule | None = None) -> pd.DataFrame:
        """逐资产对估计并检验事件前后条件相关变化 / Estimate each pair and test the pre/post change in conditional correlation."""
        cfg = self.config
        if cfg.correlation_targeting not in {"full", "pre_event"}:
            raise ValueError("correlation_targeting must be 'full' or 'pre_event'.")
        if pairs is None:
            pairs = list(combinations(residuals.columns, 2))
        rows: list[dict[str, object]] = []
        for left, right in pairs:
            mode = (alignment_for(left, right) if alignment_for else "overlap2") if cfg.alignment_mode == "region_aware" else cfg.alignment_mode
            try:
                aligned = DataAligner.pair_align(residuals[left], residuals[right], mode=mode)
                if len(aligned) < 100:
                    raise ValueError(f"DCC pair has fewer than 100 aligned residual observations (mode={mode}).")
                ec = event_position(aligned.index, cfg.event_date)
                conf = positions_for_dates(aligned.index, cfg.confounding_dates)
                fit_slice = slice(0, ec) if cfg.correlation_targeting == "pre_event" else None
                rho, a, b, a_se, b_se, ab, ab_se = self.fit_pair(aligned["left"], aligned["right"], fit_slice=fit_slice)
                ba, bb, bab = self.block_bootstrap_se(aligned["left"], aligned["right"], fit_slice=fit_slice)
                bits = ["OK"]
                if np.isnan(a_se) or np.isnan(b_se) or np.isnan(ab_se):
                    bits.append("SE_UNAVAILABLE")
                if cfg.bootstrap_se_reps > 0 and (np.isnan(ba) or np.isnan(bb) or np.isnan(bab)):
                    bits.append("BOOTSTRAP_SE_UNAVAILABLE")
                for label, pre_l, pre_r, post_l, post_r in cfg.windows:
                    pre, post = rho[ec + pre_l : ec + pre_r + 1], rho[ec + post_l : ec + post_r + 1]
                    if len(pre) < 3 or len(post) < 3:
                        raise ValueError(f"Window {label} too short for pair {left}-{right}")
                    diff = float(post.mean() - pre.mean())
                    pp, pn = self._placebo_pvalue(rho, ec, len(pre), len(post), diff, conf)
                    rows.append(
                        {
                            "Pair": f"{left}|{right}", "Window_Type": label, "Mean_Pre": float(pre.mean()), "Mean_Post": float(post.mean()), "Diff": diff,
                            "Naive_ttest_p_uncorrected": float(stats.ttest_ind(post, pre, equal_var=False).pvalue),
                            "Placebo_p": pp, "Placebo_N_draws": pn, "Placebo_Min_Resolvable_p": 1.0 / (pn + 1) if pn > 0 else np.nan,
                            "DCC_a": a, "DCC_a_SE_QMLE": a_se, "DCC_a_t_QMLE": a / a_se if a_se > 0 else np.nan,
                            "DCC_b": b, "DCC_b_SE_QMLE": b_se, "DCC_b_t_QMLE": b / b_se if b_se > 0 else np.nan,
                            "DCC_a_plus_b": ab, "DCC_a_plus_b_SE_QMLE": ab_se, "DCC_a_SE_boot": ba, "DCC_b_SE_boot": bb, "DCC_a_plus_b_SE_boot": bab,
                            "Correlation_Targeting": cfg.correlation_targeting, "Alignment_Mode": mode, "Pair_N_obs": int(len(aligned)), "Model_Status": "|".join(bits),
                        }
                    )
            except Exception as exc:
                rows.append({"Pair": f"{left}|{right}", "Window_Type": "all", "FDR_Reject_5pct": False, "Correlation_Targeting": cfg.correlation_targeting, "Alignment_Mode": mode, "Pair_N_obs": 0, "Model_Status": f"DCC_FAILED: {exc}"})
        out = pd.DataFrame(rows).reindex(columns=DCC_COLUMNS)
        out["FDR_Family"] = out["Window_Type"]
        for _w, idx in out.groupby("Window_Type").groups.items():
            p = out.loc[idx, "Placebo_p"].to_numpy(dtype=float)
            out.loc[idx, "FDR_Adjusted_p"] = bh_fdr(p)
            out.loc[idx, "FDR_Family_Size"] = int(np.sum(~np.isnan(p)))
        out["FDR_Reject_5pct"] = out["FDR_Adjusted_p"] <= cfg.fdr_alpha
        return out


@dataclass(frozen=True)
class ContagionConfig:
    event_date: str
    windows: tuple[tuple[str, int], ...] = (("w60", 60), ("w40", 40), ("w90", 90))
    placebo_replications: int = 999
    random_seed: int = 3702014
    fdr_alpha: float = 0.05
    confounding_dates: tuple[str, ...] = ()
    confounding_block: int = 5


class ForbesRigobonEngine:
    """Forbes & Rigobon (2002)：ρ* = ρ / sqrt(1 + δ(1-ρ²))，δ 取两侧方差增幅的较大者。
    Forbes & Rigobon (2002): ρ* = ρ / sqrt(1 + δ(1-ρ²)), δ being the larger relative variance increase.
    """

    def __init__(self, config: ContagionConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    @staticmethod
    def _fr_adjust(rho_post: float, delta: float) -> float:
        denom = 1.0 + delta * (1.0 - rho_post**2)
        return rho_post / np.sqrt(denom) if denom > 0 else float("nan")

    @classmethod
    def _stat(cls, x: np.ndarray, y: np.ndarray, center: int, window: int) -> dict[str, float] | None:
        """以 center 为界的前后窗口相关及校正相关 / Pre/post correlations and adjusted correlation around center."""
        if center - window < 0 or center + window > len(x):
            return None
        xp, yp = x[center - window : center], y[center - window : center]
        xq, yq = x[center : center + window], y[center : center + window]
        if len(xp) < 10 or len(xq) < 10 or xp.std() == 0 or yp.std() == 0 or xq.std() == 0 or yq.std() == 0:
            return None
        rho_pre, rho_post = float(np.corrcoef(xp, yp)[0, 1]), float(np.corrcoef(xq, yq)[0, 1])
        vx, vy = xp.var(ddof=1), yp.var(ddof=1)
        delta = float(max(max((xq.var(ddof=1) / vx - 1.0) if vx > 0 else 0.0, (yq.var(ddof=1) / vy - 1.0) if vy > 0 else 0.0), 0.0))
        adj = cls._fr_adjust(rho_post, delta)
        return {"rho_pre": rho_pre, "rho_post_raw": rho_post, "rho_post_FRadj": adj, "delta": delta, "diff_raw": rho_post - rho_pre, "diff_FRadj": adj - rho_pre if np.isfinite(adj) else float("nan")}

    def _placebo_p(self, x, y, event_center, window, actual, confounding_positions) -> tuple[float, int]:
        """单侧安慰剂 p 值：H1 为校正相关上升（传染）/ One-sided placebo p-value; H1 = adjusted correlation rises (contagion)."""
        n = len(x)
        cands = placebo_centers(
            n=n, event_center=event_center, lo=-window, hi=window - 1, valid_start=window, valid_stop=n - window,
            base_block=(-window - 5, window + 5), confounding_positions=confounding_positions, confounding_block=self.config.confounding_block,
        )
        if len(cands) == 0 or not np.isfinite(actual):
            return float("nan"), 0
        centers = self._rng.choice(cands, size=min(self.config.placebo_replications, len(cands)), replace=False)
        diffs = np.array([s["diff_FRadj"] for s in (self._stat(x, y, int(c), window) for c in centers) if s is not None and np.isfinite(s["diff_FRadj"])])
        if len(diffs) < 30:
            return float("nan"), int(len(diffs))
        return float((np.sum(diffs >= actual) + 1) / (len(diffs) + 1)), int(len(diffs))

    def run(self, returns: pd.DataFrame, pairs: list[tuple[str, str]], alignment_for: AlignmentRule | None = None) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for left, right in pairs:
            try:
                mode = alignment_for(left, right) if alignment_for else "contemporaneous"
                aligned = DataAligner.pair_align(returns[left], returns[right], mode=mode)
                x, y = aligned["left"].to_numpy(dtype=float), aligned["right"].to_numpy(dtype=float)
                ec = event_position(aligned.index, self.config.event_date)
                conf = positions_for_dates(aligned.index, self.config.confounding_dates)
                for label, window in self.config.windows:
                    st = self._stat(x, y, ec, window)
                    if st is None:
                        raise ValueError(f"window {label} too short")
                    pp, draws = self._placebo_p(x, y, ec, window, st["diff_FRadj"], conf)
                    rows.append({"Pair": f"{left}|{right}", "Window": label, "Alignment_Mode": mode, "N": len(aligned), **st, "Contagion_Placebo_p": pp, "Placebo_N_draws": draws,
                                 "Interpretation": "contagion" if np.isfinite(pp) and pp < 0.05 else "interdependence_or_none", "Model_Status": "OK"})
            except Exception as exc:
                rows.append({"Pair": f"{left}|{right}", "Window": "all", "Model_Status": f"FR_FAILED: {exc}"})
        out = pd.DataFrame(rows)
        if "Contagion_Placebo_p" in out.columns:
            out["FDR_Adjusted_p"] = np.nan
            for _w, idx in out.groupby("Window").groups.items():
                out.loc[idx, "FDR_Adjusted_p"] = bh_fdr(out.loc[idx, "Contagion_Placebo_p"].to_numpy(dtype=float))
            out["FDR_Reject_contagion_5pct"] = out["FDR_Adjusted_p"] <= self.config.fdr_alpha
        return out
