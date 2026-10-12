"""第 7 章：GARCH(1,1) 分布选择与多水平 VaR 回测。
Chapter 7: GARCH(1,1) distribution selection and multi-level VaR backtesting.

* 正态 / Student-t / 偏 t 三种新息，按惩罚 BIC 选择（α+β≥0.99 或 β>0.99 加大罚项）；
  Normal / Student-t / skewed-t innovations chosen by penalised BIC (large penalty if α+β≥0.99 or β>0.99);
* 扩展窗口样本外：前 252 日预热，每 21 日重估，只使用重估点之前的数据；
  Expanding-window out-of-sample: 252-day burn-in, refit every 21 days using only past data;
* Kupiec 无条件覆盖、Christoffersen 条件覆盖、Kratz-Lok-McNeil 多项式多水平检验 + Nass 校正。
  Kupiec unconditional coverage, Christoffersen conditional coverage, Kratz-Lok-McNeil
  multinomial multi-level test with the Nass correction.

全样本拟合与各重估点选中的模型按序列内容缓存，表格、多水平回测与作图共用一次计算。
Full-sample fits and per-refit selected models are cached by series content and shared by
the tables, the multi-level backtest and the figures.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

warnings.filterwarnings("ignore", message="The optimizer returned code .*", category=ConvergenceWarning)


@dataclass(frozen=True)
class GarchVarConfig:
    distributions: tuple[str, ...] = ("normal", "studentst", "skewt")
    var_levels: tuple[float, ...] = (0.95, 0.975, 0.99, 0.995)
    persistence_penalty: float = 1.0e9
    persistence_alpha_beta_cutoff: float = 0.99
    beta_cutoff: float = 0.99
    residual_arch_lags: int = 5
    oos_burn_in: int = 252
    oos_refit_frequency: int = 21


def _series_key(x: pd.Series) -> str:
    """按数值与日期生成缓存键 / Cache key from values and dates."""
    h = hashlib.sha1(np.ascontiguousarray(x.to_numpy(dtype=float)).tobytes())
    h.update(np.asarray(x.index.astype("int64") if isinstance(x.index, pd.DatetimeIndex) else np.arange(len(x))).tobytes())
    return h.hexdigest()


class GarchVarEngine:
    """GARCH(1,1) 拟合、分布选择与 VaR 回测引擎 / GARCH(1,1) fitting, distribution selection and VaR backtesting."""

    def __init__(self, config: GarchVarConfig | None = None) -> None:
        self.config = config or GarchVarConfig()
        self.convergence_diagnostics_ = pd.DataFrame()
        self.oos_refit_diagnostics_ = pd.DataFrame()
        self._fit_cache: dict[str, tuple] = {}
        self._refit_cache: dict[str, tuple[list, list]] = {}

    # -------------------------------------------------- 分布 / distributions
    @staticmethod
    def _arch_dist_name(dist: str) -> str:
        name = dist.lower()
        if name in {"normal", "gaussian"}:
            return "normal"
        if name in {"studentst", "student-t", "t"}:
            return "t"
        if name in {"skewt", "skewed student-t", "skewedstudent"}:
            return "skewt"
        raise ValueError(f"Unsupported distribution: {dist}")

    @staticmethod
    def _display_dist(dist: str) -> str:
        return {"normal": "Normal", "studentst": "studentst", "skewt": "skewt"}.get(dist.lower(), dist)

    # ------------------------------------------------------ 回测 / backtests
    @staticmethod
    def _lr_uc(failures: int, n: int, alpha: float) -> float:
        if failures == 0:
            return float(-2.0 * (n * np.log(1.0 - alpha)))
        if failures == n:
            return float(-2.0 * (n * np.log(alpha)))
        phat = failures / n
        ll_null = failures * np.log(alpha) + (n - failures) * np.log(1 - alpha)
        ll_alt = failures * np.log(phat) + (n - failures) * np.log(1 - phat)
        return float(-2 * (ll_null - ll_alt))

    @classmethod
    def kupiec_test(cls, returns: np.ndarray, var_series: np.ndarray, alpha: float) -> tuple[int, float, float]:
        """Kupiec (1995) 无条件覆盖 LR；零突破时使用有限边界值。
        Kupiec (1995) unconditional-coverage LR; a finite boundary value is used for zero failures.
        """
        hits = returns < -var_series
        failures, n = int(hits.sum()), int(len(hits))
        if n == 0:
            return failures, float("nan"), float("nan")
        lr = cls._lr_uc(failures, n, alpha)
        return failures, lr, float(stats.chi2.sf(lr, 1))

    @classmethod
    def christoffersen_cc_test(cls, returns: np.ndarray, var_series: np.ndarray, alpha: float) -> tuple[float, float, float, float, float]:
        """Christoffersen (1998) 条件覆盖分解，返回 (LR_uc, p_uc, LR_ind, LR_cc, p_cc)。
        Christoffersen (1998) conditional-coverage decomposition returning (LR_uc, p_uc, LR_ind, LR_cc, p_cc).
        """
        hits = (returns < -var_series).astype(int)
        n = len(hits)
        if n < 2:
            return (float("nan"),) * 5
        h0, h1 = hits[:-1], hits[1:]
        n00 = int(np.sum((h0 == 0) & (h1 == 0)))
        n01 = int(np.sum((h0 == 0) & (h1 == 1)))
        n10 = int(np.sum((h0 == 1) & (h1 == 0)))
        n11 = int(np.sum((h0 == 1) & (h1 == 1)))
        pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
        pi01 = n01 / max(n00 + n01, 1)
        pi11 = n11 / max(n10 + n11, 1)
        eps = 1.0e-12

        def log(v):
            return np.log(np.clip(v, eps, 1 - eps))

        ll_ind = (n00 + n10) * log(1 - pi) + (n01 + n11) * log(pi)
        ll_dep = n00 * log(1 - pi01) + n01 * log(pi01) + n10 * log(1 - pi11) + n11 * log(pi11)
        lr_ind = -2 * (ll_ind - ll_dep)
        lr_uc = cls._lr_uc(int(hits.sum()), n, alpha)
        lr_cc = lr_uc + lr_ind
        return float(lr_uc), float(stats.chi2.sf(lr_uc, 1)), float(lr_ind), float(lr_cc), float(stats.chi2.sf(lr_cc, 2))

    @staticmethod
    def multinomial_var_backtest(returns: np.ndarray, var_by_level: dict[float, np.ndarray]) -> dict[str, object]:
        """Kratz, Lok & McNeil (2018) 多项式多水平 VaR 检验 + Nass (1959) 小样本校正。
        Kratz, Lok & McNeil (2018) multinomial multi-level VaR test with the Nass (1959) small-sample correction.
        """
        levels = sorted(var_by_level)
        tail = [1.0 - lv for lv in levels]
        n, k = int(len(returns)), len(levels)
        n_buckets = k + 1
        if n < 20 or k < 2:
            return {"stat": np.nan, "df": k, "p_pearson": np.nan, "p_nass": np.nan, "nass_c": np.nan, "nass_df": np.nan, "n": n, "observed": [], "expected": []}
        breaches = np.zeros(n, dtype=int)
        for lv in levels:
            breaches += (returns < -np.asarray(var_by_level[lv], dtype=float)).astype(int)
        observed = np.array([float((breaches == b).sum()) for b in range(n_buckets)])
        probs = np.asarray([1.0 - tail[0]] + [tail[i] - tail[i + 1] for i in range(k - 1)] + [tail[-1]], dtype=float)
        expected = n * probs
        statistic = float(np.nansum(np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)))
        ex2 = n_buckets - 1
        vx2 = 2 * (n_buckets - 1) + (float(np.sum(1.0 / probs)) - n_buckets**2 - 2 * n_buckets + 2) / n
        nass_c = (2 * ex2 / vx2) if vx2 > 0 else 1.0
        nass_df = nass_c * ex2
        return {
            "stat": statistic,
            "df": k,
            "p_pearson": float(stats.chi2.sf(statistic, k)),
            "p_nass": float(stats.chi2.sf(nass_c * statistic, nass_df)) if nass_df > 0 else np.nan,
            "nass_c": float(nass_c),
            "nass_df": float(nass_df),
            "n": n,
            "observed": observed.astype(int).tolist(),
            "expected": expected.round(2).tolist(),
        }

    # -------------------------------------------------------- 拟合 / fitting
    def _fit_one(self, returns_decimal: pd.Series, dist: str) -> dict[str, object]:
        """以百分比收益拟合一个 GARCH(1,1) / Fit one GARCH(1,1) on percentage returns."""
        scaled = returns_decimal.dropna().astype(float) * 100.0
        model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist=self._arch_dist_name(dist), rescale=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})
        p = res.params
        a1, b1 = float(p.get("alpha[1]", np.nan)), float(p.get("beta[1]", np.nan))
        boundary = bool(a1 + b1 >= self.config.persistence_alpha_beta_cutoff or b1 > self.config.beta_cutoff)
        std_resid = pd.Series(res.std_resid, index=scaled.index).replace([np.inf, -np.inf], np.nan).dropna()
        lags = self.config.residual_arch_lags
        arch_p = float(het_arch(std_resid, nlags=lags)[1]) if len(std_resid) > lags + 5 else float("nan")
        return {
            "dist": dist,
            "res": res,
            "omega": float(p.get("omega", np.nan)),
            "alpha_1": a1,
            "beta_1": b1,
            "alpha+beta": a1 + b1,
            "bic": float(res.bic),
            "penalized_bic": float(res.bic + (self.config.persistence_penalty if boundary else 0.0)),
            "boundary": boundary,
            "Residual_ARCH_p": arch_p,
            "std_resid": std_resid,
        }

    def fit_asset(self, ticker: str, returns_decimal: pd.Series) -> tuple[dict[str, object], list[dict[str, object]], pd.Series]:
        """全样本拟合三种分布并按惩罚 BIC 选择（缓存）/ Full-sample fit of all distributions, selected by penalised BIC (cached)."""
        key = _series_key(returns_decimal.dropna())
        if key not in self._fit_cache:
            candidates, failures = [], []
            for dist in self.config.distributions:
                try:
                    candidates.append(self._fit_one(returns_decimal, dist))
                except Exception as exc:
                    failures.append({"Ticker": ticker, "Dist": dist, "Model_Status": "NON_CONVERGENCE", "Reason": str(exc)})
            if not candidates:
                raise RuntimeError(f"All GARCH distributions failed for {ticker}: {failures}")
            selected = min(candidates, key=lambda c: float(c["penalized_bic"]))
            self._fit_cache[key] = (selected, candidates + failures)
        selected, rows = self._fit_cache[key]
        return selected, [dict(r) if "res" not in r else r for r in rows], selected["std_resid"]  # type: ignore[index]

    @staticmethod
    def _distribution_quantile(selected: dict[str, object], alpha: float) -> float:
        """拟合分布的标准化新息分位数 / Standardised innovation quantile of the fitted distribution."""
        res = selected["res"]
        dist = res.model.distribution
        shape = [float(res.params[name]) for name in dist.parameter_names()]
        return float(dist.ppf(alpha, shape if shape else None))

    def _refits(self, x: pd.Series) -> tuple[list[tuple[int, int, dict]], list[dict[str, object]]]:
        """每个重估点选中的模型（缓存）/ Selected model at each refit point (cached)."""
        key = _series_key(x)
        if key in self._refit_cache:
            return self._refit_cache[key]
        burn, freq = self.config.oos_burn_in, self.config.oos_refit_frequency
        xr = x.reset_index(drop=True)
        n = len(xr)
        points = list(range(burn, n, freq))
        fits, rows = [], []
        for i, t_refit in enumerate(points):
            t_next = points[i + 1] if i + 1 < len(points) else n
            candidates = []
            for dist in self.config.distributions:
                try:
                    candidates.append(self._fit_one(xr.iloc[:t_refit].astype(float), dist))
                except Exception as exc:
                    rows.append({"refit_point": t_refit, "dist": dist, "status": "NON_CONVERGENCE", "reason": str(exc)})
            if not candidates:
                rows.append({"refit_point": t_refit, "dist": "ALL", "status": "RISK_MODEL_FAILED", "reason": "all distributions failed at refit"})
                continue
            sel = min(candidates, key=lambda c: float(c["penalized_bic"]))
            fits.append((t_refit, t_next, sel))
            rows.append(
                {
                    "refit_point": t_refit,
                    "forecast_start": t_refit,
                    "forecast_end": t_next - 1,
                    "dist": sel["dist"],
                    "alpha_plus_beta": sel["alpha+beta"],
                    "penalized_bic": sel["penalized_bic"],
                    "status": "OK" if not bool(sel["boundary"]) else "PARAMETER_BOUNDARY_WARNING",
                    "reason": "",
                }
            )
        self._refit_cache[key] = (fits, rows)
        return fits, rows

    def _rolling_oos_forecasts(self, returns_decimal: pd.Series, alpha_tails: tuple[float, ...]):
        """扩展窗口样本外 VaR 路径与标准化残差。
        Expanding-window out-of-sample VaR paths and standardised residuals.
        Returns (var_paths, std_resid_oos, oos_index, refit_rows).
        """
        burn, freq = self.config.oos_burn_in, self.config.oos_refit_frequency
        x_original = returns_decimal.dropna().astype(float)
        n = len(x_original)
        if n <= burn + freq:
            raise ValueError(f"Series too short for OOS backtest: n={n}, burn={burn}. Need at least {burn + freq + 1} observations.")
        fits, refit_rows = self._refits(x_original)
        xv = x_original.to_numpy()
        var_paths = {a: np.full(n - burn, np.nan) for a in alpha_tails}
        std_resid_oos = np.full(n - burn, np.nan)
        for t_refit, t_next, sel in fits:
            res = sel["res"]
            mu_pct = float(res.params.get("mu", 0.0))
            mu_dec = mu_pct / 100.0
            omega, a1, b1 = float(sel["omega"]), float(sel["alpha_1"]), float(sel["beta_1"])
            q = {a: self._distribution_quantile(sel, a) for a in alpha_tails}
            h = float(np.asarray(res.forecast(horizon=1, reindex=False).variance)[-1, 0])
            for step in range(t_next - t_refit):
                i = t_refit + step
                if step > 0:
                    eps = float(xv[i - 1]) * 100.0 - mu_pct
                    h = omega + a1 * eps**2 + b1 * h
                sigma = np.sqrt(max(h, 1.0e-12)) / 100.0
                pos = i - burn
                std_resid_oos[pos] = (float(xv[i]) - mu_dec) / sigma if sigma > 0 else np.nan
                for a, qa in q.items():
                    var_paths[a][pos] = -(mu_dec + sigma * qa)
        return var_paths, std_resid_oos, x_original.index[burn:], [dict(r) for r in refit_rows]

    def _rolling_oos_var(self, returns_decimal: pd.Series, alpha_tail: float):
        """单一尾部概率的样本外 VaR / Out-of-sample VaR for a single tail probability."""
        paths, resid, idx, _ = self._rolling_oos_forecasts(returns_decimal, (alpha_tail,))
        return paths[alpha_tail], resid, idx

    # ---------------------------------------------------------- 表格 / tables
    def multilevel_backtest(self, returns: pd.DataFrame) -> pd.DataFrame:
        """表 7-3：逐资产样本外多水平多项式回测 / Table 7-3: per-asset out-of-sample multinomial backtest."""
        levels = tuple(self.config.var_levels)
        tails = tuple(1.0 - lv for lv in levels)
        rows = []
        for ticker in returns.columns:
            x = returns[ticker].dropna().astype(float)
            try:
                paths, _r, oos_idx, _ = self._rolling_oos_forecasts(x, tails)
                x_oos = x.loc[oos_idx].to_numpy(dtype=float)
                valid = ~np.isnan(x_oos)
                for a in tails:
                    valid &= ~np.isnan(paths[a])
                bt = self.multinomial_var_backtest(x_oos[valid], {lv: paths[1.0 - lv][valid] for lv in levels})
                rows.append(
                    {
                        "Ticker": ticker,
                        "OOS_T": int(valid.sum()),
                        "Multinomial_chi2": bt["stat"],
                        "df": bt["df"],
                        "Multinomial_p_pearson": bt["p_pearson"],
                        "Multinomial_p_nass": bt["p_nass"],
                        "Nass_c": bt["nass_c"],
                        "Nass_df": bt["nass_df"],
                        "Observed_buckets": bt["observed"],
                        "Expected_buckets": bt["expected"],
                        "Model_Status": "OK" if bt["p_nass"] >= 0.05 else "OK|MULTILEVEL_COVERAGE_WARNING",
                    }
                )
            except Exception as exc:
                rows.append({"Ticker": ticker, "Model_Status": f"MULTILEVEL_FAILED: {exc}"})
        return pd.DataFrame(rows)

    def run(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """表 7-1 / 7-2：参数与分布选择、逐水平覆盖回测；并返回全样本标准化残差（供 DCC 使用）。
        Tables 7-1 / 7-2: parameters, distribution choice and per-level coverage backtests;
        also returns full-sample standardised residuals (input to DCC).
        """
        table_a, table_b, conv, refit_all = [], [], [], []
        resid: dict[str, pd.Series] = {}
        tails = tuple(1.0 - lv for lv in self.config.var_levels)
        for ticker in returns.columns:
            x = returns[ticker].dropna().astype(float)
            try:
                selected, candidates, _ = self.fit_asset(ticker, x)
                resid[ticker] = pd.Series(selected["std_resid"], name=ticker)
                for c in candidates:
                    if "res" in c:
                        conv.append(
                            {
                                "Ticker": ticker,
                                "Dist": c["dist"],
                                "Converged": True,
                                "BIC": c["bic"],
                                "Penalized_BIC": c["penalized_bic"],
                                "Boundary_Penalized": c["boundary"],
                                "Model_Status": "OK" if not bool(c["boundary"]) else "PARAMETER_BOUNDARY_WARNING",
                                "Reason": "",
                            }
                        )
                    else:
                        conv.append(c)
                paths, _r, oos_idx, refit_rows = self._rolling_oos_forecasts(x, tails)
                for r in refit_rows:
                    r["Ticker"] = ticker
                refit_all.extend(refit_rows)
                x_oos = x.loc[oos_idx].to_numpy(dtype=float)
                b_rows = []
                for level in self.config.var_levels:
                    a = 1.0 - level
                    v = paths[a]
                    valid = ~np.isnan(v) & ~np.isnan(x_oos)
                    failures, _lr, kup_p = self.kupiec_test(x_oos[valid], v[valid], a)
                    cc_p = self.christoffersen_cc_test(x_oos[valid], v[valid], a)[4]
                    bits = ["OK"]
                    if bool(selected["boundary"]):
                        bits.append("PARAMETER_BOUNDARY_WARNING")
                    if int(valid.sum()) == 0:
                        bits.append("OOS_FORECAST_FAILED")
                    if kup_p < 0.05:
                        bits.append("VAR_COVERAGE_WARNING")
                    if cc_p < 0.05:
                        bits.append("VAR_CONDITIONAL_COVERAGE_WARNING")
                    b_rows.append(
                        {
                            "Ticker": ticker,
                            "VaR_Level": level,
                            "OOS_T": int(valid.sum()),
                            "Failures": failures,
                            "Expected_Failures": round(a * int(valid.sum()), 1),
                            "Kupiec_p": kup_p,
                            "Christoffersen_cc_p": cc_p,
                            "Model_Status": "|".join(bits),
                        }
                    )
                table_a.append(
                    {
                        "Ticker": ticker,
                        "Best_Dist": self._display_dist(str(selected["dist"])),
                        "omega": selected["omega"],
                        "alpha_1": selected["alpha_1"],
                        "beta_1": selected["beta_1"],
                        "alpha+beta": selected["alpha+beta"],
                        "Residual_ARCH_p": selected["Residual_ARCH_p"],
                        "BIC": selected["bic"],
                        "Penalized_BIC": selected["penalized_bic"],
                        "Model_Status": "PARAMETER_BOUNDARY_WARNING" if bool(selected["boundary"]) else "OK",
                    }
                )
                table_b.extend(b_rows)
            except Exception as exc:
                conv.append({"Ticker": ticker, "Dist": "ALL", "Converged": False, "Model_Status": "RISK_MODEL_FAILED", "Reason": str(exc)})
                table_a.append({"Ticker": ticker, "Best_Dist": "NONE", "Model_Status": "RISK_MODEL_FAILED", "Reason": str(exc)})
                for level in self.config.var_levels:
                    table_b.append({"Ticker": ticker, "VaR_Level": level, "OOS_T": np.nan, "Failures": np.nan, "Expected_Failures": np.nan, "Kupiec_p": np.nan, "Christoffersen_cc_p": np.nan, "Model_Status": "RISK_MODEL_FAILED"})
        self.convergence_diagnostics_ = pd.DataFrame(conv)
        self.oos_refit_diagnostics_ = pd.DataFrame(refit_all)
        panel = pd.concat(resid.values(), axis=1) if resid else pd.DataFrame()
        return pd.DataFrame(table_a), pd.DataFrame(table_b), panel
