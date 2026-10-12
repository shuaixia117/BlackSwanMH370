"""第 4、6 章：市场模型事件研究、截面推断、机制回归、AVR 与 DiD。
Chapters 4 and 6: market-model event study, cross-sectional inference,
mechanism regressions, abnormal volatility ratio (AVR) and DiD.

推断层次 / Inference layers
* 公司层面：Corrado 秩检验（诊断用）、iid 残差自助法（次要）、日历安慰剂（主推断）+ 按窗口 BH-FDR；
  Firm level: Corrado rank test (diagnostic), iid residual bootstrap (secondary),
  calendar placebo (primary) with BH-FDR within each window family;
* 聚合层面：Plain-CS t、Patell、BMP、Kolari-Pynnönen ADJ-BMP（主推断）。
  Aggregate level: plain cross-sectional t, Patell, BMP and Kolari-Pynnönen ADJ-BMP (primary).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS, add_constant

from .stats_utils import (
    EXPOSURE_RANK,
    TREATED_GROUPS,
    bh_fdr,
    event_position,
    permutation_mean_diff,
    placebo_centers,
    positions_for_dates,
    region_of,
)

# 检验窗口（交易日，相对事件日）/ Test windows (trading days relative to the event)
DEFAULT_TEST_WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("short", -1, 1),
    ("sym3", -3, 3),
    ("sym5", -5, 5),
    ("med", -5, 10),
    ("long", -5, 30),
    ("post20", 1, 20),
    ("post60", 1, 60),
    ("post120", 1, 120),
)


@dataclass(frozen=True)
class EventStudyConfig:
    event_date: str
    estimation_window: tuple[int, int] = (-210, -30)
    test_windows: tuple[tuple[str, int, int], ...] = DEFAULT_TEST_WINDOWS
    bootstrap_replications: int = 1999
    placebo_replications: int = 999
    random_seed: int = 3702014
    fdr_alpha: float = 0.05
    confounding_dates: tuple[str, ...] = ()
    confounding_block: int = 5
    placebo_band: int = 40  # 事件日 ±40 日不作为安慰剂中心 / ±40 days around the event are never placebo centres


class EventStudyEngine:
    """在每只资产与其本地基准的共同交易日上做 AR/CAR 推断。
    AR/CAR inference on the common trading days of each asset and its local benchmark.
    """

    _region_of = staticmethod(region_of)
    bh_fdr = staticmethod(bh_fdr)

    def __init__(self, config: EventStudyConfig) -> None:
        self.config = config
        # 安慰剂与自助法使用相互独立的随机数流 / Independent random streams for placebo and bootstrap
        placebo_seed, boot_seed = np.random.SeedSequence(config.random_seed).spawn(2)
        self._placebo_rng = np.random.default_rng(placebo_seed)
        self._boot_rng = np.random.default_rng(boot_seed)
        self.est_std_residuals_: dict[str, pd.Series] = {}

    # ------------------------------------------------------------ 辅助 / helpers
    @staticmethod
    def _event_index(index: pd.Index, event_date: str) -> int:
        return event_position(index, event_date)

    @staticmethod
    def _fit_market_model(asset: pd.Series, market: pd.Series) -> tuple[float, float]:
        """OLS 市场模型 R_i = α + β R_m / OLS market model R_i = α + β R_m."""
        res = OLS(asset.to_numpy(dtype=float), add_constant(market.to_numpy(dtype=float))).fit()
        return float(res.params[0]), float(res.params[1])

    @staticmethod
    def _pair_frame(asset_returns, market_returns, benchmark_map, ticker) -> tuple[pd.DataFrame, str]:
        """资产与其基准在共同交易日上对齐 / Align an asset with its benchmark on common trading days."""
        asset = asset_returns[ticker].dropna()
        if isinstance(market_returns, pd.Series):
            market, name = market_returns.dropna(), market_returns.name or "market"
        else:
            if benchmark_map is None or ticker not in benchmark_map:
                raise ValueError(f"Missing benchmark mapping for {ticker}")
            name = benchmark_map[ticker]
            if name not in market_returns.columns:
                raise ValueError(f"Benchmark {name} not found for {ticker}")
            market = market_returns[name].dropna()
        return pd.concat([asset.rename("a"), market.rename("m")], axis=1).dropna(), str(name)

    @staticmethod
    def _check_bounds(center: int, n: int, windows) -> None:
        """窗口越界即报错，避免静默截断 / Raise on out-of-sample windows instead of silently truncating."""
        for left, right in windows:
            if center + left < 0 or center + right >= n:
                raise ValueError(f"window [{left},{right}] outside sample")

    @staticmethod
    def _corrado_pvalue(ar_full: pd.Series, event_positions: np.ndarray, est_start: int, est_end: int) -> float:
        """单资产 Corrado (1989) 秩检验 / Single-asset Corrado (1989) rank test."""
        est_window = ar_full.iloc[est_start : est_end + 1]
        n_est = len(est_window)
        combined = pd.concat([est_window, ar_full.iloc[event_positions]])
        total = len(combined)
        expected = (total + 1) / 2.0
        rank_std = np.sqrt((total**2 - 1) / 12.0)
        event_ranks = combined.rank(method="average").iloc[n_est:].to_numpy()
        z = ((event_ranks.mean() - expected) / rank_std) * np.sqrt(len(event_positions))
        return float(2 * stats.norm.sf(abs(z)))

    def _rbar_from(self, tickers) -> float:
        """估计窗标准化残差的平均两两相关 r̄ / Mean pairwise correlation r̄ of estimation-window residuals."""
        series = [self.est_std_residuals_[str(t)] for t in tickers if str(t) in self.est_std_residuals_]
        if len(series) < 2:
            return float("nan")
        corr = pd.concat(series, axis=1).corr()
        if corr.shape[0] < 2:
            return float("nan")
        off = corr.to_numpy()[~np.eye(corr.shape[0], dtype=bool)]
        off = off[~np.isnan(off)]
        return float(off.mean()) if off.size else float("nan")

    def _bootstrap_pvalue(self, residuals: pd.Series, actual_car: float, window_length: int) -> float:
        """iid 残差自助法 p 值（矩阵一次性抽样）/ iid residual bootstrap p-value (single matrix draw)."""
        r = residuals.dropna().to_numpy(dtype=float)
        if len(r) < window_length:
            return float("nan")
        draws = self._boot_rng.choice(r, size=(self.config.bootstrap_replications, window_length), replace=True).sum(axis=1)
        return float((np.sum(np.abs(draws) >= abs(actual_car)) + 1) / (len(draws) + 1))

    def _placebo_pvalue(self, ar_full: pd.Series, event_center: int, left: int, right: int, actual_car: float, confounding_positions=()) -> tuple[float, int]:
        """日历安慰剂 p 值：伪事件窗口不与真实事件窗口或混淆日期重叠。
        Calendar-placebo p-value: pseudo-event windows never overlap the true event window or confounding dates.
        """
        vals = ar_full.to_numpy(dtype=float)
        n = len(vals)
        candidates = placebo_centers(
            n=n, event_center=event_center, lo=left, hi=right, valid_start=abs(left), valid_stop=n - abs(right),
            base_block=(-self.config.placebo_band, self.config.placebo_band + 1),
            confounding_positions=confounding_positions, confounding_block=self.config.confounding_block,
        )
        if len(candidates) == 0:
            return float("nan"), 0
        centers = self._placebo_rng.choice(candidates, size=min(self.config.placebo_replications, len(candidates)), replace=False)
        placebo = np.array([vals[c + left : c + right + 1].sum() for c in centers])
        return float((np.sum(np.abs(placebo) >= abs(actual_car)) + 1) / (len(placebo) + 1)), int(len(placebo))

    # ---------------------------------------------------- 主估计 / main estimation
    def run(self, asset_returns: pd.DataFrame, market_returns, benchmark_map: dict[str, str] | None = None) -> pd.DataFrame:
        """逐资产 AR/CAR；估计窗、事件中心、秩、自助法与安慰剂均在资产-基准两两日历上计算。
        Per-asset AR/CAR; estimation window, event centre, ranks, bootstrap and placebo all use the pairwise calendar.
        """
        self.est_std_residuals_ = {}
        est_left, est_right = self.config.estimation_window
        rows: list[dict[str, object]] = []
        for ticker in asset_returns.columns:
            try:
                aligned, market_name = self._pair_frame(asset_returns, market_returns, benchmark_map, ticker)
                if aligned.empty:
                    raise ValueError(f"No common asset-benchmark trading dates for {ticker}")
                idx = aligned.index
                ec = self._event_index(idx, self.config.event_date)
                conf_pos = positions_for_dates(idx, self.config.confounding_dates)
                est_start, est_end = ec + est_left, ec + est_right
                if est_start < 0 or est_end >= len(idx):
                    raise ValueError(f"Estimation window outside sample for {ticker}")
                a, m = aligned["a"], aligned["m"]
                alpha, beta = self._fit_market_model(a.iloc[est_start : est_end + 1], m.iloc[est_start : est_end + 1])
                mkt_est = m.iloc[est_start : est_end + 1].to_numpy(dtype=float)
                resid_est = a.iloc[est_start : est_end + 1].to_numpy(dtype=float) - (alpha + beta * mkt_est)
                l_est = len(mkt_est)
                resid_var = float((resid_est**2).sum() / (l_est - 2))
                resid_sd = float(np.sqrt(resid_var))
                std_resid = resid_est / resid_sd if resid_sd > 0 else np.full_like(resid_est, np.nan)
                self.est_std_residuals_[ticker] = pd.Series(std_resid, index=idx[est_start : est_end + 1], name=ticker)
                mkt_bar = float(mkt_est.mean())
                ss_m = float(((mkt_est - mkt_bar) ** 2).sum())
                ar_full = a - (alpha + beta * m)
                residuals = ar_full.iloc[est_start : est_end + 1]
                for label, left, right in self.config.test_windows:
                    positions = np.arange(ec + left, ec + right + 1)
                    if positions.min() < 0 or positions.max() >= len(idx):
                        raise ValueError(f"Test window {label} outside sample for {ticker}")
                    car = float(ar_full.iloc[positions].sum())
                    mkt_win = m.iloc[positions].to_numpy(dtype=float)
                    # 含预测误差修正的 CAR 方差（BMP 标准化）/ CAR variance with prediction-error correction (BMP)
                    car_var = resid_var * float(np.sum(1.0 + 1.0 / l_est + ((mkt_win - mkt_bar) ** 2 / ss_m if ss_m > 0 else 0.0)))
                    scar = car / np.sqrt(car_var) if car_var > 0 else float("nan")
                    pp, pn = self._placebo_pvalue(ar_full, ec, left, right, car, conf_pos)
                    rows.append(
                        {
                            "Ticker": ticker, "Benchmark": market_name, "Window_Type": label, "Window_Left": left, "Window_Right": right,
                            "Actual_CAR": car, "Alpha": alpha, "Beta": beta, "SCAR": scar, "L_est": l_est, "Resid_Var": resid_var,
                            "Corrado_p": self._corrado_pvalue(ar_full, positions, est_start, est_end),
                            "Corrado_Inference_Scope": "per_asset_not_pooled_naively",
                            "Bootstrap_p": self._bootstrap_pvalue(residuals, car, len(positions)),
                            "Bootstrap_Inference_Note": "iid_residual_bootstrap_secondary_placebo_primary",
                            "Placebo_p": pp, "Placebo_N_draws": pn,
                            "Placebo_Min_Resolvable_p": 1.0 / (pn + 1) if pn > 0 else np.nan,
                            "Model_Status": "OK",
                        }
                    )
            except Exception as exc:
                for label, left, right in self.config.test_windows:
                    rows.append(
                        {
                            "Ticker": ticker, "Benchmark": "", "Window_Type": label, "Window_Left": left, "Window_Right": right,
                            "Actual_CAR": np.nan, "SCAR": np.nan, "Corrado_p": np.nan,
                            "Corrado_Inference_Scope": "per_asset_not_pooled_naively", "Bootstrap_p": np.nan,
                            "Bootstrap_Inference_Note": "iid_residual_bootstrap_secondary_placebo_primary",
                            "Placebo_p": np.nan, "Placebo_N_draws": np.nan, "Placebo_Min_Resolvable_p": np.nan,
                            "Model_Status": f"EVENT_STUDY_FAILED: {exc}",
                        }
                    )
        out = pd.DataFrame(rows)
        out["FDR_Adjusted_p"] = np.nan
        for _w, idx in out.groupby("Window_Type").groups.items():
            out.loc[idx, "FDR_Adjusted_p"] = bh_fdr(out.loc[idx, "Placebo_p"].to_numpy(dtype=float))
        out["FDR_Reject_5pct"] = out["FDR_Adjusted_p"] <= self.config.fdr_alpha
        return out

    # ------------------------------------------------ 聚合推断 / aggregate inference
    def cross_sectional_inference(self, event_results: pd.DataFrame, estimation_window_length: int) -> pd.DataFrame:
        """表 4-1：Plain-CS、Patell (1976)、BMP (1991)、ADJ-BMP (Kolari & Pynnönen 2010)。
        Table 4-1: plain CS, Patell (1976), BMP (1991) and ADJ-BMP (Kolari & Pynnönen 2010).

        SE_CAR_plain = sd(CAR)/√N；SE_CAR_KP = SE_CAR_plain × KP 膨胀因子（CAR 尺度，用于 MDE）。
        SE_CAR_plain = sd(CAR)/√N; SE_CAR_KP = SE_CAR_plain × KP inflation factor (CAR scale, used for MDE).
        """
        nan = float("nan")
        keys = ["Plain_CS_t", "Plain_CS_p", "BMP_t", "BMP_p", "ADJ_BMP_t", "ADJ_BMP_p", "Patell_z", "Patell_p", "Rbar_Estimation_Residual_Correlation", "SE_CAR_plain", "SE_CAR_KP"]
        rows = []
        for window_type, grp in event_results.groupby("Window_Type"):
            valid = grp[grp["Model_Status"].astype(str).eq("OK")].copy() if "Model_Status" in grp.columns else grp.copy()
            cars = valid["Actual_CAR"].dropna().to_numpy(dtype=float)
            scars = valid["SCAR"].dropna().to_numpy(dtype=float) if "SCAR" in valid.columns else np.array([])
            n = len(cars)
            row: dict[str, object] = {"Window_Type": window_type, "N": n, "Mean_CAR": float(cars.mean()) if n else nan}
            if n < 3:
                row.update({k: nan for k in keys})
                rows.append(row)
                continue
            cs_std = float(cars.std(ddof=1))
            se_plain = cs_std / np.sqrt(n) if cs_std > 0 else nan
            cs_t = row["Mean_CAR"] / se_plain if cs_std > 0 else nan
            cs_p = float(2 * stats.t.sf(abs(cs_t), df=n - 1)) if not np.isnan(cs_t) else nan
            rbar = self._rbar_from(valid["Ticker"].dropna().unique().tolist())
            adj = nan
            if len(scars) >= 3:
                mean_scar, sd_scar = float(scars.mean()), float(scars.std(ddof=1))
                l_values = valid.dropna(subset=["SCAR"])["L_est"].to_numpy(dtype=float) if "L_est" in valid.columns else np.full(len(scars), estimation_window_length)
                patell_den = np.sqrt(np.nansum((l_values - 2.0) / (l_values - 4.0)))
                patell_z = float(scars.sum() / patell_den) if patell_den > 0 else nan
                patell_p = float(2 * stats.norm.sf(abs(patell_z))) if not np.isnan(patell_z) else nan
                bmp_t = mean_scar / (sd_scar / np.sqrt(len(scars))) if sd_scar > 0 else nan
                bmp_p = float(2 * stats.t.sf(abs(bmp_t), df=len(scars) - 1)) if not np.isnan(bmp_t) else nan
                denom = 1 + (len(scars) - 1) * rbar if not np.isnan(rbar) else nan
                adj = np.sqrt((1 - rbar) / denom) if not np.isnan(denom) and denom > 0 else nan
                adj_t = float(bmp_t * adj) if not np.isnan(bmp_t) and not np.isnan(adj) else nan
                adj_p = float(2 * stats.t.sf(abs(adj_t), df=len(scars) - 1)) if not np.isnan(adj_t) else nan
            else:
                bmp_t = bmp_p = patell_z = patell_p = adj_t = adj_p = nan
            se_kp = se_plain / adj if np.isfinite(se_plain) and np.isfinite(adj) and adj > 0 else nan
            row.update(dict(zip(keys, [cs_t, cs_p, bmp_t, bmp_p, adj_t, adj_p, patell_z, patell_p, rbar, se_plain, se_kp])))
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------ 机制回归 / mechanism regressions
    @staticmethod
    def _mechanism_design(df: pd.DataFrame, with_interaction: bool) -> pd.DataFrame:
        """设计矩阵：暴露等级、渠道哑变量、规模、区域固定效应（基准：美洲、航空）。
        Design matrix: exposure rank, channel dummies, size, region FE (baseline: Americas, airlines).
        """
        x = pd.DataFrame(index=df.index)
        exposure = df["exposure_group"].map(EXPOSURE_RANK).fillna(0.0)
        size = df["company_size_group"].eq("large").astype(float)
        if with_interaction:
            x["size_large"] = size
            x["exposure_rank"] = exposure
            x["size_x_exposure"] = size * exposure
        else:
            x["exposure_rank"] = exposure
        x["is_airport"] = df["industry_group"].eq("airport").astype(float)
        x["is_tourism"] = df["industry_group"].eq("tourism_hotel").astype(float)
        x["is_insurance"] = df["industry_group"].eq("insurance_finance").astype(float)
        if not with_interaction:
            x["size_large"] = size
        region = df["country_or_region"].map(region_of)
        for reg in ("Asia", "Europe", "Oceania"):
            x[f"region_{reg}"] = region.eq(reg).astype(float)
        return x.loc[:, x.nunique(dropna=False) > 1]

    @staticmethod
    def _hc3_table(y: pd.Series, x: pd.DataFrame, n_obs: int, extra: dict[str, object]) -> pd.DataFrame:
        res = OLS(y, add_constant(x, has_constant="add")).fit(cov_type="HC3")
        return pd.DataFrame(
            {**extra, "term": list(res.params.index), "coef": res.params.to_numpy(), "se_HC3": res.bse.to_numpy(),
             "t": res.tvalues.to_numpy(), "p": res.pvalues.to_numpy(), "n_obs": n_obs, "Status": "OK"}
        ).reset_index(drop=True)

    @staticmethod
    def cross_sectional_car_regression(table: pd.DataFrame, window_type: str = "long") -> pd.DataFrame:
        """表 4-2：CAR 的 HC3 机制回归。exposure_rank 识别的是航空公司内部的暴露梯度。
        Table 4-2: HC3 mechanism regression of CAR. exposure_rank identifies the within-airline exposure gradient.
        """
        needed = {"Window_Type", "Actual_CAR", "exposure_group", "industry_group", "company_size_group", "country_or_region"}

        def status(n_obs: int, s: str) -> pd.DataFrame:
            return pd.DataFrame([{"Window_Type": window_type, "term": "NA", "coef": np.nan, "se_HC3": np.nan, "t": np.nan, "p": np.nan, "n_obs": n_obs, "Status": s}])

        if not needed.issubset(table.columns):
            return status(0, "MISSING_CHARACTERISTIC_COLUMNS")
        df = table[table["Model_Status"].astype(str).eq("OK")] if "Model_Status" in table.columns else table
        df = df[df["Window_Type"].astype(str).eq(window_type)].dropna(subset=["Actual_CAR"])
        n_obs = len(df)
        if n_obs == 0:
            return status(0, "NO_VALID_OBS")
        x = EventStudyEngine._mechanism_design(df, with_interaction=False)
        if x.shape[1] == 0 or n_obs < x.shape[1] + 2:
            return status(n_obs, "INSUFFICIENT_VARIATION_OR_N")
        try:
            return EventStudyEngine._hc3_table(df["Actual_CAR"].astype(float), x, n_obs, {"Window_Type": window_type})
        except Exception as exc:
            return status(n_obs, f"OLS_FAILED: {exc}")

    @staticmethod
    def size_interaction_regression(table: pd.DataFrame, outcome_col: str, window_type: str | None = None) -> pd.DataFrame:
        """表 6-4：规模 × 暴露交互的 HC3 回归 / Table 6-4: HC3 regression with a size × exposure interaction."""
        needed = {outcome_col, "exposure_group", "industry_group", "company_size_group", "country_or_region"}

        def status(n_obs: int, s: str) -> pd.DataFrame:
            return pd.DataFrame([{"term": "NA", "coef": np.nan, "se_HC3": np.nan, "t": np.nan, "p": np.nan, "n_obs": n_obs, "Status": s}])

        if not needed.issubset(table.columns):
            return status(0, "MISSING_COLUMNS")
        df = table[table["Model_Status"].astype(str).eq("OK")] if "Model_Status" in table.columns else table
        if window_type is not None and "Window_Type" in df.columns:
            df = df[df["Window_Type"].astype(str).eq(window_type)]
        df = df.dropna(subset=[outcome_col])
        n_obs = len(df)
        if n_obs == 0:
            return status(0, "NO_OBS")
        x = EventStudyEngine._mechanism_design(df, with_interaction=True)
        if x.shape[1] == 0 or n_obs < x.shape[1] + 2:
            return status(n_obs, "INSUFFICIENT")
        try:
            return EventStudyEngine._hc3_table(df[outcome_col].astype(float), x, n_obs, {})
        except Exception as exc:
            return status(n_obs, f"OLS_FAILED:{exc}")

    @staticmethod
    def size_group_contrasts(table: pd.DataFrame, outcome_col: str, channels=("airport", "tourism_hotel"), n_perm: int = 9999, seed: int = 3702014) -> pd.DataFrame:
        """大型 vs 中小型公司置换检验（总体与渠道内）/ Large vs mid-small permutation contrasts (overall and within channel)."""
        rng = np.random.default_rng(seed)
        df = table.dropna(subset=[outcome_col])
        if "Model_Status" in df.columns:
            df = df[df["Model_Status"].astype(str).eq("OK")]
        rows = []

        def contrast(sub: pd.DataFrame, scope: str) -> None:
            large = sub[sub["company_size_group"] == "large"][outcome_col].to_numpy(dtype=float)
            small = sub[sub["company_size_group"] == "mid_small"][outcome_col].to_numpy(dtype=float)
            if len(large) < 2 or len(small) < 2:
                rows.append({"scope": scope, "n_large": len(large), "n_mid_small": len(small), "mean_large": np.nan, "mean_mid_small": np.nan, "diff": np.nan, "perm_p": np.nan})
                return
            rows.append({"scope": scope, "n_large": len(large), "n_mid_small": len(small), "mean_large": float(large.mean()), "mean_mid_small": float(small.mean()),
                         "diff": float(large.mean() - small.mean()), "perm_p": permutation_mean_diff(large, small, n_perm, rng)})

        contrast(df, "overall_confounded_by_industry")
        for channel in channels:
            contrast(df[df["industry_group"] == channel], f"within_{channel}")
        return pd.DataFrame(rows)

    # -------------------------------------------------------- 波动率 / volatility
    def volatility_response(self, asset_returns: pd.DataFrame, market_returns, benchmark_map=None, pre=(-180, -61), posts=((1, 60), (1, 120))) -> pd.DataFrame:
        """表 6-1：特质波动放大比 AVR = σ(事件后残差)/σ(事件前残差)。
        Table 6-1: abnormal volatility ratio AVR = σ(post-event residuals) / σ(pre-event residuals).
        """
        est_left, est_right = self.config.estimation_window
        rows = []
        for ticker in asset_returns.columns:
            try:
                aligned, bench = self._pair_frame(asset_returns, market_returns, benchmark_map, ticker)
                if aligned.empty:
                    raise ValueError("no common asset-benchmark trading dates")
                idx = aligned.index
                c = self._event_index(idx, self.config.event_date)
                if c + est_left < 0 or c + est_right >= len(idx):
                    raise ValueError("estimation window outside sample")
                self._check_bounds(c, len(idx), (pre, *posts))
                alpha, beta = self._fit_market_model(aligned["a"].iloc[c + est_left : c + est_right + 1], aligned["m"].iloc[c + est_left : c + est_right + 1])
                abnormal = aligned["a"] - (alpha + beta * aligned["m"])

                def window_std(left: int, right: int) -> float:
                    seg = abnormal.iloc[c + left : c + right + 1]
                    return float(seg.std(ddof=1)) if len(seg) > 2 else float("nan")

                sp = window_std(*pre)
                row: dict[str, object] = {"Ticker": ticker, "Benchmark": bench, "Sigma_pre": sp}
                for pl, pr in posts:
                    so = window_std(pl, pr)
                    row[f"Sigma_post_{pr}"] = so
                    row[f"AVR_{pr}"] = so / sp if sp and sp > 0 else np.nan
                    row[f"logAVR_{pr}"] = float(np.log(so / sp)) if sp and sp > 0 and so and so > 0 else np.nan
                row["Model_Status"] = "OK"
                rows.append(row)
            except Exception as exc:
                rows.append({"Ticker": ticker, "Model_Status": f"VOL_RESPONSE_FAILED: {exc}"})
        return pd.DataFrame(rows)

    def did_volatility_response(self, asset_returns, market_returns, benchmark_map, char, pre=(-180, -61), posts=((1, 60), (1, 120)), treated_groups=TREATED_GROUPS, n_perm: int = 9999, seed: int = 3702014):
        """表 6-2：处理组（高暴露航空）− 对照组 的 logAVR 差异，置换检验。
        Table 6-2: treated (high-exposure airlines) minus control difference in logAVR, permutation test.
        """
        vol = self.volatility_response(asset_returns, market_returns, benchmark_map, pre=pre, posts=posts)
        vol = vol[vol["Model_Status"].astype(str).eq("OK")]
        if char is not None and not char.empty and "Ticker" in char.columns:
            vol = vol.merge(char, on="Ticker", how="left")
        last = posts[-1][1]
        bench_logavr: dict[str, float] = {}
        if isinstance(market_returns, pd.DataFrame) and benchmark_map:
            for bench in set(benchmark_map.values()):
                if bench not in market_returns.columns:
                    continue
                s = market_returns[bench].dropna()
                c = self._event_index(s.index, self.config.event_date)
                try:
                    self._check_bounds(c, len(s), (pre, posts[-1]))
                except ValueError:
                    bench_logavr[bench] = np.nan
                    continue
                sp = float(s.iloc[c + pre[0] : c + pre[1] + 1].std(ddof=1))
                so = float(s.iloc[c + posts[-1][0] : c + posts[-1][1] + 1].std(ddof=1))
                bench_logavr[bench] = float(np.log(so / sp)) if sp > 0 and so > 0 else np.nan
        key = "benchmark_ticker" if "benchmark_ticker" in vol.columns else ("Benchmark" if "Benchmark" in vol.columns else None)
        vol["bench_logAVR"] = vol[key].map(bench_logavr) if key else np.nan
        # 相对市场的 logAVR = 特质 logAVR − 基准指数 logAVR / Market-relative logAVR
        vol["market_relative_logAVR"] = vol[f"logAVR_{last}"] - vol["bench_logAVR"]
        vol["treated"] = vol["exposure_group"].isin(treated_groups) if "exposure_group" in vol.columns else False
        rng = np.random.default_rng(seed)
        rows = []
        for column, label in ((f"logAVR_{last}", "idiosyncratic_logAVR_DiD"), ("market_relative_logAVR", "market_relative_logAVR_DiD")):
            valid = vol.dropna(subset=[column])
            tr = valid[valid["treated"]][column].to_numpy(dtype=float)
            co = valid[~valid["treated"]][column].to_numpy(dtype=float)
            if len(tr) >= 2 and len(co) >= 2:
                rows.append({"outcome": label, "n_treated": len(tr), "n_control": len(co), "mean_treated": float(tr.mean()), "mean_control": float(co.mean()),
                             "DiD_treated_minus_control": float(tr.mean() - co.mean()), "perm_p": permutation_mean_diff(tr, co, n_perm, rng)})
            else:
                rows.append({"outcome": label, "n_treated": len(tr), "n_control": len(co), "mean_treated": np.nan, "mean_control": np.nan, "DiD_treated_minus_control": np.nan, "perm_p": np.nan})
        return pd.DataFrame(rows), vol

    # -------------------------------------------------- 事件时间 / event time
    def event_time_caar(self, asset_returns, market_returns, benchmark_map=None, tau_min: int = -10, tau_max: int = 120, char=None):
        """图 4-1：事件时间 CAAR(τ)，含总体与高/低暴露组。
        Figure 4-1: event-time CAAR(τ) for all firms and for high/low exposure groups.
        """
        est_left, est_right = self.config.estimation_window
        taus = list(range(tau_min, tau_max + 1))
        cumulative: dict[str, np.ndarray] = {}
        for ticker in asset_returns.columns:
            try:
                aligned, _ = self._pair_frame(asset_returns, market_returns, benchmark_map, ticker)
                c = self._event_index(aligned.index, self.config.event_date)
                if c + est_left < 0 or c + tau_min < 0 or c + tau_max >= len(aligned):
                    continue
                sl = slice(c + est_left, c + est_right + 1)
                alpha, beta = self._fit_market_model(aligned["a"].iloc[sl], aligned["m"].iloc[sl])
                ab = (aligned["a"] - (alpha + beta * aligned["m"])).to_numpy(dtype=float)
                cumulative[ticker] = np.cumsum(ab[c + tau_min : c + tau_max + 1])
            except Exception:
                continue
        if not cumulative:
            return pd.DataFrame(), pd.DataFrame()
        matrix = pd.DataFrame(cumulative, index=taus).T
        n = len(matrix)
        overall = pd.DataFrame({"tau": taus, "CAAR": matrix.mean(axis=0).to_numpy(), "SE_CAAR": (matrix.std(ddof=1, axis=0) / np.sqrt(n)).to_numpy(), "N": n})
        grouped = pd.DataFrame()
        if char is not None and not char.empty and {"Ticker", "exposure_group"}.issubset(char.columns):
            emap = dict(zip(char["Ticker"].astype(str), char["exposure_group"].astype(str)))
            labels = pd.Series({t: "treated_high_exposure" if emap.get(t) in TREATED_GROUPS else "control_lower_exposure" for t in matrix.index})
            parts = []
            for label in ("treated_high_exposure", "control_lower_exposure"):
                sub = matrix[labels.reindex(matrix.index) == label]
                if len(sub) >= 3:
                    parts.append(pd.DataFrame({"group": label, "tau": taus, "CAAR": sub.mean(axis=0).to_numpy(), "SE_CAAR": (sub.std(ddof=1, axis=0) / np.sqrt(len(sub))).to_numpy(), "N": len(sub)}))
            grouped = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return overall, grouped

    def estimation_std_residuals_frame(self) -> pd.DataFrame:
        """估计窗标准化残差（按末端对齐）/ Estimation-window standardised residuals (right-aligned)."""
        if not self.est_std_residuals_:
            return pd.DataFrame()
        max_len = max(len(v) for v in self.est_std_residuals_.values())
        cols = {}
        for ticker, vec in self.est_std_residuals_.items():
            arr = np.asarray(vec, dtype=float)
            padded = np.full(max_len, np.nan)
            padded[-len(arr):] = arr
            cols[ticker] = padded
        frame = pd.DataFrame(cols)
        frame.insert(0, "estimation_residual_position", np.arange(max_len))
        return frame
