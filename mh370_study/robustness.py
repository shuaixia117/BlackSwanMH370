"""稳健性检验：只重跑替代设定，不修改主估计对象。
Robustness checks: re-run alternative specifications without touching the main estimates.

表 4-3：替代估计窗 + 扩展事件窗；表 5-3：DCC 事件前相关目标。
Table 4-3: alternative estimation windows and an extended event window;
Table 5-3: DCC with pre-event correlation targeting.
另含时区滞后 DCC、缩尾 GARCH、替代基准三项附加检验（未进入正文）。
Also provides time-zone-lag DCC, winsorised GARCH and alternative-benchmark checks (not in the main text).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contagion_tests import AlignmentRule, DccConfig, DccFdrEngine
from .data_alignment import AlignmentConfig, DataAligner
from .event_study import EventStudyConfig, EventStudyEngine
from .garch_var import GarchVarConfig, GarchVarEngine


@dataclass(frozen=True)
class RobustnessConfig:
    event_date: str
    random_seed: int = 3702014
    confounding_dates: tuple[str, ...] = ()
    confounding_block: int = 5


class RobustnessEngine:
    def __init__(self, config: RobustnessConfig) -> None:
        self.config = config

    def _common(self) -> dict:
        c = self.config
        return dict(event_date=c.event_date, random_seed=c.random_seed, confounding_dates=c.confounding_dates, confounding_block=c.confounding_block)

    def event_window_sensitivity(self, asset_returns, market_returns, benchmark_map) -> pd.DataFrame:
        """表 4-3：估计窗 [-100,-20] 与 [-150,-10]，检验窗 [-5,+40]。
        Table 4-3: estimation windows [-100,-20] and [-150,-10]; test window [-5,+40].
        """
        specs = [
            ("5C_estimation_100_20", (-100, -20), (("long_extended", -5, 40),)),
            ("5C_estimation_150_10", (-150, -10), (("long_extended", -5, 40),)),
        ]
        parts = []
        for name, est, tests in specs:
            out = EventStudyEngine(EventStudyConfig(estimation_window=est, test_windows=tests, **self._common())).run(asset_returns, market_returns, benchmark_map)
            out["Robustness_Test"] = name
            parts.append(out)
        return pd.concat(parts, ignore_index=True)

    def dcc_pre_event_targeting(self, residuals, pairs, alignment_for: AlignmentRule | None = None) -> pd.DataFrame:
        """表 5-3：全样本 vs 仅事件前样本估计 Q̄ / Table 5-3: Q̄ from the full sample vs the pre-event sample only."""
        parts = []
        for targeting in ("full", "pre_event"):
            out = DccFdrEngine(DccConfig(correlation_targeting=targeting, **self._common())).run(residuals, pairs, alignment_for=alignment_for)
            out["Robustness_Test"] = "5E_dcc_pre_event_targeting"
            out["Targeting_Specification"] = targeting
            parts.append(out)
        return pd.concat(parts, ignore_index=True)

    # --------------------------------------------- 附加检验 / supplementary checks
    def timezone_lag_dcc(self, residuals, pairs, lag_tickers: set[str] | None = None) -> pd.DataFrame:
        """同步 vs 一侧滞后一期的 DCC / DCC with contemporaneous vs one-side-lagged alignment."""
        engine = DccFdrEngine(DccConfig(alignment_mode="contemporaneous", **self._common()))
        lag0 = engine.run(residuals, pairs)
        lag0["Lag_Specification"] = "Lag-0"
        parts = [lag0]
        for left, right in pairs:
            lag_left = left if lag_tickers is None or left in lag_tickers else right
            other = right if lag_left == left else left
            try:
                frame = pd.DataFrame({lag_left: residuals[lag_left].shift(1), other: residuals[other]}).dropna()
                if len(frame) < 100:
                    raise ValueError(f"Lag-1 frame for {left}-{right} has only {len(frame)} rows after dropna.")
                res = engine.run(frame[[left, right]], [(left, right)])
                res["Lag_Specification"] = "Lag-1"
                res["Lagged_Ticker"] = lag_left
                parts.append(res)
            except Exception as exc:
                parts.append(pd.DataFrame([{"Pair": f"{left}|{right}", "Window_Type": "all", "Lag_Specification": "Lag-1", "Lagged_Ticker": lag_left, "Model_Status": f"LAG_DCC_FAILED: {exc}"}]))
        out = pd.concat(parts, ignore_index=True)
        out["Robustness_Test"] = "5A_timezone_lag"
        return out

    def winsorization_garch(self, price_panel, winsor_levels=(0.005, 0.02)) -> pd.DataFrame:
        """缩尾后重估 GARCH / Re-estimate GARCH on winsorised returns."""
        parts = []
        for level in winsor_levels:
            aligner = DataAligner(AlignmentConfig(winsor_limits=(level, level)))
            native, _ = aligner.build_native_returns(price_panel)
            ta, _b, _r = GarchVarEngine(GarchVarConfig()).run(aligner.winsorize_frame(native, (level, level)))
            sel = ta.sort_values(["Ticker", "Penalized_BIC"]).groupby("Ticker", as_index=False).head(1)
            sel["Robustness_Test"] = "5B_winsorization"
            sel["Winsor_Level"] = level
            parts.append(sel)
        return pd.concat(parts, ignore_index=True)

    def alternative_benchmark(self, asset_returns, alternative_market: pd.Series, label: str = "alternative_blue_chip_index") -> pd.DataFrame:
        """替代基准指数下的 CAR 方向一致性 / CAR direction consistency under an alternative benchmark."""
        out = EventStudyEngine(EventStudyConfig(**self._common())).run(asset_returns, alternative_market)
        out["Robustness_Test"] = "5D_alternative_benchmark"
        out["Alternative_Benchmark_Label"] = label
        out["Consistency_Direction"] = np.where(out["Actual_CAR"] >= 0, "nonnegative", "negative")
        return out
