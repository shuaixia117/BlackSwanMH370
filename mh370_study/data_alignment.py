"""第 3 章：价格读取、交易日对齐与单变量诊断。
Chapter 3: price loading, trading-calendar alignment and univariate diagnostics.

* build_native_returns：每只资产按本地交易日历计算对数收益，再按日期外连接，不做全局交集。
  Log returns on each asset's own trading calendar, outer-joined by date (no global intersection).
* pair_align：两两对齐（同步 / 两日重叠 overlap2 / 滞后一期 lag1）。
  Pairwise alignment (contemporaneous / two-day overlapping overlap2 / one-period lag1).
* align_from_prices：全样本同步 + 缩尾，仅供诊断使用。
  Fully synchronised and winsorised panel, for diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

PRICE_COLUMNS = ("adjusted_close", "adj_close", "Adj Close", "close", "Close")
META_COLS = ("industry_group", "exposure_group", "company_size_group", "country_or_region", "currency", "benchmark_ticker")


@dataclass(frozen=True)
class AlignmentConfig:
    winsor_limits: tuple[float, float] = (0.01, 0.01)
    winsorize: bool = True
    min_observations: int = 300


def _log_returns(px: pd.Series) -> pd.Series:
    """一日对数收益（小数单位）/ One-day log returns in decimal units."""
    px = pd.to_numeric(px, errors="coerce").where(lambda v: v > 0)
    return np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


class DataAligner:
    """读取价格、构造收益、对齐、缩尾与诊断。
    Load prices, build returns, align, winsorise and diagnose.
    """

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or AlignmentConfig()

    @staticmethod
    def _select_price_column(df: pd.DataFrame) -> str:
        for col in PRICE_COLUMNS:
            if col in df.columns:
                return col
        raise ValueError(f"No usable price column found. Expected one of {PRICE_COLUMNS}.")

    def _read_price_file(self, file: Path) -> tuple[str, pd.Series, dict[str, object]]:
        """读取单个价格文件，返回 (代码, 价格序列, 元数据)。
        Read one price file and return (ticker, price series, metadata).
        """
        df = pd.read_csv(file)
        price_col = self._select_price_column(df)
        if "date" not in df.columns:
            raise ValueError(f"Missing date column in {file}")
        has_ticker = "ticker" in df.columns and df["ticker"].notna().any()
        ticker = str(df["ticker"].dropna().iloc[0]) if has_ticker else file.stem
        s = df[["date", price_col]].copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s[price_col] = pd.to_numeric(s[price_col], errors="coerce")
        s = s.dropna(subset=["date", price_col]).drop_duplicates("date").sort_values("date")
        meta: dict[str, object] = {"Ticker": ticker}
        for col in META_COLS:
            meta[col] = df[col].dropna().iloc[0] if col in df.columns and df[col].notna().any() else None
        return ticker, s.set_index("date")[price_col].rename(ticker), meta

    def load_price_folder(self, folder: str | Path) -> pd.DataFrame:
        """文件夹 → 宽格式价格表 / Folder of CSVs → wide price panel."""
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Price folder does not exist: {folder}")
        series = [self._read_price_file(f)[1] for f in sorted(folder.glob("*.csv"))]
        if not series:
            raise ValueError(f"No CSV price files found in {folder}")
        return pd.concat(series, axis=1).sort_index()

    def load_price_panel(self, path: str | Path) -> pd.DataFrame:
        """读取文件夹或宽表 CSV / Load a folder or a wide CSV panel."""
        path = Path(path)
        if path.is_dir():
            return self.load_price_folder(path)
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Wide price panel must contain a date column.")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).set_index("date").sort_index().apply(pd.to_numeric, errors="coerce")

    def build_native_returns(self, folder_or_panel) -> tuple[pd.DataFrame, pd.DataFrame]:
        """本地日历对数收益宽表（NaN = 非交易日）与逐资产元数据。
        Wide native-calendar log-return frame (NaN = non-trading day) plus per-asset metadata.
        """
        returns: dict[str, pd.Series] = {}
        char_rows: list[dict[str, object]] = []
        path = None if isinstance(folder_or_panel, pd.DataFrame) else Path(folder_or_panel)
        if path is not None and path.is_dir():
            files = sorted(path.glob("*.csv"))
            if not files:
                raise ValueError(f"No CSV price files found in {path}")
            for file in files:
                ticker, px, meta = self._read_price_file(file)
                r = _log_returns(px)
                if not r.empty:
                    returns[ticker] = r.rename(ticker)
                    char_rows.append(meta)
        else:
            panel = folder_or_panel.copy() if path is None else self.load_price_panel(path)
            for col in panel.columns:
                r = _log_returns(panel[col])
                if not r.empty:
                    returns[str(col)] = r.rename(str(col))
                    char_rows.append({"Ticker": str(col), **{c: None for c in META_COLS}})
        if not returns:
            raise ValueError(f"No usable return series built from {folder_or_panel}")
        return pd.concat(returns.values(), axis=1).sort_index(), pd.DataFrame(char_rows)

    @staticmethod
    def log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
        clean = price_panel.apply(pd.to_numeric, errors="coerce")
        clean = clean.where(clean > 0)
        return np.log(clean / clean.shift(1))

    @staticmethod
    def synchronize(return_panel: pd.DataFrame) -> pd.DataFrame:
        """严格共同交易日交集 / Strict common trading-day intersection."""
        return return_panel.replace([np.inf, -np.inf], np.nan).dropna(how="any").sort_index()

    @staticmethod
    def winsorize_frame(df: pd.DataFrame, limits: tuple[float, float]) -> pd.DataFrame:
        """逐列双侧分位数缩尾 / Column-wise two-sided quantile clipping."""
        lower, upper = limits
        if lower < 0 or upper < 0 or lower + upper >= 1:
            raise ValueError("Winsor limits must be non-negative and sum to less than 1.")
        out = df.copy()
        for col in out.columns:
            out[col] = out[col].clip(out[col].quantile(lower), out[col].quantile(1 - upper))
        return out

    @staticmethod
    def arch_lm_pvalue(series: pd.Series, lags: int) -> float:
        """Engle (1982) ARCH-LM 检验 p 值（去均值），H0：无 ARCH 效应。
        Engle (1982) ARCH-LM p-value on demeaned returns; H0: no ARCH effects.
        """
        y = pd.Series(series).dropna().astype(float)
        if len(y) <= lags + 5:
            return float("nan")
        return float(het_arch(y - y.mean(), nlags=lags)[1])

    @staticmethod
    def pair_align(a: pd.Series, b: pd.Series, mode: str = "contemporaneous", lag_first: bool = True) -> pd.DataFrame:
        """两两对齐；overlap2 用相邻两日收益和缓解收盘时差。
        Pairwise alignment; overlap2 uses adjacent two-day sums to soften closing-time differences.
        """
        left, right = a.dropna(), b.dropna()
        if mode == "overlap2":
            left, right = left.rolling(2).sum(), right.rolling(2).sum()
        elif mode == "lag1":
            if lag_first:
                left = left.shift(1)
            else:
                right = right.shift(1)
        elif mode != "contemporaneous":
            raise ValueError(f"unknown alignment mode: {mode}")
        return pd.concat([left, right], axis=1, keys=["left", "right"]).dropna()

    def coverage_diagnostics(self, returns_frame: pd.DataFrame, event_date, estimation_window: tuple[int, int], char_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """表 3-1：逐资产样本覆盖诊断 / Table 3-1: per-asset sample coverage diagnostics."""
        event_ts = pd.Timestamp(event_date)
        need_pre = abs(int(estimation_window[0]))
        char = char_df.set_index("Ticker") if char_df is not None and "Ticker" in getattr(char_df, "columns", []) else None
        rows = []
        for ticker in returns_frame.columns:
            idx = returns_frame[ticker].dropna().index
            n, n_pre = int(len(idx)), int((idx < event_ts).sum())
            row: dict[str, object] = {
                "Ticker": ticker,
                "N": n,
                "First": idx.min().date().isoformat() if n else None,
                "Last": idx.max().date().isoformat() if n else None,
                "N_pre_event": n_pre,
                "N_post_event": n - n_pre,
                "Est_Window_Satisfiable": bool(n_pre >= need_pre + 1),
            }
            if char is not None and ticker in char.index:
                for col in ("exposure_group", "industry_group", "country_or_region"):
                    row[col] = char.loc[ticker, col] if col in char.columns else None
            rows.append(row)
        return pd.DataFrame(rows)

    def diagnostics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """表 3-3：日收益描述统计与 ARCH-LM / Table 3-3: daily-return statistics and ARCH-LM."""
        rows = []
        for ticker in returns.columns:
            x = returns[ticker].dropna().astype(float)
            rows.append(
                {
                    "Ticker": ticker,
                    "T": int(len(x)),
                    "Mean": float(x.mean()),
                    "Std": float(x.std(ddof=1)),
                    "Skew": float(stats.skew(x, bias=False)),
                    "Kurt": float(stats.kurtosis(x, fisher=False, bias=False)),
                    "ARCH_LM1_p": self.arch_lm_pvalue(x, 1),
                    "ARCH_LM5_p": self.arch_lm_pvalue(x, 5),
                }
            )
        return pd.DataFrame(rows)

    def align_from_prices(self, price_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        synced = self.synchronize(self.log_returns(price_panel))
        if len(synced) < self.config.min_observations:
            raise ValueError(f"Synchronized panel too short: T={len(synced)} < {self.config.min_observations}")
        out = self.winsorize_frame(synced, self.config.winsor_limits) if self.config.winsorize else synced
        return out, self.diagnostics(out)


def common_columns(*frames: pd.DataFrame) -> list[str]:
    """多个表的有序列交集 / Ordered column intersection across frames."""
    if not frames:
        return []
    out = list(frames[0].columns)
    for frame in frames[1:]:
        have = set(frame.columns)
        out = [c for c in out if c in have]
    return out
