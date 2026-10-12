"""表 4-4：FF3 / FF5 多因子正常收益模型稳健性。
Table 4-4: FF3 / FF5 multi-factor normal-return robustness.

口径 / Specification
* 美元超额收益：本币价格按日度汇率（在股票交易日上前向填充，≤4 个自然日）换算为美元，再减美国无风险利率；
  USD excess returns: local prices converted with daily FX (forward-filled onto stock trading days,
  at most 4 calendar days) minus the US risk-free rate;
* 发达市场用 Kenneth French 区域因子，新兴市场（中、马、韩、土、泰）用 JKP 国别因子，不以发达市场因子代替；
  Developed markets use Kenneth French regional factors; emerging markets (CN, MY, KR, TR, TH) use JKP
  country factors, with no developed-market proxy;
* SCAR 方差含多因子预测误差项；MDE 采用 CAR 尺度上经 KP 膨胀的截面标准误。
  The SCAR variance includes the multi-factor prediction-error term; MDE uses the KP-inflated
  cross-sectional standard error on the CAR scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS, add_constant

from .config import EVENT_DATE, Paths
from .event_study import DEFAULT_TEST_WINDOWS, EventStudyConfig, EventStudyEngine
from .stats_utils import Z_80_POWER, bh_fdr

ESTIMATION_WINDOW = (-210, -30)
MIN_EST_OBS = 120
TEST_WINDOWS = DEFAULT_TEST_WINDOWS
TABLE4_WINDOWS = ("short", "sym3", "sym5", "long")
WINDOW_LABEL = {"short": "[-1,+1]", "sym3": "[-3,+3]", "sym5": "[-5,+5]", "med": "[-5,+10]", "long": "[-5,+30]", "post20": "[+1,+20]", "post60": "[+1,+60]", "post120": "[+1,+120]"}
FACTOR_COLS = {"FF3": ["Mkt-RF", "SMB", "HML"], "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]}
FRENCH_REGIONS = {"North America": "North_America", "Europe": "Europe", "Japan": "Japan", "Asia Pacific": "Asia_Pacific"}
JKP_COUNTRY = {"China": "chn", "Malaysia": "mys", "South Korea": "kor", "Turkey": "tur", "Thailand": "tha"}
ADR_TICKERS = {"IHG", "RYAAY", "TCOM", "ASR", "PAC", "OMAB"}
EUROPE = {"France", "Germany", "United Kingdom", "Switzerland", "Austria", "Spain", "Norway", "Ireland", "Germany/United Kingdom"}
FX_TOLERANCE = pd.Timedelta("4D")


@dataclass
class FactorAssignment:
    factor_region: str
    factor_region_raw: str
    factor_source: str
    factor_available: bool
    mapping_rule: str
    exclusion_reason: str


# --------------------------------------------------------------- 数据读取 / loading
def load_factors(paths: Paths) -> dict[tuple[str, str], pd.DataFrame]:
    """读取已解析的因子文件，键为 (区域, 模型) / Load parsed factor files keyed by (region, model)."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for region, stem in FRENCH_REGIONS.items():
        for model in ("FF3", "FF5"):
            f = paths.factors / "french" / f"{stem}_{model}_daily_decimal.csv"
            if f.exists():
                out[(region, model)] = pd.read_csv(f, parse_dates=["date"])
    for code in JKP_COUNTRY.values():
        for model in ("FF3", "FF5"):
            f = paths.factors / "jkp" / f"JKP_{code}_{model}_daily_decimal.csv"
            if f.exists():
                out[(f"JKP:{code}", model)] = pd.read_csv(f, parse_dates=["date"])
    return out


def _token(ticker: str) -> str:
    return str(ticker).strip().replace("^", "").replace(".", "_").replace("-", "_")


def company_file(paths: Paths, ticker: str) -> Path | None:
    matches = sorted(paths.companies.glob(f"{_token(ticker)}_*.csv"))
    return matches[0] if matches else None


def load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["date", "adjusted_close"])
    df["date"] = pd.to_datetime(df["date"])
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def assign_region(row: pd.Series, factors: dict) -> FactorAssignment:
    """按交易市场 / 国别分配因子来源 / Assign the factor source by trading market or country."""
    country, exchange, ticker = str(row.get("country_or_region", "")), str(row.get("exchange", "")), str(row.get("ticker", ""))
    kf = "Kenneth French regional"
    if exchange in {"NYSE", "NASDAQ", "NYSE ADR", "NASDAQ ADR"} or "ADR" in exchange or ticker in ADR_TICKERS:
        return FactorAssignment("North America", "North America", kf, True, "US/Canada/ADR trading-market assignment", "")
    if country in {"United States", "Canada", "Mexico/United States ADR", "Panama/United States"}:
        return FactorAssignment("North America", "North America", kf, True, "North America country assignment", "")
    if country in EUROPE:
        return FactorAssignment("Europe", "Europe", kf, True, "Europe trading/country assignment", "")
    if country == "Japan":
        return FactorAssignment("Japan", "Japan", kf, True, "Japan country assignment", "")
    if country in {"Australia", "Hong Kong", "New Zealand", "Singapore"}:
        return FactorAssignment("Asia Pacific", "Asia Pacific", kf, True, "Developed Asia-Pacific assignment", "")
    if country in JKP_COUNTRY:
        region = f"JKP:{JKP_COUNTRY[country]}"
        if (region, "FF3") in factors and (region, "FF5") in factors:
            return FactorAssignment(region, "Emerging", "JKP country theme factors", True, "Emerging-market country factors from JKP (JKP China used for China; no CH-3 series fabricated).", "")
        return FactorAssignment(region, "Emerging", "unavailable", False, "Emerging-market daily factors required; no proxy used", f"factor_unavailable_emerging:{country}")
    return FactorAssignment("", "Unmapped", "unavailable", False, "Unmapped market; no proxy used", f"factor_unavailable_unmapped:{country}")


def usd_log_return(price: pd.DataFrame, currency: str, fx: pd.DataFrame) -> pd.DataFrame:
    """本币价格 → 美元日对数收益 / Local-currency prices → daily USD log returns."""
    c = "GBP" if currency == "GBp" else currency
    if c == "USD":
        tmp = price[["date", "adjusted_close"]].copy()
        tmp["usd_price"] = tmp["adjusted_close"]
    else:
        if c not in fx.columns:
            raise ValueError(f"FX currency unavailable: {currency}")
        tmp = pd.merge_asof(price.sort_values("date"), fx[["date", c]].dropna().sort_values("date"), on="date", direction="backward", tolerance=FX_TOLERANCE).dropna(subset=[c])
        tmp["usd_price"] = tmp["adjusted_close"] * tmp[c]
    tmp["usd_ret"] = np.log(tmp["usd_price"]).diff()
    return tmp[["date", "usd_ret"]].dropna().reset_index(drop=True)


# ------------------------------------------------------------- 逐公司估计 / per firm
def _unavailable(row, model, a: FactorAssignment, status: str) -> list[dict[str, object]]:
    return [
        {"model": model, "Ticker": str(row["ticker"]), "Benchmark": "", "Window_Type": w, "Window_Left": wl, "Window_Right": wr,
         "Actual_CAR": np.nan, "SCAR": np.nan, "L_est": 0, "Resid_Var": np.nan, "Firm_p": np.nan, "Model_Status": status,
         "factor_region": a.factor_region, "factor_region_raw": a.factor_region_raw, "factor_source": a.factor_source,
         "factor_available": a.factor_available, "mapping_rule": a.mapping_rule, "exclusion_reason": a.exclusion_reason, "company_name": str(row["company_name"])}
        for w, wl, wr in TEST_WINDOWS
    ]


def fit_asset(row: pd.Series, model: str, factors: dict, fx: pd.DataFrame, paths: Paths) -> dict[str, object]:
    """单公司多因子事件研究 / Multi-factor event study for one firm."""
    ticker, name = str(row["ticker"]), str(row["company_name"])
    a = assign_region(row, factors)
    base_diag = {"model": model, "ticker": ticker, "company_name": name, "factor_region": a.factor_region, "factor_region_raw": a.factor_region_raw, "factor_source": a.factor_source, "factor_available": a.factor_available}
    empty = {"n_est": 0, "r2": np.nan, "residual_var": np.nan}
    if not a.factor_available:
        rows = _unavailable(row, model, a, a.exclusion_reason or "FACTOR_UNAVAILABLE")
        return {"rows": rows, "std_resid": None, "diag": {**base_diag, "status": rows[0]["Model_Status"], **empty}}
    p = company_file(paths, ticker)
    if p is None:
        return {"rows": _unavailable(row, model, a, "PRICE_FILE_MISSING"), "std_resid": None, "diag": {**base_diag, "status": "PRICE_FILE_MISSING", **empty}}
    try:
        df = usd_log_return(load_price(p), str(row["currency"]), fx).merge(factors[(a.factor_region, model)], on="date", how="inner").sort_values("date").reset_index(drop=True)
        df["merged_pos"] = np.arange(len(df))
        pos = np.flatnonzero(df["date"] >= pd.Timestamp(EVENT_DATE))
        if len(pos) == 0:
            raise ValueError("event date outside sample")
        ec = int(pos[0])
        l, r = ESTIMATION_WINDOW
        if ec + l < 1 or ec + r >= len(df):
            raise ValueError("estimation window outside stock-factor common sample")
        cols = FACTOR_COLS[model]
        est = df[(df["merged_pos"] >= ec + l) & (df["merged_pos"] <= ec + r)].copy()
        est["y"] = est["usd_ret"] - est["RF"]
        est = est.dropna(subset=["y"] + cols)
        if len(est) < MIN_EST_OBS:
            raise ValueError(f"insufficient_factor_window:{len(est)}")
        X = add_constant(est[cols].to_numpy(float), has_constant="add")
        res = OLS(est["y"].to_numpy(float), X).fit()
        sigma2 = float(np.var(res.resid, ddof=len(cols) + 1))
        xtx_inv = np.linalg.inv(X.T @ X)
        std_resid = None
        if sigma2 > 0:
            std_resid = pd.Series(res.resid / np.sqrt(sigma2), index=est["date"], name=ticker)
            std_resid = std_resid[~std_resid.index.duplicated(keep="first")]
        params = {"const": float(res.params[0]), **{c: float(res.params[i]) for i, c in enumerate(cols, start=1)}}
        rows = []
        for w, wl, wr in TEST_WINDOWS:
            need = list(range(ec + wl, ec + wr + 1))
            ev = df[df["merged_pos"].isin(need)].copy()
            ev["y"] = ev["usd_ret"] - ev["RF"]
            ev = ev.dropna(subset=["y"] + cols)
            base = {"model": model, "Ticker": ticker, "Benchmark": "", "Window_Type": w, "Window_Left": wl, "Window_Right": wr, "L_est": len(est), "Resid_Var": sigma2,
                    "factor_region": a.factor_region, "factor_region_raw": a.factor_region_raw, "factor_source": a.factor_source, "factor_available": a.factor_available,
                    "mapping_rule": a.mapping_rule, "exclusion_reason": a.exclusion_reason, "company_name": name}
            if sorted(ev["merged_pos"].tolist()) != need:
                missing = sorted(set(need) - set(ev["merged_pos"].tolist()))
                rows.append({**base, "Actual_CAR": np.nan, "SCAR": np.nan, "Firm_p": np.nan, "Model_Status": f"EVENT_FACTOR_DATES_MISSING:{len(missing)}"})
                continue
            Xw = add_constant(ev[cols].to_numpy(float), has_constant="add")
            car = float(np.sum(ev["y"].to_numpy(float) - res.predict(Xw)))
            # Var(CAR) = σ² [L + 1'X_w (X'X)^{-1} X_w'1]（含预测误差 / incl. prediction error)
            s = Xw.sum(axis=0)
            car_var = sigma2 * (len(ev) + float(s @ xtx_inv @ s))
            scar = float(car / np.sqrt(car_var)) if sigma2 > 0 else np.nan
            rows.append({**base, "Actual_CAR": car, "SCAR": scar, "Firm_p": float(2 * stats.norm.sf(abs(scar))) if np.isfinite(scar) else np.nan, "Model_Status": "OK"})
        diag = {**base_diag, "status": "OK", "n_est": len(est), "r2": float(res.rsquared), "residual_var": sigma2, **{f"coef_{k}": v for k, v in params.items()}}
        return {"rows": rows, "std_resid": std_resid, "diag": diag}
    except Exception as exc:
        status = f"FACTOR_MODEL_FAILED:{exc}"
        return {"rows": _unavailable(row, model, a, status), "std_resid": None, "diag": {**base_diag, "status": status, **empty}}


# --------------------------------------------------------------- 聚合 / aggregation
def _fmt_rows(inf: pd.DataFrame, firm_fdr: dict[str, int], model: str, region: str, notes_fn) -> pd.DataFrame:
    rows = []
    for _, r in inf.iterrows():
        w = str(r["Window_Type"])
        g = lambda k: float(r[k]) if k in r and pd.notna(r[k]) else np.nan  # noqa: E731
        se_p, se_a = g("SE_CAR_plain"), g("SE_CAR_KP")
        rows.append(
            {
                "window": WINDOW_LABEL.get(w, w), "Window_Type": w, "model": model, "region": region, "CAAR": g("Mean_CAR"),
                "Plain_CS_t": g("Plain_CS_t"), "Plain_CS_p": g("Plain_CS_p"), "Patell_p": g("Patell_p"), "BMP_p": g("BMP_p"),
                "ADJBMP_t": g("ADJ_BMP_t"), "ADJBMP_p": g("ADJ_BMP_p"), "firm_FDR": firm_fdr.get(w, 0),
                "SE_plainCS": se_p, "MDE_plainCS": Z_80_POWER * se_p, "SE_ADJBMP": se_a, "MDE_ADJBMP": Z_80_POWER * se_a,
                "N_firms": int(r["N"]), "Rbar_Estimation_Residual_Correlation": g("Rbar_Estimation_Residual_Correlation"), "notes": notes_fn(w),
            }
        )
    return pd.DataFrame(rows)


def _ff_notes(w: str) -> str:
    note = "USD excess-return FF factor model; no Developed proxy for emerging markets."
    if w in {"post60", "post120"}:
        note += " Long post-event USD window contains FX drift and is reported as reference only."
    return note


def infer_aggregate(event_df: pd.DataFrame, std_resids: dict[str, pd.Series], model: str, region: str) -> pd.DataFrame:
    """聚合 CAAR 与截面检验（全样本或某区域）/ Aggregate CAAR and cross-sectional tests (all firms or one region)."""
    engine = EventStudyEngine(EventStudyConfig(event_date=EVENT_DATE))
    engine.est_std_residuals_ = std_resids
    sub = event_df[event_df["model"].eq(model)]
    if region != "all":
        sub = sub[sub["factor_region_raw"].eq(region)]
    sub = sub[sub["Model_Status"].eq("OK")].copy()
    if sub.empty:
        return pd.DataFrame()
    inf = engine.cross_sectional_inference(sub, abs(ESTIMATION_WINDOW[1] - ESTIMATION_WINDOW[0]) + 1)
    firm_fdr = {str(w): int(np.nansum(bh_fdr(g["Firm_p"].to_numpy(float)) <= 0.05)) for w, g in sub.groupby("Window_Type")}
    return _fmt_rows(inf, firm_fdr, model, region, _ff_notes)


def market_model_rows(cs: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """把主设定（本币市场模型）结果整理成同一格式 / Format the main (local-currency market-model) results alike."""
    firm_fdr = {}
    for w in cs["Window_Type"].astype(str):
        sub = ev[(ev["Window_Type"].eq(w)) & ev["Model_Status"].eq("OK")]
        firm_fdr[w] = int(sub["FDR_Reject_5pct"].fillna(False).sum()) if len(sub) else 0
    return _fmt_rows(cs, firm_fdr, "market", "all", lambda w: "Local-currency market model; main specification.")


def run_factor_models(paths: Paths, market_cs: pd.DataFrame, market_events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """生成第 4 章多因子相关的全部表格 / Build all Chapter 4 multi-factor tables."""
    comp = pd.read_csv(paths.company_sample)
    factors = load_factors(paths)
    fx = pd.read_csv(paths.fx, parse_dates=["date"])
    region_map = pd.DataFrame([{"ticker": r["ticker"], "company_name": r["company_name"], "exchange": r["exchange"], "country_or_region": r["country_or_region"], "currency": r["currency"], **assign_region(r, factors).__dict__} for _, r in comp.iterrows()])

    event_rows, diag_rows = [], []
    std_resids: dict[str, dict[str, pd.Series]] = {"FF3": {}, "FF5": {}}
    for model in ("FF3", "FF5"):
        for _, row in comp.iterrows():
            run = fit_asset(row, model, factors, fx, paths)
            event_rows.extend(run["rows"])
            diag_rows.append(run["diag"])
            if run["diag"]["status"] == "OK" and run["std_resid"] is not None:
                std_resids[model][str(row["ticker"])] = run["std_resid"]
    event_df, diag = pd.DataFrame(event_rows), pd.DataFrame(diag_rows)
    event_df["FDR_Adjusted_p"] = np.nan
    event_df["FDR_Reject_5pct"] = False
    ok = event_df["Model_Status"].eq("OK")
    for _k, idx in event_df[ok].groupby(["model", "Window_Type"]).groups.items():
        q = bh_fdr(event_df.loc[idx, "Firm_p"].to_numpy(float))
        event_df.loc[idx, "FDR_Adjusted_p"] = q
        event_df.loc[idx, "FDR_Reject_5pct"] = q <= 0.05

    parts, by_region = [market_model_rows(market_cs, market_events)], []
    regions = sorted(event_df.loc[ok, "factor_region_raw"].dropna().unique())
    for model in ("FF3", "FF5"):
        parts.append(infer_aggregate(event_df, std_resids[model], model, "all"))
        for reg in regions:
            part = infer_aggregate(event_df, std_resids[model], model, reg)
            if not part.empty:
                by_region.append(part)
    agg = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    order = {n: i for i, (n, _, _) in enumerate(TEST_WINDOWS)}
    agg = agg.assign(_m=agg["model"].map({"market": 0, "FF3": 1, "FF5": 2}), _o=agg["Window_Type"].map(order)).sort_values(["_m", "_o"]).drop(columns=["_m", "_o"])
    by_region_df = pd.concat(by_region, ignore_index=True) if by_region else pd.DataFrame()

    t44 = agg[agg["Window_Type"].isin(TABLE4_WINDOWS)][["model", "Window_Type", "window", "N_firms", "CAAR", "ADJBMP_p", "firm_FDR", "MDE_ADJBMP", "MDE_plainCS", "Rbar_Estimation_Residual_Correlation", "notes"]]

    checks = []
    for model in ("FF3", "FF5"):
        okd = diag[diag["model"].eq(model) & diag["status"].eq("OK")]
        checks += [
            {"check": f"{model}_ok_assets", "value": int(len(okd)), "status": "PASS" if len(okd) else "FAIL"},
            {"check": f"{model}_median_r2", "value": float(okd["r2"].median()) if len(okd) else np.nan, "status": "INFO"},
            {"check": f"{model}_total_fdr_rejections", "value": int((event_df["model"].eq(model) & event_df["FDR_Reject_5pct"].eq(True)).sum()), "status": "INFO"},
        ]
    checks.append({"check": "event_rows_not_ok", "value": int(event_df["Model_Status"].ne("OK").sum()), "status": "INFO"})
    return {
        "ch4_ff_factor_region_assignment": region_map,
        "ch4_ff_firm_level_car": event_df,
        "ch4_ff_regression_diagnostics": diag,
        "ch4_ff_caar_all_windows": agg,
        "appendix_ff_caar_by_region": by_region_df,
        "table4_4_multifactor_robustness": t44.reset_index(drop=True),
        "ch4_ff_validation_checks": pd.DataFrame(checks),
    }
