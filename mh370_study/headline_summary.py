"""第 9 章：全文主结果汇总（表 9-1），结论文字由结果自动生成。
Chapter 9: headline results (Table 9-1); verdict text is generated from the estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mde(power: pd.DataFrame, target_substring: str, estimator: str | None = None) -> float:
    if power.empty:
        return np.nan
    m = power[power["target"].astype(str).str.contains(target_substring, na=False)]
    if estimator is not None:
        m = m[m["estimator"].astype(str).eq(estimator)]
    return float(m["MDE_80pct"].iloc[0]) if len(m) else np.nan


def build_headline(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    get = lambda k: tables.get(k, pd.DataFrame())  # noqa: E731
    cs, power, fr = get("table4_1_aggregate_caar_tests"), get("table8_1_power_mde"), get("table5_1_forbes_rigobon")
    ml, did, expo = get("table7_3_multilevel_var"), get("table6_2_volatility_did"), get("table4_2_exposure_gradient_hc3")
    rows: list[dict[str, object]] = []

    # 1. 价格冲击 / price impact
    if not cs.empty and "long" in set(cs["Window_Type"].astype(str)):
        r = cs.set_index("Window_Type").loc["long"]
        est, p, mde = float(r["Mean_CAR"]), float(r["ADJ_BMP_p"]), _mde(power, "aggregate_CAR_long", "ADJ_BMP")
        sig = "significant under ADJ-BMP" if p < 0.05 else ("significant only before cross-sectional-correlation correction" if float(r["Plain_CS_p"]) < 0.05 else "not significant")
        rows.append({"result": "Aggregate CAR (long, [-5,+30])", "estimate": round(est, 4), "primary_inference": f"ADJ-BMP p={p:.3f}", "MDE_80pct": round(mde, 4),
                     "verdict": f"{'decline' if est < 0 else 'increase'} of {abs(est):.2%}; {sig}; {'underpowered' if mde > 0.03 else 'adequately powered'} for a 3% effect (MDE={mde:.3f})"})

    # 2. 传染 / contagion
    if not fr.empty:
        clean = fr[fr["Window"].astype(str).isin(["w40", "w60"]) & fr["Model_Status"].astype(str).str.startswith("OK")]
        rej = int(clean["FDR_Reject_contagion_5pct"].sum())
        rows.append({"result": "Contagion (Forbes-Rigobon, MH17-free windows <=+60)", "estimate": round(float(clean["diff_FRadj"].mean()), 4) if len(clean) else np.nan,
                     "primary_inference": f"FDR-reject contagion {rej}/{len(clean)} (w90 spanning MH17 reported separately)", "MDE_80pct": round(_mde(power, "dcc_shift"), 4),
                     "verdict": f"{rej}/{len(clean)} pair-windows show contagion after FDR" if rej else "no contagion in MH17-free windows; interdependence, not contagion"})

    # 3. 持续特质波动 / persistent idiosyncratic volatility
    if not did.empty:
        r = did[did["outcome"] == "idiosyncratic_logAVR_DiD"]
        if len(r):
            est, pp, nt, mde = float(r["DiD_treated_minus_control"].iloc[0]), float(r["perm_p"].iloc[0]), int(r["n_treated"].iloc[0]), _mde(power, "logAVR")
            v = "MH370-specific idiosyncratic volatility amplification" if pp < 0.05 and est > 0 else ("treated volatility fell relative to control" if pp < 0.05 else "no MH370-specific amplification after differencing the 2013-14 regime")
            rows.append({"result": "Persistent idiosyncratic volatility (DiD: treated vs control)", "estimate": round(est, 4), "primary_inference": f"perm p={pp:.3f}; treated n={nt} (MDE={mde:.3f})", "MDE_80pct": round(mde, 4), "verdict": v})

    # 4. 尾部模型风险 / tail model risk
    if not ml.empty:
        ok = ml[ml["Model_Status"].astype(str).str.startswith("OK")]
        rej = int(ok["Model_Status"].astype(str).str.contains("MULTILEVEL_COVERAGE_WARNING").sum())
        rows.append({"result": "Tail model risk (multinomial multi-level VaR, Nass-corrected)", "estimate": f"{rej}/{len(ok)} firms reject", "primary_inference": "Kratz multinomial chi-square + Nass", "MDE_80pct": np.nan,
                     "verdict": f"standard GARCH under-covers tails for {rej}/{len(ok)} firms" if rej else "multi-level VaR coverage not rejected"})

    # 5. 暴露梯度 / exposure gradient
    if not expo.empty:
        r = expo[expo["term"] == "exposure_rank"]
        mde = _mde(power, "exposure_gradient")
        v = ("significant exposure gradient" if len(r) and float(r["p"].iloc[0]) < 0.05 else "no detectable exposure gradient") + ("; underpowered" if mde > 0.01 else "")
        rows.append({"result": "Exposure gradient & size moderation (supplementary)", "estimate": round(float(r["coef"].iloc[0]), 4) if len(r) else np.nan, "primary_inference": "HC3 regression / permutation", "MDE_80pct": round(mde, 4), "verdict": v})
    return pd.DataFrame(rows)
