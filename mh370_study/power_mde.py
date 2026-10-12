"""第 8 章：统计功效与最小可检测效应（表 8-1）。
Chapter 8: statistical power and minimum detectable effects (Table 8-1).

MDE(80%) = (z_0.975 + z_0.80) × SE ≈ 2.80 × SE。

判定规则 / Verdict rule
* |观测值| / SE ≥ 1.96 → SIGNIFICANT_EFFECT（显著效应）/ significant effect;
* 否则 MDE ≤ 合理效应 → INFORMATIVE_NULL（有信息的零结果）/ otherwise MDE ≤ plausible → informative null;
* 否则 → UNDERPOWERED（功效不足）/ otherwise → underpowered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .contagion_tests import DccConfig, DccFdrEngine
from .data_alignment import DataAligner
from .stats_utils import Z_80_POWER, event_position, placebo_centers

ALPHA, POWER = 0.05, 0.80
# 依据文献设定的“合理效应量” / Literature-based plausible effect sizes
PLAUSIBLE = {
    "agg_CAR": 0.03,
    "exposure_per_rank": 0.01,
    "size_diff_CAR": 0.03,
    "size_diff_logAVR": 0.10,
    "dcc_delta_rho": 0.10,
    "var_true_rate_99": 0.015,
}
# 代表性 DCC 资产对：区域内高相关 / 跨区域低相关
# Representative DCC pairs: high-correlation intra-region / low-correlation cross-region
DCC_POWER_PAIRS = [("0670.HK", "0753.HK", "intra_region_high_corr"), ("5099.KL", "MUV2.DE", "cross_region_low_corr")]


def kupiec_p(failures: int, n: int, alpha: float) -> float:
    """Kupiec 检验 p 值 / Kupiec test p-value."""
    if failures == 0:
        return float(stats.chi2.sf(-2 * n * np.log(1 - alpha), 1))
    if failures == n:
        return float(stats.chi2.sf(-2 * n * np.log(alpha), 1))
    phat = failures / n
    lr = -2 * (failures * np.log(alpha) + (n - failures) * np.log(1 - alpha) - failures * np.log(phat) - (n - failures) * np.log(1 - phat))
    return float(stats.chi2.sf(lr, 1))


def verdict(observed: float, se: float, mde: float, plausible: float) -> str:
    """功效判定 / Power verdict."""
    if not np.isfinite(mde):
        return "NA"
    if np.isfinite(observed) and np.isfinite(se) and se > 0 and abs(observed) / se >= stats.norm.ppf(1 - ALPHA / 2):
        return "SIGNIFICANT_EFFECT"
    return "INFORMATIVE_NULL" if mde <= plausible else "UNDERPOWERED"


def build_power_table(tables: dict[str, pd.DataFrame], residuals: pd.DataFrame, event_date: str, alignment_for) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(target, estimator, observed, se, mde, plausible, n, note=""):
        rows.append({"target": target, "estimator": estimator, "n_or_neff": round(float(n), 1), "observed": observed, "SE": se, "MDE_80pct": mde, "plausible": plausible,
                     "MDE_over_plausible": mde / plausible if np.isfinite(mde) else np.nan, "verdict": verdict(observed, se, mde, plausible), "note": note})

    resid = tables["ch4_estimation_window_std_residuals"]
    n_assets = len([c for c in resid.columns if c != "estimation_residual_position"])
    cs = tables["table4_1_aggregate_caar_tests"].set_index("Window_Type")
    rbar = float(cs["Rbar_Estimation_Residual_Correlation"].iloc[0])
    n_eff = n_assets / (1 + (n_assets - 1) * rbar)  # 有效独立公司数 / effective number of independent firms

    # A. 聚合 CAR / aggregate CAR
    for window in ("long", "post120"):
        if window not in cs.index:
            continue
        r = cs.loc[window]
        mean_car = float(r["Mean_CAR"])
        base_note = f"|observed|={abs(mean_car):.4f}"
        add(f"aggregate_CAR_{window}", "Plain_CS", mean_car, float(r["SE_CAR_plain"]), Z_80_POWER * float(r["SE_CAR_plain"]), PLAUSIBLE["agg_CAR"], n_assets, base_note)
        se_bmp = abs(mean_car / float(r["BMP_t"])) if float(r["BMP_t"]) != 0 else np.nan
        add(f"aggregate_CAR_{window}", "BMP", mean_car, se_bmp, Z_80_POWER * se_bmp, PLAUSIBLE["agg_CAR"], n_assets, base_note + "; SE backed out from the standardised BMP t (approximate CAR scale)")
        add(f"aggregate_CAR_{window}", "ADJ_BMP", mean_car, float(r["SE_CAR_KP"]), Z_80_POWER * float(r["SE_CAR_KP"]), PLAUSIBLE["agg_CAR"], n_eff, base_note + "; SE = KP-inflated cross-sectional SE on CAR scale")

    # B. 暴露梯度 / exposure gradient
    exp = tables["table4_2_exposure_gradient_hc3"]
    r = exp[exp["term"] == "exposure_rank"]
    if len(r):
        se = float(r["se_HC3"].iloc[0])
        add("exposure_gradient_long", "HC3_slope_per_rank", float(r["coef"].iloc[0]), se, Z_80_POWER * se, PLAUSIBLE["exposure_per_rank"], n_assets,
            f"direct-vs-global (x3 ranks): MDE={3 * Z_80_POWER * se:.4f} vs plausible {3 * PLAUSIBLE['exposure_per_rank']:.3f}")

    # C. 规模调节 / size moderation (logAVR120)
    si = tables["table6_4_size_interaction_logavr120"]
    for term, label in (("size_large", "size_main_logAVR120"), ("size_x_exposure", "size_x_exposure_logAVR120")):
        rr = si[si["term"] == term]
        if len(rr):
            se = float(rr["se_HC3"].iloc[0])
            add("size_moderation_AVR", label, float(rr["coef"].iloc[0]), se, Z_80_POWER * se, PLAUSIBLE["size_diff_logAVR"], n_assets)

    # C2. 渠道内规模对照 / within-channel size contrasts (CAR)
    ev = tables["ch4_firm_level_ar_car"]
    long = ev[(ev["Window_Type"] == "long") & (ev["Model_Status"] == "OK")]
    for channel in ("airport", "tourism_hotel"):
        sub = long[long["industry_group"] == channel]
        large = sub[sub["company_size_group"] == "large"]["Actual_CAR"].dropna().to_numpy()
        small = sub[sub["company_size_group"] == "mid_small"]["Actual_CAR"].dropna().to_numpy()
        if len(large) >= 2 and len(small) >= 2:
            df = len(large) + len(small) - 2
            sp = np.sqrt(((len(large) - 1) * large.var(ddof=1) + (len(small) - 1) * small.var(ddof=1)) / df)
            se = sp * np.sqrt(1 / len(large) + 1 / len(small))
            tfac = float(stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(POWER, df))
            add(f"size_contrast_CAR_within_{channel}", f"two_sample_t(n={len(large)}v{len(small)})", float(large.mean() - small.mean()), se, tfac * se, PLAUSIBLE["size_diff_CAR"], len(large) + len(small))

    # D. 聚合特质波动放大 / aggregate idiosyncratic volatility amplification
    v = tables["table6_1_idiosyncratic_volatility_avr"]["logAVR_120"].dropna().to_numpy()
    if len(v) >= 3:
        se = v.std(ddof=1) / np.sqrt(len(v))
        add("aggregate_logAVR_120", "one_sample_mean", float(v.mean()), se, Z_80_POWER * se, PLAUSIBLE["size_diff_logAVR"], len(v), "tests whether mean idiosyncratic vol amplification != 0")

    # E. DCC 相关变化：安慰剂零分布离散度 / DCC shift: dispersion of the placebo null
    for pa, pb, kind in DCC_POWER_PAIRS:
        try:
            mode = alignment_for(pa, pb)
            aligned = DataAligner.pair_align(residuals[pa], residuals[pb], mode=mode)
            rho = DccFdrEngine(DccConfig(event_date=event_date, alignment_mode=mode)).fit_pair(aligned["left"], aligned["right"])[0]
            ec = event_position(aligned.index, event_date)
            L = 60
            cands = placebo_centers(n=len(rho), event_center=ec, lo=-L, hi=L - 1, valid_start=L, valid_stop=len(rho) - L, base_block=(-60, 61), confounding_positions=(), confounding_block=0)
            diffs = np.array([rho[c : c + L].mean() - rho[c - L : c].mean() for c in cands])
            sd_null = float(diffs.std(ddof=1))
            observed = float(rho[ec : ec + L].mean() - rho[ec - L : ec].mean())
            add(f"dcc_shift_{kind}", f"placebo_diff[{pa}|{pb}]", observed, sd_null, Z_80_POWER * sd_null, PLAUSIBLE["dcc_delta_rho"], len(rho), f"mode={mode}, placebo windows={len(cands)}")
        except Exception as exc:
            add(f"dcc_shift_{kind}", f"[{pa}|{pb}]", np.nan, np.nan, np.nan, PLAUSIBLE["dcc_delta_rho"], 0, f"FAILED:{exc}")

    # F. 99% VaR 的 Kupiec 功效（二项模拟）/ Kupiec power for 99% VaR (binomial simulation)
    vb = tables["table7_2_var_backtests"]
    oos = int(vb[vb["VaR_Level"] == 0.99]["OOS_T"].median())
    rng = np.random.default_rng(3702014)
    rate = PLAUSIBLE["var_true_rate_99"]
    power = np.mean([kupiec_p(int(rng.binomial(oos, rate)), oos, 0.01) < 0.05 for _ in range(3000)])
    rows.append({"target": "VaR_99_coverage", "estimator": f"Kupiec(OOS_T={oos})", "n_or_neff": oos, "observed": float(vb[vb["VaR_Level"] == 0.99]["Kupiec_p"].median()),
                 "SE": np.nan, "MDE_80pct": np.nan, "plausible": rate, "MDE_over_plausible": np.nan, "verdict": "WELL_POWERED" if power >= 0.80 else "UNDERPOWERED",
                 "note": f"power to detect true 1.5% vs nominal 1% = {power:.2f}"})
    return pd.DataFrame(rows)
