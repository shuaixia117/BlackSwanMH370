"""方法实现测试：安慰剂排除、KP 标准误、功效判定、窗口越界、对齐规则与数值实现。
Method tests: placebo exclusion, KP standard error, power verdicts, window bounds,
alignment rules and numerical implementations.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from mh370_study.contagion_tests import ContagionConfig, DccConfig, DccFdrEngine, ForbesRigobonEngine
from mh370_study.event_study import EventStudyConfig, EventStudyEngine
from mh370_study.garch_var import GarchVarConfig, GarchVarEngine
from mh370_study.power_mde import verdict
from mh370_study.stats_utils import bh_fdr, kp_inflation, placebo_centers


def _matrix_dcc_filter(z, a, b, qbar):
    """矩阵形式的 DCC 递推（参照实现）/ Matrix-form DCC recursion (reference implementation)."""
    q = qbar.copy()
    out = []
    for t in range(len(z)):
        outer = np.outer(z[t - 1], z[t - 1]) if t > 0 else qbar
        q = (1 - a - b) * qbar + a * outer + b * q
        r = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        out.append(float(np.clip(r[0, 1], -0.999, 0.999)))
    return np.asarray(out)


def test_scalar_dcc_filter_matches_matrix_recursion():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(800, 2))
    z[:, 1] = 0.5 * z[:, 0] + 0.8 * z[:, 1]
    qbar = np.cov(z.T)
    np.testing.assert_array_equal(DccFdrEngine._dcc_filter(z, 0.04, 0.93, qbar), _matrix_dcc_filter(z, 0.04, 0.93, qbar))


def test_matrix_bootstrap_draws_equal_sequential_draws():
    r = np.random.default_rng(5).normal(size=181)
    g1, g2 = np.random.default_rng(1), np.random.default_rng(1)
    seq = np.array([g1.choice(r, size=36, replace=True).sum() for _ in range(500)])
    vec = g2.choice(r, size=(500, 36), replace=True).sum(axis=1)
    np.testing.assert_array_equal(seq, vec)


def test_bh_fdr_matches_statsmodels_with_ties():
    p = np.array([0.01, 0.01, 0.04, 0.2, np.nan, 0.04, 1.0])
    valid = ~np.isnan(p)
    np.testing.assert_allclose(bh_fdr(p)[valid], multipletests(p[valid], method="fdr_bh")[1])


def test_garch_cache_returns_identical_paths():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.standard_t(6, 500) * 0.01, index=pd.bdate_range("2010-01-01", periods=500))
    eng = GarchVarEngine(GarchVarConfig(distributions=("normal",), oos_burn_in=300, oos_refit_frequency=50))
    first = eng._rolling_oos_forecasts(x, (0.05, 0.01))
    again = eng._rolling_oos_forecasts(x, (0.05, 0.01))
    fresh = GarchVarEngine(eng.config)._rolling_oos_forecasts(x, (0.05, 0.01))
    np.testing.assert_array_equal(first[0][0.05], again[0][0.05])
    np.testing.assert_array_equal(first[0][0.01], fresh[0][0.01])


def test_placebo_windows_never_overlap_event_or_mh17():
    centers = placebo_centers(n=1200, event_center=550, lo=1, hi=120, valid_start=1, valid_stop=1080, base_block=(-40, 41), confounding_positions=(640,), confounding_block=5)
    assert len(centers) > 0
    assert not any((c + 1 <= 670) and (c + 120 >= 551) for c in centers)  # 真实事件窗 / true event window [551, 670]
    assert not any((c + 1 <= 645) and (c + 120 >= 635) for c in centers)  # MH17 ± 5


def test_long_window_placebo_pool_is_smaller_than_short_window_pool():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2012-01-02", periods=1200)
    z = pd.DataFrame(rng.normal(size=(1200, 2)), index=idx, columns=["A", "B"])
    out = DccFdrEngine(DccConfig(event_date=str(idx[550].date()), alignment_mode="contemporaneous", placebo_replications=5000)).run(z, [("A", "B")])
    draws = out.set_index("Window_Type")["Placebo_N_draws"]
    assert draws["long"] < draws["short"]
    fr = ForbesRigobonEngine(ContagionConfig(event_date=str(idx[550].date()), windows=(("w60", 60),), placebo_replications=5000)).run(z * 0.01, [("A", "B")])
    assert fr["Placebo_N_draws"].iloc[0] > 30


def test_kp_se_is_inflated_and_on_car_scale():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2012-01-02", periods=460)
    common = rng.normal(0, 0.004, len(idx))
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    assets = pd.DataFrame({f"A{i}": market.to_numpy() + common + rng.normal(0, 0.01, len(idx)) for i in range(12)}, index=idx)
    eng = EventStudyEngine(EventStudyConfig(event_date=str(idx[300].date()), bootstrap_replications=10, placebo_replications=10))
    row = eng.cross_sectional_inference(eng.run(assets, market), 181).set_index("Window_Type").loc["long"]
    assert row["Rbar_Estimation_Residual_Correlation"] > 0
    assert row["SE_CAR_KP"] > row["SE_CAR_plain"]
    assert np.isclose(row["SE_CAR_KP"] / row["SE_CAR_plain"], kp_inflation(int(row["N"]), row["Rbar_Estimation_Residual_Correlation"]))


def test_power_verdict_rules():
    assert verdict(-0.156, 0.033, 0.094, 0.10) == "SIGNIFICANT_EFFECT"
    assert verdict(0.01, 0.033, 0.094, 0.10) == "INFORMATIVE_NULL"
    assert verdict(0.01, 0.050, 0.140, 0.10) == "UNDERPOWERED"


def test_truncated_post_window_is_flagged():
    idx = pd.bdate_range("2012-01-02", periods=330)
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    asset = (market + rng.normal(0, 0.01, len(idx))).to_frame("A")
    out = EventStudyEngine(EventStudyConfig(event_date=str(idx[250].date()))).volatility_response(asset, market)
    assert out.loc[0, "Model_Status"].startswith("VOL_RESPONSE_FAILED")


def test_region_aware_dcc_uses_rule():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2012-01-02", periods=400)
    z = pd.DataFrame(rng.normal(size=(400, 2)), index=idx, columns=["A", "B"])
    out = DccFdrEngine(DccConfig(event_date=str(idx[250].date()), placebo_replications=50)).run(z, [("A", "B")], alignment_for=lambda a, b: "contemporaneous")
    assert set(out["Alignment_Mode"]) == {"contemporaneous"}
