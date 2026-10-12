"""核心计量方法单元测试。
Unit tests for the core econometric methods.
"""
import numpy as np
import pandas as pd
from arch.univariate.distribution import StudentsT
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

from mh370_study.data_alignment import AlignmentConfig, DataAligner, common_columns
from mh370_study.contagion_tests import ContagionConfig, DccConfig, DccFdrEngine, ForbesRigobonEngine
from mh370_study.event_study import EventStudyConfig, EventStudyEngine
from mh370_study.garch_var import GarchVarConfig, GarchVarEngine
from mh370_study.run_all import leave_one_out_market


def test_bh_fdr_matches_statsmodels():
    p = np.array([0.001, 0.02, 0.03, 0.2, np.nan, 0.8])
    got = DccFdrEngine.bh_fdr(p)
    valid = ~np.isnan(p)
    expected = np.full_like(p, np.nan)
    expected[valid] = multipletests(p[valid], method="fdr_bh")[1]
    np.testing.assert_allclose(got, expected)


def test_kupiec_zero_failure_boundary_uses_finite_lr():
    returns = np.zeros(100)
    var = np.ones(100)
    failures, lr, p = GarchVarEngine.kupiec_test(returns, var, 0.01)
    assert failures == 0
    assert np.isfinite(lr)
    assert np.isclose(lr, -2 * 100 * np.log(0.99))
    assert np.isclose(p, stats.chi2.sf(lr, 1))


def test_market_model_matches_statsmodels_fixture():
    x = pd.Series(np.linspace(-0.02, 0.03, 30))
    y = 0.001 + 1.35 * x
    alpha, beta = EventStudyEngine._fit_market_model(y, x)
    res = OLS(y.to_numpy(), add_constant(x.to_numpy())).fit()
    assert np.isclose(alpha, res.params[0])
    assert np.isclose(beta, res.params[1])


def test_corrado_and_cross_sectional_inference_are_distinct():
    rng = np.random.default_rng(123)
    idx = pd.bdate_range("2013-01-01", periods=180)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    assets = pd.DataFrame(index=idx)
    for i in range(6):
        assets[f"A{i}"] = 0.0002 + (0.8 + i * 0.1) * market + rng.normal(0, 0.01, len(idx))
        assets.loc[idx[130], f"A{i}"] += 0.03
    engine = EventStudyEngine(
        EventStudyConfig(
            event_date=str(idx[130].date()),
            estimation_window=(-120, -10),
            test_windows=(("short", -1, 1),),
            bootstrap_replications=999,
            placebo_replications=999,
        )
    )
    results = engine.run(assets, market)
    cross = engine.cross_sectional_inference(results, estimation_window_length=111)
    assert results["SCAR"].notna().any()
    assert "Est_Std_Residuals_JSON" not in results.columns
    assert {"Plain_CS_t", "Patell_z", "BMP_t", "ADJ_BMP_t"}.issubset(cross.columns)
    row = cross[cross["Window_Type"] == "short"].iloc[0]
    assert not np.isclose(row["Patell_z"], row["Plain_CS_t"])
    assert not np.isclose(row["BMP_t"], row["Plain_CS_t"])


def test_student_t_quantile_is_arch_standardized():
    expected = StudentsT().ppf(0.01, [5.0])
    scipy_unstandardized_reference = -3.36493
    assert np.isclose(expected, -2.606463, atol=1e-3)
    assert abs(scipy_unstandardized_reference) > abs(expected)


def test_skewt_quantile_via_engine_does_not_crash_and_matches_arch():
    rng = np.random.default_rng(0)
    e = rng.standard_t(5, 1500) - 0.4 * np.abs(rng.standard_t(5, 1500))
    r = pd.Series(0.0005 + 0.01 * e, index=pd.bdate_range("2010-01-01", periods=1500))
    engine = GarchVarEngine(GarchVarConfig())
    selected = engine._fit_one(r, "skewt")
    names = selected["res"].model.distribution.parameter_names()
    assert names == ["eta", "lambda"]
    q = engine._distribution_quantile(selected, 0.01)
    ref = float(selected["res"].model.distribution.ppf(0.01, [float(selected["res"].params[name]) for name in names]))
    assert np.isfinite(q)
    assert np.isclose(q, ref)


def test_corrado_is_calibrated_under_the_null():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2012-01-01", periods=200)
    reject = 0
    trials = 200
    for _ in range(trials):
        market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
        assets = pd.DataFrame({"A": 0.0002 + market + rng.normal(0, 0.01, len(idx))}, index=idx)
        engine = EventStudyEngine(
            EventStudyConfig(
                event_date=str(idx[150].date()),
                estimation_window=(-120, -10),
                test_windows=(("short", -1, 1),),
                bootstrap_replications=2,
                placebo_replications=2,
            )
        )
        pvalues = engine.run(assets, market)["Corrado_p"].dropna()
        reject += int((pvalues < 0.05).any())
    assert reject / trials < 0.15


def test_oos_var_coverage_is_near_nominal():
    rng = np.random.default_rng(3)
    n = 1200
    omega, alpha, beta = 0.02, 0.08, 0.90
    h = np.empty(n)
    x = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    x[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, n):
        h[t] = omega + alpha * x[t - 1] ** 2 + beta * h[t - 1]
        x[t] = np.sqrt(h[t]) * rng.standard_normal()
    returns = pd.DataFrame({"A": x * 0.01}, index=pd.bdate_range("2008-01-01", periods=n))
    engine = GarchVarEngine(
        GarchVarConfig(
            distributions=("normal",),
            var_levels=(0.95,),
            oos_burn_in=300,
            oos_refit_frequency=50,
        )
    )
    _table_a, table_b, _residuals = engine.run(returns)
    row = table_b.iloc[0]
    rate = row["Failures"] / row["OOS_T"]
    assert 0.025 < rate < 0.085


def test_raw_vs_winsor_alignment_shapes():
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
        },
        index=pd.bdate_range("2010-01-01", periods=400),
    )
    raw, _diag_raw = DataAligner(AlignmentConfig(winsorize=False)).align_from_prices(prices)
    wins, _diag_wins = DataAligner(AlignmentConfig(winsorize=True)).align_from_prices(prices)
    assert list(raw.columns) == list(wins.columns)
    assert raw.shape == wins.shape
    assert (raw.abs().max() >= wins.abs().max()).all()


def test_oos_var_outputs_oos_t_and_unique_table_a():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-01", periods=170)
    returns = pd.DataFrame({"A": rng.standard_t(8, len(idx)) * 0.01}, index=idx)
    engine = GarchVarEngine(
        GarchVarConfig(
            distributions=("normal", "studentst"),
            var_levels=(0.95,),
            oos_burn_in=80,
            oos_refit_frequency=20,
        )
    )
    table_a, table_b, residuals = engine.run(returns)
    assert table_a["Ticker"].nunique() == len(table_a)
    assert "OOS_T" in table_b.columns
    assert table_b["OOS_T"].notna().any()
    assert not residuals.empty


def test_dcc_reports_standard_errors_and_placebo_fdr_family():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2012-01-01", periods=180)
    z1 = rng.normal(size=len(idx))
    z2 = 0.4 * z1 + rng.normal(size=len(idx)) * 0.9
    residuals = pd.DataFrame({"A": z1, "B": z2}, index=idx)
    engine = DccFdrEngine(DccConfig(event_date=str(idx[120].date()), alignment_mode="overlap2", placebo_replications=999, bootstrap_se_reps=35, bootstrap_block=20))
    out = engine.run(residuals, [("A", "B")])
    assert {
        "Naive_ttest_p_uncorrected",
        "DCC_a_SE_QMLE",
        "DCC_b_SE_QMLE",
        "DCC_a_plus_b_SE_QMLE",
        "DCC_a_SE_boot",
        "DCC_b_SE_boot",
        "DCC_a_plus_b_SE_boot",
        "FDR_Family",
        "FDR_Family_Size",
    }.issubset(out.columns)
    assert "Raw_p" not in out.columns
    assert out[["DCC_a_SE_boot", "DCC_b_SE_boot", "DCC_a_plus_b_SE_boot"]].notna().all().all()
    assert out["FDR_Family"].isin(["short", "medium", "long"]).all()


def test_dcc_placebo_invariant_to_bootstrap_toggle_long_sample():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2008-01-01", periods=1800)
    z1 = rng.normal(size=len(idx))
    z2 = 0.4 * z1 + 0.9 * rng.normal(size=len(idx))
    residuals = pd.DataFrame({"A": z1, "B": z2}, index=idx)
    event_date = str(idx[900].date())
    off = DccFdrEngine(DccConfig(event_date=event_date, alignment_mode="overlap2", bootstrap_se_reps=0)).run(residuals, [("A", "B")])
    on = DccFdrEngine(DccConfig(event_date=event_date, alignment_mode="overlap2", bootstrap_se_reps=50)).run(residuals, [("A", "B")])
    merged = off.merge(on, on=["Pair", "Window_Type"], suffixes=("_off", "_on"))
    np.testing.assert_allclose(merged["Placebo_p_off"].to_numpy(), merged["Placebo_p_on"].to_numpy())


def test_event_placebo_invariant_to_bootstrap_toggle():
    rng = np.random.default_rng(12)
    idx = pd.bdate_range("2012-01-01", periods=420)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    assets = pd.DataFrame({"A": 0.0001 + market + rng.normal(0, 0.01, len(idx))}, index=idx)
    event_date = str(idx[250].date())
    low_boot = EventStudyEngine(EventStudyConfig(event_date=event_date, bootstrap_replications=2)).run(assets, market)
    high_boot = EventStudyEngine(EventStudyConfig(event_date=event_date, bootstrap_replications=50)).run(assets, market)
    merged = low_boot.merge(high_boot, on=["Ticker", "Window_Type"], suffixes=("_low", "_high"))
    np.testing.assert_allclose(merged["Placebo_p_low"].to_numpy(), merged["Placebo_p_high"].to_numpy())


def test_placebo_resolution_and_confounding_exclusion_are_reported():
    rng = np.random.default_rng(13)
    idx = pd.bdate_range("2012-01-01", periods=360)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    assets = pd.DataFrame({"A": 0.0001 + market + rng.normal(0, 0.01, len(idx))}, index=idx)
    event_date = str(idx[220].date())
    base = EventStudyEngine(EventStudyConfig(event_date=event_date, placebo_replications=999)).run(assets, market)
    conf = EventStudyEngine(
        EventStudyConfig(
            event_date=event_date,
            placebo_replications=999,
            confounding_dates=(str(idx[80].date()),),
            confounding_block=10,
        )
    ).run(assets, market)
    assert {"Placebo_N_draws", "Placebo_Min_Resolvable_p"}.issubset(base.columns)
    assert base["Placebo_N_draws"].notna().all()
    assert (conf["Placebo_N_draws"] < base["Placebo_N_draws"]).any()
    np.testing.assert_allclose(base["Placebo_Min_Resolvable_p"], 1.0 / (base["Placebo_N_draws"] + 1))


def test_dcc_pre_event_targeting_does_not_use_post_event_data_for_pre_rho():
    rng = np.random.default_rng(14)
    idx = pd.bdate_range("2010-01-01", periods=500)
    z1 = rng.normal(size=len(idx))
    z2 = 0.3 * z1 + rng.normal(size=len(idx))
    residuals = pd.DataFrame({"A": z1, "B": z2}, index=idx)
    event_center = 250
    changed = residuals.copy()
    changed.iloc[event_center + 20 :, 1] = changed.iloc[event_center + 20 :, 1] * 8 + 5
    engine = DccFdrEngine(DccConfig(event_date=str(idx[event_center].date()), correlation_targeting="pre_event"))
    rho_base, *_ = engine.fit_pair(residuals["A"], residuals["B"], fit_slice=slice(0, event_center))
    rho_changed, *_ = engine.fit_pair(changed["A"], changed["B"], fit_slice=slice(0, event_center))
    np.testing.assert_allclose(rho_base[:event_center], rho_changed[:event_center], atol=1e-10)


def test_leave_one_out_market_excludes_self():
    idx = pd.bdate_range("2020-01-01", periods=5)
    panel = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [2, 2, 2, 2, 2], "C": [5, 4, 3, 2, 1]}, index=idx, dtype=float)
    market, mapping = leave_one_out_market(panel)
    assert mapping == {"A": "EW_LOO_A", "B": "EW_LOO_B", "C": "EW_LOO_C"}
    np.testing.assert_allclose(market["EW_LOO_A"], panel[["B", "C"]].mean(axis=1))
    np.testing.assert_allclose(market["EW_LOO_B"], panel[["A", "C"]].mean(axis=1))


def test_estimation_window_default_is_210_30_and_runs():
    cfg = EventStudyConfig(event_date="2014-03-10")
    assert cfg.estimation_window == (-210, -30)
    idx = pd.bdate_range("2012-01-02", periods=520)
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    asset = pd.Series(0.0002 + market.to_numpy() + rng.normal(0, 0.01, len(idx)), index=idx)
    out = EventStudyEngine(
        EventStudyConfig(
            event_date=str(idx[300].date()),
            bootstrap_replications=20,
            placebo_replications=20,
        )
    ).run(asset.to_frame("A"), market)
    assert (out["Model_Status"] == "OK").any()
    assert out["Actual_CAR"].notna().any()


def test_volatility_response_detects_postevent_vol_jump():
    idx = pd.bdate_range("2012-01-02", periods=500)
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.008, len(idx)), index=idx, name="MKT")
    abnormal = rng.normal(0, 0.01, len(idx))
    abnormal[300:] = rng.normal(0, 0.03, len(idx) - 300)
    asset = pd.Series(market.to_numpy() + abnormal, index=idx)
    vr = EventStudyEngine(EventStudyConfig(event_date=str(idx[300].date()))).volatility_response(asset.to_frame("A"), market)
    assert vr.loc[0, "AVR_60"] > 1.3
    assert np.isfinite(vr.loc[0, "logAVR_120"])


def test_size_group_contrasts_within_channel():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(8):
        rows.append(
            {
                "Ticker": f"L{i}",
                "industry_group": "airport",
                "company_size_group": "large",
                "exposure_group": "airport_infrastructure_exposure",
                "outcome": -0.02 + rng.normal(0, 0.003),
            }
        )
        rows.append(
            {
                "Ticker": f"S{i}",
                "industry_group": "airport",
                "company_size_group": "mid_small",
                "exposure_group": "airport_infrastructure_exposure",
                "outcome": rng.normal(0, 0.003),
            }
        )
    res = EventStudyEngine.size_group_contrasts(pd.DataFrame(rows), "outcome", channels=("airport",), n_perm=2000)
    row = res[res["scope"] == "within_airport"].iloc[0]
    assert row["diff"] < 0
    assert row["perm_p"] < 0.05


def test_size_interaction_regression_recovers_size():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(40):
        large = i % 2 == 0
        rows.append(
            {
                "Ticker": f"T{i}",
                "company_size_group": "large" if large else "mid_small",
                "industry_group": "tourism_hotel",
                "exposure_group": "tourism_demand_exposure",
                "country_or_region": "United States",
                "logAVR_120": (0.0 if large else 0.30) + rng.normal(0, 0.05),
            }
        )
    out = EventStudyEngine.size_interaction_regression(pd.DataFrame(rows), "logAVR_120").set_index("term")
    assert out.loc["size_large", "coef"] < 0
    assert out.loc["size_large", "p"] < 0.05


def test_core_pairs_filters_to_present():
    from mh370_study.run_all import core_pairs

    got = core_pairs(["0670.HK", "0753.HK", "5099.KL", "ZZZ"])
    assert ("0670.HK", "0753.HK") in got
    assert all(left != "ZZZ" and right != "ZZZ" for left, right in got)


def test_garch_returns_fullsample_residuals():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2010-01-04", periods=900)
    returns = pd.DataFrame({"A": rng.standard_t(7, len(idx)) * 0.01}, index=idx)
    _table_a, _table_b, resid = GarchVarEngine(
        GarchVarConfig(distributions=("normal",), var_levels=(0.95,), oos_burn_in=300, oos_refit_frequency=60)
    ).run(returns)
    assert not resid.empty
    assert resid["A"].dropna().shape[0] > 700


def test_common_columns_handles_winsorized_subset():
    raw = pd.DataFrame(columns=["A", "B", "C"])
    wins = pd.DataFrame(columns=["A", "C"])
    assert common_columns(raw, wins) == ["A", "C"]


def test_pairwise_psd_diagnostic_reports_min_eigenvalue():
    table = pd.DataFrame(
        {
            "Pair": ["A|B", "A|C", "B|C"],
            "Window_Type": ["short", "short", "short"],
            "Mean_Post": [0.2, 0.1, 0.3],
        }
    )
    out = DccFdrEngine.pairwise_psd_diagnostic(table)
    assert {"Window_Type", "N_Assets", "Min_Eigenvalue", "Is_PSD"}.issubset(out.columns)
    assert out.loc[0, "N_Assets"] == 3


def test_native_returns_no_global_dropna(tmp_path):
    idx = pd.bdate_range("2012-01-02", periods=400)
    a_dates = idx
    b_dates = idx[[i for i in range(len(idx)) if i % 7 != 0]]
    for ticker, dates in [("AAA", a_dates), ("BBB", b_dates)]:
        prices = 100 * np.exp(np.cumsum(np.full(len(dates), 0.0005)))
        pd.DataFrame({"date": dates, "adjusted_close": prices, "ticker": ticker}).to_csv(tmp_path / f"{ticker}.csv", index=False)
    returns_frame, char = DataAligner().build_native_returns(tmp_path)
    assert set(returns_frame.columns) == {"AAA", "BBB"}
    assert returns_frame["AAA"].dropna().shape[0] == len(a_dates) - 1
    assert returns_frame["BBB"].dropna().shape[0] == len(b_dates) - 1
    assert set(char["Ticker"]) == {"AAA", "BBB"}


def test_event_study_uses_per_asset_event_center():
    idx = pd.bdate_range("2012-01-02", periods=400)
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    a = pd.Series(0.0002 + market.to_numpy() + rng.normal(0, 0.01, len(idx)), index=idx)
    b_full = pd.Series(0.0002 + 1.1 * market.to_numpy() + rng.normal(0, 0.01, len(idx)), index=idx)
    b_idx = idx[[i for i in range(len(idx)) if i % 5 != 0]]
    frame = pd.concat([a.rename("A"), b_full.loc[b_idx].rename("B")], axis=1)
    engine = EventStudyEngine(
        EventStudyConfig(
            event_date=str(idx[250].date()),
            estimation_window=(-120, -10),
            test_windows=(("short", -1, 1),),
            bootstrap_replications=50,
            placebo_replications=50,
        )
    )
    out = engine.run(frame, market)
    ok = out[out["Model_Status"] == "OK"]
    assert set(ok["Ticker"].unique()) == {"A", "B"}
    assert ok["Actual_CAR"].notna().all()


def test_pair_align_modes():
    idx = pd.bdate_range("2012-01-02", periods=10)
    a = pd.Series(np.arange(10, dtype=float), index=idx)
    b = pd.Series(np.arange(10, dtype=float) * 2, index=idx)
    contemporaneous = DataAligner.pair_align(a, b, mode="contemporaneous")
    assert list(contemporaneous.columns) == ["left", "right"]
    assert len(contemporaneous) == 10
    overlap = DataAligner.pair_align(a, b, mode="overlap2")
    assert len(overlap) == 9
    assert np.isclose(overlap["left"].iloc[0], a.iloc[0] + a.iloc[1])
    assert np.isclose(overlap["right"].iloc[0], b.iloc[0] + b.iloc[1])
    lagged = DataAligner.pair_align(a, b, mode="lag1")
    assert len(lagged) == 9
    assert np.isclose(lagged["left"].iloc[0], a.iloc[0])
    assert np.isclose(lagged["right"].iloc[0], b.iloc[1])


def test_dcc_recovers_cross_calendar_pair_sample():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-04", periods=600)
    z1 = pd.Series(rng.normal(size=len(idx)), index=idx)
    z2 = pd.Series(0.4 * z1.to_numpy() + 0.9 * rng.normal(size=len(idx)), index=idx)
    z2 = z2.iloc[[i for i in range(len(idx)) if i % 6 != 0]]
    residuals = pd.concat([z1.rename("A"), z2.rename("B")], axis=1)
    out = DccFdrEngine(DccConfig(event_date=str(idx[300].date()), alignment_mode="overlap2")).run(residuals, [("A", "B")])
    assert (out["Model_Status"] == "OK").all()
    assert int(out["Pair_N_obs"].iloc[0]) >= 100
    assert out["DCC_a"].notna().all()


def test_cross_sectional_car_regression_recovers_injected_exposure():
    rng = np.random.default_rng(7)
    groups = (
        ["direct_malaysia_china_exposure"] * 5
        + ["asia_pacific_airline_exposure"] * 8
        + ["regional_airline_exposure"] * 7
        + ["global_airline_exposure"] * 10
    )
    rank = {
        "direct_malaysia_china_exposure": 3,
        "asia_pacific_airline_exposure": 2,
        "regional_airline_exposure": 1,
        "global_airline_exposure": 0,
    }
    rows = [
        {
            "Ticker": f"T{i}",
            "Window_Type": "long",
            "Actual_CAR": -0.01 * rank[group] + rng.normal(0, 0.005),
            "Model_Status": "OK",
            "exposure_group": group,
            "industry_group": "airline",
            "company_size_group": "large",
            "country_or_region": "China",
        }
        for i, group in enumerate(groups)
    ]
    out = EventStudyEngine.cross_sectional_car_regression(pd.DataFrame(rows), "long")
    coef = out.set_index("term")
    assert coef.loc["exposure_rank", "coef"] < 0
    assert coef.loc["exposure_rank", "p"] < 0.05


def test_save_figure_dcc_guarded_when_unfittable(tmp_path):
    from mh370_study import figures

    idx = pd.bdate_range("2012-01-02", periods=50)
    residuals = pd.DataFrame(
        {
            "A": np.random.default_rng(1).normal(size=50),
            "B": np.random.default_rng(2).normal(size=50),
        },
        index=idx,
    )
    figures.dcc_path(residuals, tmp_path / "fig.png", str(idx[25].date()))
    assert not (tmp_path / "fig.png").exists()


def test_forbes_rigobon_distinguishes_interdependence_from_contagion():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-04", periods=600)
    event_center = 300
    factor = rng.normal(0, 1, len(idx))
    factor_scale = np.ones(len(idx))
    factor_scale[event_center:] = 2.0
    x = 0.5 * factor * factor_scale + rng.normal(0, 1, len(idx))
    y = 0.5 * factor * factor_scale + rng.normal(0, 1, len(idx))
    returns = pd.DataFrame({"A": x * 0.01, "B": y * 0.01}, index=idx)
    out = ForbesRigobonEngine(ContagionConfig(event_date=str(idx[event_center].date()))).run(returns, [("A", "B")])
    w60 = out[out["Window"] == "w60"].iloc[0]
    assert w60["diff_raw"] > 0.03
    assert w60["diff_FRadj"] < w60["diff_raw"]


def test_multinomial_backtest_flags_fat_tail_model():
    rng = np.random.default_rng(1)
    n = 1500
    levels = (0.95, 0.975, 0.99, 0.995)
    var = {level: np.full(n, abs(stats.norm.ppf(1 - level)) * 0.01) for level in levels}
    fat_tail = rng.standard_t(3, n) * 0.01
    backtest_fat = GarchVarEngine.multinomial_var_backtest(fat_tail, var)
    assert backtest_fat["df"] == 4
    assert backtest_fat["p_nass"] < 0.05
    normal = rng.normal(0, 0.01, n)
    assert GarchVarEngine.multinomial_var_backtest(normal, var)["p_nass"] > 0.01


def test_volatility_response_pre_window_param():
    idx = pd.bdate_range("2012-01-02", periods=500)
    rng = np.random.default_rng(2)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    asset = pd.Series(market.to_numpy() + rng.normal(0, 0.01, len(idx)), index=idx)
    response = EventStudyEngine(EventStudyConfig(event_date=str(idx[300].date()))).volatility_response(
        asset.to_frame("A"),
        market,
        pre=(-180, -61),
    )
    assert "AVR_120" in response.columns
    assert response.loc[0, "Model_Status"] == "OK"


def test_event_time_caar_shape_and_drift():
    idx = pd.bdate_range("2012-01-02", periods=500)
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    abnormal = rng.normal(0, 0.01, len(idx))
    abnormal[300:] -= 0.002
    asset = pd.Series(market.to_numpy() + abnormal, index=idx)
    overall, _groups = EventStudyEngine(EventStudyConfig(event_date=str(idx[300].date()))).event_time_caar(
        asset.to_frame("A"),
        market,
        tau_min=-10,
        tau_max=60,
    )
    assert list(overall.columns) == ["tau", "CAAR", "SE_CAAR", "N"]
    assert overall["CAAR"].iloc[-1] < overall["CAAR"].iloc[0]


def test_did_volatility_recovers_treated_effect():
    idx = pd.bdate_range("2012-01-02", periods=600)
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="MKT")
    columns = {}
    char_rows = []
    for i in range(20):
        treated = i < 8
        abnormal = rng.normal(0, 0.01, len(idx))
        if treated:
            abnormal[300:] = rng.normal(0, 0.02, len(idx) - 300)
        columns[f"T{i}"] = pd.Series(market.to_numpy() + abnormal, index=idx)
        char_rows.append(
            {
                "Ticker": f"T{i}",
                "exposure_group": "direct_malaysia_china_exposure" if treated else "global_airline_exposure",
                "industry_group": "airline",
                "company_size_group": "large",
                "country_or_region": "China",
                "benchmark_ticker": "MKT",
            }
        )
    did, _full = EventStudyEngine(EventStudyConfig(event_date=str(idx[300].date()))).did_volatility_response(
        pd.DataFrame(columns),
        market,
        {f"T{i}": "MKT" for i in range(20)},
        pd.DataFrame(char_rows),
        n_perm=2000,
    )
    row = did[did["outcome"] == "idiosyncratic_logAVR_DiD"].iloc[0]
    assert row["DiD_treated_minus_control"] > 0
    assert row["perm_p"] < 0.05


def test_multinomial_nass_present_and_conservative():
    rng = np.random.default_rng(2)
    n = 1200
    var = {level: np.full(n, abs(stats.norm.ppf(1 - level)) * 0.01) for level in (0.95, 0.975, 0.99, 0.995)}
    backtest = GarchVarEngine.multinomial_var_backtest(rng.standard_t(3, n) * 0.01, var)
    assert {"p_pearson", "p_nass", "nass_c"}.issubset(backtest)
    assert backtest["p_nass"] < 0.05
    assert backtest["nass_c"] <= 1.0 + 1.0e-9


def test_fr_constant_window_returns_none_no_warning():
    x = np.r_[np.zeros(40), np.random.default_rng(0).normal(0, 1, 40)]
    y = np.random.default_rng(1).normal(0, 1, 80)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ForbesRigobonEngine._stat(x, y, 40, 40) is None
        assert not any("invalid value" in str(warning.message) for warning in caught)


def test_confounding_blocks_mh17_from_placebo():
    idx = pd.bdate_range("2013-06-03", periods=400)
    x = pd.Series(np.random.default_rng(0).normal(0, 1, len(idx)), index=idx, name="A")
    y = pd.Series(np.random.default_rng(1).normal(0, 1, len(idx)), index=idx, name="B")
    base_config = ContagionConfig(
        event_date="2014-03-10",
        windows=(("w60", 60),),
        confounding_dates=(),
        confounding_block=5,
    )
    blocked_config = ContagionConfig(
        event_date="2014-03-10",
        windows=(("w60", 60),),
        confounding_dates=("2014-07-17",),
        confounding_block=5,
    )
    no_block = ForbesRigobonEngine(base_config).run(pd.concat([x, y], axis=1), [("A", "B")])
    blocked = ForbesRigobonEngine(blocked_config).run(pd.concat([x, y], axis=1), [("A", "B")])
    assert blocked.iloc[0]["Model_Status"] == "OK"
    assert blocked.iloc[0]["Placebo_N_draws"] < no_block.iloc[0]["Placebo_N_draws"]
