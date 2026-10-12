"""一条命令生成全部章节的结果表与图。
Generate all result tables and figures with a single command.

用法 / Usage:
    python -m mh370_study.run_all
    python -m mh370_study.run_all --run-all-pair-screening   # 附加探索性筛查 / optional exploratory screening
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .config import StudyConfig
from .contagion_tests import ContagionConfig, DccConfig, DccFdrEngine, ForbesRigobonEngine
from .data_alignment import AlignmentConfig, DataAligner
from .event_study import EventStudyConfig, EventStudyEngine
from .factor_models import run_factor_models
from .garch_var import GarchVarConfig, GarchVarEngine
from .headline_summary import build_headline
from .power_mde import build_power_table
from .robustness import RobustnessConfig, RobustnessEngine
from .stats_utils import region_of

# 第 5 章 19 个核心资产对 / The 19 core pairs of Chapter 5
CORE_PAIRS = [
    ("0670.HK", "0753.HK"), ("0753.HK", "1055.HK"), ("0670.HK", "1055.HK"),
    ("0753.HK", "5099.KL"), ("1055.HK", "5099.KL"), ("0670.HK", "5099.KL"),
    ("5099.KL", "AIRPORT.KL"), ("0753.HK", "0293.HK"), ("5099.KL", "C6L.SI"),
    ("0753.HK", "9201.T"), ("5099.KL", "THAI.BK"), ("0753.HK", "003490.KS"),
    ("5099.KL", "MUV2.DE"), ("5099.KL", "SREN.SW"), ("0753.HK", "2318.HK"),
    ("0753.HK", "1339.HK"), ("9201.T", "9202.T"), ("003490.KS", "020560.KS"),
    ("QAN.AX", "AIR.NZ"),
]
AVR_PRE_WINDOWS = [(-180, -61), (-150, -61), (-120, -61), (-60, -1)]

# 结果表说明 / Result-table descriptions: 文件名 → (章节, 中文说明, English description)
TABLE_INDEX = {
    "table3_1_sample_coverage_diagnostics": ("第3章 Ch.3", "样本覆盖诊断", "Sample coverage diagnostics"),
    "table3_3_return_statistics_archlm": ("第3章 Ch.3", "日收益描述统计与 ARCH-LM 检验", "Daily-return statistics and ARCH-LM tests"),
    "ch4_firm_level_ar_car": ("第4章 Ch.4", "逐公司各窗口 CAR 与公司层面检验", "Firm-level CARs and tests by window"),
    "ch4_estimation_window_std_residuals": ("第4章 Ch.4", "估计窗标准化残差", "Estimation-window standardised residuals"),
    "table4_1_aggregate_caar_tests": ("第4章 Ch.4", "聚合 CAAR 与截面检验（Plain-CS / Patell / BMP / ADJ-BMP）", "Aggregate CAAR and cross-sectional tests"),
    "table4_2_exposure_gradient_hc3": ("第4章 Ch.4", "暴露梯度 HC3 截面回归", "Exposure-gradient HC3 cross-sectional regression"),
    "table4_2_size_contrast_long_car": ("第4章 Ch.4", "长窗 CAR 规模对照（置换检验）", "Size contrasts of long-window CAR (permutation)"),
    "table4_3_window_sensitivity": ("第4章 Ch.4", "替代估计窗与扩展事件窗稳健性", "Alternative estimation and extended event windows"),
    "table4_4_multifactor_robustness": ("第4章 Ch.4", "FF3/FF5 多因子稳健性", "FF3/FF5 multi-factor robustness"),
    "ch4_ff_caar_all_windows": ("第4章 Ch.4", "多因子模型全部窗口 CAAR", "Multi-factor CAAR for all windows"),
    "ch4_ff_firm_level_car": ("第4章 Ch.4", "多因子模型逐公司 CAR", "Multi-factor firm-level CARs"),
    "ch4_ff_regression_diagnostics": ("第4章 Ch.4", "多因子回归诊断（R²、系数）", "Multi-factor regression diagnostics"),
    "ch4_ff_factor_region_assignment": ("第4章 Ch.4", "公司因子区域分配", "Factor-region assignment by firm"),
    "ch4_ff_validation_checks": ("第4章 Ch.4", "多因子运行检查", "Multi-factor run checks"),
    "appendix_ff_caar_by_region": ("附录 Appendix", "多因子 CAAR 分区域", "Multi-factor CAAR by region"),
    "figure4_1_event_time_caar_data": ("第4章 Ch.4", "图 4-1 数据：事件时间 CAAR", "Figure 4-1 data: event-time CAAR"),
    "figure4_1_caar_by_exposure_group_data": ("第4章 Ch.4", "图 4-1 数据：分暴露组 CAAR", "Figure 4-1 data: CAAR by exposure group"),
    "table5_1_forbes_rigobon": ("第5章 Ch.5", "Forbes-Rigobon 核心资产对检验（w40/w60）", "Forbes-Rigobon tests on core pairs (w40/w60)"),
    "ch5_fr_contagion_exceptions": ("第5章 Ch.5", "FDR 后判定为传染的资产对", "Pairs flagged as contagion after FDR"),
    "table5_2_dcc_core_pairs": ("第5章 Ch.5", "DCC-GARCH 核心资产对检验", "DCC-GARCH tests on core pairs"),
    "table5_3_dcc_pre_event_targeting": ("第5章 Ch.5", "DCC 事件前相关目标稳健性", "DCC pre-event correlation targeting"),
    "table5_4_fr_mh17_cluster_window": ("第5章 Ch.5", "跨越 MH17 的 90 日灾难簇窗口", "90-day disaster-cluster window spanning MH17"),
    "table6_1_idiosyncratic_volatility_avr": ("第6章 Ch.6", "特质波动率与放大比 AVR", "Idiosyncratic volatility and AVR"),
    "table6_2_volatility_did": ("第6章 Ch.6", "特质波动率 DiD 检验", "Idiosyncratic-volatility DiD test"),
    "table6_2_volatility_did_sample": ("第6章 Ch.6", "DiD 样本明细", "DiD sample detail"),
    "table6_3_avr_prewindow_robustness": ("第6章 Ch.6", "AVR 前窗稳健性", "AVR pre-window robustness"),
    "table6_4_size_contrast_logavr60": ("第6章 Ch.6", "logAVR60 规模对照", "Size contrasts of logAVR60"),
    "table6_4_size_interaction_logavr120": ("第6章 Ch.6", "logAVR120 规模交互回归", "Size-interaction regression of logAVR120"),
    "table7_1_garch_parameters": ("第7章 Ch.7", "GARCH 参数与分布选择", "GARCH parameters and distribution choice"),
    "table7_1_garch_convergence": ("第7章 Ch.7", "GARCH 各分布收敛诊断", "GARCH convergence by distribution"),
    "table7_1_garch_oos_refits": ("第7章 Ch.7", "GARCH 样本外逐次重估", "GARCH out-of-sample refits"),
    "table7_2_var_backtests": ("第7章 Ch.7", "逐水平 VaR 覆盖回测", "Per-level VaR coverage backtests"),
    "table7_3_multilevel_var": ("第7章 Ch.7", "多水平 VaR 多项式检验（Nass 校正）", "Multi-level multinomial VaR test (Nass)"),
    "table8_1_power_mde": ("第8章 Ch.8", "功效与最小可检测效应", "Power and minimum detectable effects"),
    "table9_1_headline_results": ("第9章 Ch.9", "全文主结果汇总", "Headline results"),
    "ch5_dcc_all_pair_screening": ("第5章 Ch.5", "全部资产对 DCC 探索性筛查（可选）", "All-pair DCC screening (optional)"),
    "ch5_dcc_pairwise_psd": ("第5章 Ch.5", "两两相关矩阵 PSD 诊断（可选）", "Pairwise-correlation PSD diagnostic (optional)"),
    "ch5_fr_all_pair_screening": ("第5章 Ch.5", "全部资产对 FR 探索性筛查（可选）", "All-pair FR screening (optional)"),
}


def core_pairs(columns) -> list[tuple[str, str]]:
    have = set(columns)
    return [(a, b) for a, b in CORE_PAIRS if a in have and b in have]


def choose_dcc_pairs(columns, requested: int = 0) -> list[tuple[str, str]]:
    pairs = list(combinations(columns, 2))
    return pairs if requested <= 0 else pairs[: min(requested, len(pairs))]


def leave_one_out_market(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """无外部基准时的备用基准：剔除自身的等权组合。
    Fallback benchmark without external indices: leave-one-out equal-weight portfolio.
    """
    cols = list(panel.columns)
    if len(cols) < 2:
        raise ValueError("Leave-one-out equal-weight benchmark requires at least two assets.")
    market = pd.DataFrame({f"EW_LOO_{c}": panel.drop(columns=[c]).mean(axis=1) for c in cols}, index=panel.index)
    return market, {c: f"EW_LOO_{c}" for c in cols}


def load_pairing(path: Path) -> dict[str, str]:
    """公司 → 本地基准映射 / Company → local benchmark mapping."""
    df = pd.read_csv(path)
    return dict(zip(df["ticker"].astype(str), df["benchmark_ticker"].astype(str)))


def alignment_for_pair(char_df: pd.DataFrame):
    """同区域 → 同步收益；跨区域 → overlap2 / Same region → contemporaneous; cross-region → overlap2."""
    region = {}
    if char_df is not None and not char_df.empty and {"Ticker", "country_or_region"}.issubset(char_df.columns):
        region = {str(r["Ticker"]): region_of(r["country_or_region"]) for _, r in char_df.iterrows()}

    def choose(left: str, right: str) -> str:
        lr, rr = region.get(left), region.get(right)
        return "contemporaneous" if lr is not None and lr == rr else "overlap2"

    return choose


class Study:
    def __init__(self, config: StudyConfig, output_dir: Path) -> None:
        self.cfg = config
        self.out = output_dir
        for sub in ("tables", "figures", "logs"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.tables: dict[str, pd.DataFrame] = {}
        self.timings: dict[str, float] = {}

    def save(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        self.tables[name] = df
        df.to_csv(self.out / "tables" / f"{name}.csv", index=False)
        return df

    def step(self, label_zh: str, label_en: str, t0: float) -> None:
        self.timings[label_en] = round(time.time() - t0, 1)
        print(f"{label_zh} / {label_en}: {self.timings[label_en]}s", flush=True)

    def execute(self) -> None:
        cfg, P = self.cfg, self.cfg.paths
        start = time.time()
        conf = cfg.confounding_dates

        # ---- 第 3 章：数据 / Chapter 3: data
        t0 = time.time()
        aligner = DataAligner(AlignmentConfig(winsorize=False))
        assets, char = aligner.build_native_returns(P.companies)
        market, _ = aligner.build_native_returns(P.benchmarks)
        bmap = load_pairing(P.pairing)
        usable = [t for t in assets.columns if t in bmap and bmap[t] in market.columns]
        assets = assets[usable]
        char = char[char["Ticker"].astype(str).isin(usable)].reset_index(drop=True)
        rule = alignment_for_pair(char)
        ev_cfg = EventStudyConfig(event_date=cfg.event_date, confounding_dates=conf, random_seed=cfg.random_seed)
        self.save("table3_1_sample_coverage_diagnostics", aligner.coverage_diagnostics(assets, cfg.event_date, ev_cfg.estimation_window, char))
        self.save("table3_3_return_statistics_archlm", aligner.diagnostics(assets))
        self.step("第3章 数据诊断", "Ch.3 data diagnostics", t0)

        # ---- 第 4 章：事件研究 / Chapter 4: event study
        t0 = time.time()
        es = EventStudyEngine(ev_cfg)
        ev = self.save("ch4_firm_level_ar_car", es.run(assets, market, bmap).merge(char, on="Ticker", how="left"))
        self.save("ch4_estimation_window_std_residuals", es.estimation_std_residuals_frame())
        caar, caar_g = es.event_time_caar(assets, market, bmap, char=char)
        self.save("figure4_1_event_time_caar_data", caar)
        self.save("figure4_1_caar_by_exposure_group_data", caar_g)
        est_len = abs(ev_cfg.estimation_window[1] - ev_cfg.estimation_window[0]) + 1
        cs = self.save("table4_1_aggregate_caar_tests", es.cross_sectional_inference(ev, est_len))
        self.save("table4_2_exposure_gradient_hc3", EventStudyEngine.cross_sectional_car_regression(ev, "long"))
        self.save("table4_2_size_contrast_long_car", EventStudyEngine.size_group_contrasts(ev[ev["Window_Type"] == "long"], "Actual_CAR"))
        self.step("第4章 事件研究", "Ch.4 event study", t0)

        # ---- 第 6 章：波动率 / Chapter 6: volatility
        t0 = time.time()
        vol = self.save("table6_1_idiosyncratic_volatility_avr", es.volatility_response(assets, market, bmap).merge(char, on="Ticker", how="left"))
        self.save("table6_4_size_contrast_logavr60", EventStudyEngine.size_group_contrasts(vol, "logAVR_60"))
        self.save("table6_4_size_interaction_logavr120", EventStudyEngine.size_interaction_regression(vol, "logAVR_120"))
        rob_rows = []
        for pre in AVR_PRE_WINDOWS:
            vr = es.volatility_response(assets, market, bmap, pre=pre)
            vr = vr[vr["Model_Status"].astype(str).eq("OK")]
            rob_rows.append({"pre_window": f"[{pre[0]},{pre[1]}]", "n": len(vr), "mean_logAVR_120": float(vr["logAVR_120"].mean()) if len(vr) else np.nan,
                             "median_AVR_120": float(vr["AVR_120"].median()) if len(vr) else np.nan, "share_AVR120_gt1": float((vr["AVR_120"] > 1).mean()) if len(vr) else np.nan})
        self.save("table6_3_avr_prewindow_robustness", pd.DataFrame(rob_rows))
        did, did_sample = es.did_volatility_response(assets, market, bmap, char)
        self.save("table6_2_volatility_did", did)
        self.save("table6_2_volatility_did_sample", did_sample)
        self.step("第6章 波动率", "Ch.6 volatility", t0)

        # ---- 第 7 章：GARCH-VaR / Chapter 7: GARCH-VaR
        t0 = time.time()
        garch = GarchVarEngine(GarchVarConfig())
        ta, tb, resid = garch.run(assets)
        self.save("table7_1_garch_parameters", ta)
        self.save("table7_1_garch_convergence", garch.convergence_diagnostics_)
        self.save("table7_1_garch_oos_refits", garch.oos_refit_diagnostics_)
        self.save("table7_2_var_backtests", tb)
        self.save("table7_3_multilevel_var", garch.multilevel_backtest(assets))
        self.step("第7章 GARCH-VaR", "Ch.7 GARCH-VaR", t0)

        # ---- 第 5 章：传染 / Chapter 5: contagion
        t0 = time.time()
        core = core_pairs(resid.columns)
        self.save("table5_2_dcc_core_pairs", DccFdrEngine(DccConfig(event_date=cfg.event_date, confounding_dates=conf, random_seed=cfg.random_seed)).run(resid, core, alignment_for=rule))
        fr_kw = dict(event_date=cfg.event_date, confounding_dates=conf, random_seed=cfg.random_seed)
        fr = self.save("table5_1_forbes_rigobon", ForbesRigobonEngine(ContagionConfig(windows=(("w40", 40), ("w60", 60)), **fr_kw)).run(assets, core, alignment_for=rule))
        self.save("ch5_fr_contagion_exceptions", fr[fr["FDR_Reject_contagion_5pct"].eq(True)] if "FDR_Reject_contagion_5pct" in fr.columns else fr.iloc[0:0])
        self.save("table5_4_fr_mh17_cluster_window", ForbesRigobonEngine(ContagionConfig(windows=(("w90_includes_MH17", 90),), **fr_kw)).run(assets, core, alignment_for=rule))
        if cfg.run_all_pair_screening:
            scr = DccConfig(event_date=cfg.event_date, confounding_dates=conf, placebo_replications=299)
            pairs = None if cfg.max_dcc_pairs <= 0 else choose_dcc_pairs(list(resid.columns), cfg.max_dcc_pairs)
            allt = self.save("ch5_dcc_all_pair_screening", DccFdrEngine(scr).run(resid, pairs, alignment_for=rule))
            self.save("ch5_dcc_pairwise_psd", DccFdrEngine.pairwise_psd_diagnostic(allt))
            self.save("ch5_fr_all_pair_screening", ForbesRigobonEngine(ContagionConfig(windows=(("w60", 60),), placebo_replications=299, **fr_kw)).run(assets, choose_dcc_pairs(list(assets.columns)), alignment_for=rule))
        self.step("第5章 传染检验", "Ch.5 contagion tests", t0)

        # ---- 稳健性 / robustness (表 4-3、5-3)
        t0 = time.time()
        rob = RobustnessEngine(RobustnessConfig(event_date=cfg.event_date, confounding_dates=conf, random_seed=cfg.random_seed))
        self.save("table4_3_window_sensitivity", rob.event_window_sensitivity(assets, market, bmap))
        self.save("table5_3_dcc_pre_event_targeting", rob.dcc_pre_event_targeting(resid, core, alignment_for=rule))
        self.step("稳健性检验", "Robustness checks", t0)

        # ---- 表 4-4：多因子 / Table 4-4: multi-factor
        t0 = time.time()
        for name, df in run_factor_models(P, cs, ev).items():
            self.save(name, df)
        self.step("表4-4 多因子", "Table 4-4 multi-factor", t0)

        # ---- 第 8、9 章 / Chapters 8-9
        t0 = time.time()
        power = self.save("table8_1_power_mde", build_power_table(self.tables, resid, cfg.event_date, rule))
        self.save("table9_1_headline_results", build_headline(self.tables))
        self.step("第8-9章 功效与汇总", "Ch.8-9 power and summary", t0)

        # ---- 图 / figures
        t0 = time.time()
        fig = self.out / "figures"
        jobs = [
            ("figure3_1_data_coverage_heatmap", lambda p: figures.coverage_heatmap(P.companies, p)),
            ("figure4_1_event_time_caar", lambda p: figures.caar_paths(caar, caar_g, p)),
            ("figure5_1_core_pair_dcc_path", lambda p: figures.dcc_path(resid, p, cfg.event_date, alignment_for=rule)),
            ("figure7_1_garch_var_boundary", lambda p: figures.var_boundary(assets, p, garch)),
            ("figure7_2_multilevel_var_coverage", lambda p: figures.var_coverage(tb, p)),
            ("figure8_1_power_mde_forest", lambda p: figures.power_forest(power, p)),
        ]
        for name, job in jobs:
            try:
                job(fig / f"{name}.png")
            except Exception as exc:  # 作图失败不影响表格 / figure failures never block the tables
                with (self.out / "logs" / "figure_warnings.txt").open("a", encoding="utf-8") as fh:
                    fh.write(f"{name}: {exc}\n")
        self.step("作图", "Figures", t0)

        # ---- 结果表索引与运行记录 / table index and run manifest
        # 中文与英文索引分别保存 / Chinese and English indices are saved as separate files
        cn, en = [], []
        for n, df in self.tables.items():
            chap, zh, eng = TABLE_INDEX.get(n, ("", "", ""))
            chap_cn, _, chap_en = chap.partition(" ")
            cn.append({"文件": f"tables/{n}.csv", "章节": chap_cn, "说明": zh, "行数": len(df)})
            en.append({"file": f"tables/{n}.csv", "chapter": chap_en or chap_cn, "description": eng, "rows": len(df)})
        pd.DataFrame(cn).sort_values("文件").to_csv(self.out / "results_index_cn.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(en).sort_values("file").to_csv(self.out / "results_index_en.csv", index=False, encoding="utf-8")
        manifest = {
            "python": sys.version, "platform": platform.platform(),
            "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "arch", "matplotlib")},
            "event_date": cfg.event_date, "confounding_dates": list(conf), "random_seed": cfg.random_seed,
            "timings_seconds": self.timings, "elapsed_seconds": round(time.time() - start, 1), "command": " ".join(sys.argv),
        }
        (self.out / "logs" / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"全部完成 / Done in {manifest['elapsed_seconds']}s → {self.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="MH370 资本市场冲击研究 / MH370 capital-market impact study")
    ap.add_argument("--output-dir", default=None, help="默认 <项目根>/results / default <project>/results")
    ap.add_argument("--run-all-pair-screening", action="store_true", help="全部资产对 DCC/FR 筛查（耗时长）/ all-pair DCC/FR screening (slow)")
    ap.add_argument("--max-dcc-pairs", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = StudyConfig(run_all_pair_screening=args.run_all_pair_screening, max_dcc_pairs=args.max_dcc_pairs)
    Study(cfg, Path(args.output_dir) if args.output_dir else cfg.paths.results).execute()


if __name__ == "__main__":
    main()
