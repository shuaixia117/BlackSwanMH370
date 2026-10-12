"""全局参数与数据路径。
Global parameters and data paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 事件日：MH370 于 2014-03-08（周六）失联，首个交易日为 2014-03-10
# Event date: MH370 disappeared on Sat 2014-03-08; the first trading day is 2014-03-10
EVENT_DATE = "2014-03-10"
# 混淆事件：MH17 于 2014-07-17 被击落，从安慰剂抽样中屏蔽
# Confounding event: MH17 was shot down on 2014-07-17; excluded from placebo sampling
MH17_DATE = "2014-07-17"
SAMPLE_START = "2012-01-03"
SAMPLE_END = "2016-12-30"
RANDOM_SEED = 3702014


@dataclass(frozen=True)
class Paths:
    """项目目录约定 / Project directory layout."""

    root: Path = PROJECT_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def companies(self) -> Path:
        return self.data / "prices" / "companies"

    @property
    def benchmarks(self) -> Path:
        return self.data / "prices" / "benchmarks"

    @property
    def company_sample(self) -> Path:
        return self.data / "metadata" / "company_sample.csv"

    @property
    def pairing(self) -> Path:
        return self.data / "metadata" / "company_benchmark_pairing.csv"

    @property
    def factors(self) -> Path:
        return self.data / "factors"

    @property
    def fx(self) -> Path:
        return self.data / "fx" / "usd_per_local_currency.csv"

    @property
    def results(self) -> Path:
        return self.root / "results"


@dataclass(frozen=True)
class StudyConfig:
    """一次完整运行的参数 / Parameters of a full run."""

    paths: Paths = field(default_factory=Paths)
    event_date: str = EVENT_DATE
    confounding_dates: tuple[str, ...] = (MH17_DATE,)
    random_seed: int = RANDOM_SEED
    run_all_pair_screening: bool = False  # 附加：全部资产对筛查 / optional all-pair screening
    max_dcc_pairs: int = 0
