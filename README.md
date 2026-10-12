# MH370 事件对全球资本市场的冲击 / The Impact of the MH370 Disappearance on Global Capital Markets

## 1. 研究任务 / Research Task

**中文**

马来西亚航空 MH370 航班于 2014 年 3 月 8 日失联。本项目以 2014-03-10 为事件日（失联后的首个交易日），评估这一航空灾难对全球相关上市公司的冲击。研究回答五个问题：

- 是否引发了显著的股价重估？
- 冲击是否在资产之间形成“传染”，而不只是正常的“相依”？
- 是否带来持续的特质波动放大？
- 标准 GARCH-VaR 模型能否覆盖尾部风险？
- 现有样本是否有足够的统计功效检出合理规模的效应？

样本期为 2012-01-03 至 2016-12-30。2014-07-17 的 MH17 空难属于同一家航空公司的第二次灾难，研究将其作为混淆事件，从所有安慰剂抽样中排除；确认性窗口限定在事件后 +60 个交易日以内。

**English**

Malaysia Airlines flight MH370 disappeared on 8 March 2014. Using 10 March 2014, the first trading day after the disappearance, as the event date, this project assesses the shock to listed firms worldwide. The study answers five questions:

- Did markets significantly re-price exposed firms?
- Did the shock produce contagion across assets, beyond ordinary interdependence?
- Was there persistent amplification of idiosyncratic volatility?
- Do standard GARCH-VaR models cover tail risk?
- Does the sample have enough statistical power to detect plausibly sized effects?

The sample runs from 2012-01-03 to 2016-12-30. The MH17 crash of 2014-07-17, a second disaster for the same airline, is treated as a confounding event: it is excluded from all placebo sampling, and confirmatory windows are limited to +60 trading days.

## 2. 数据 / Data

**中文**

- **样本公司**：75 家，覆盖 23 个国家或地区、16 种货币。
  - 行业渠道：航空 30 家，酒店旅游 16 家，保险金融 16 家，机场 13 家。
  - 暴露梯度：直接暴露 4 家，亚太航空 9 家，区域航空 6 家，全球航空 11 家。
  - 规模：大型 51 家，中小型 24 家。
- **本地市场基准**：17 个指数，每家公司配对其上市地指数。
- **多因子数据**：
  - Kenneth French 区域 FF3/FF5 日度因子（北美、欧洲、日本、亚太、发达市场）；
  - JKP 国别主题因子（中国、马来西亚、韩国、土耳其、泰国），包括 mkt/size/value/profitability/investment 五个单因子文件，以及合成后的 FF3/FF5 总表；
  - 美元兑各本币日度汇率。
- **价格来源**：Yahoo Finance 复权收盘价。

**English**

- **Sample firms**: 75 firms in 23 countries or regions, quoted in 16 currencies.
  - Channels: 30 airlines, 16 tourism/hotel, 16 insurance/finance and 13 airport firms.
  - Exposure gradient: 4 direct, 9 Asia-Pacific airline, 6 regional airline and 11 global airline.
  - Size: 51 large and 24 mid/small firms.
- **Local benchmarks**: 17 indices; each firm is paired with the index of its listing market.
- **Factor data**:
  - Kenneth French regional FF3/FF5 daily factors (North America, Europe, Japan, Asia-Pacific, Developed);
  - JKP country theme factors (China, Malaysia, South Korea, Turkey, Thailand), with the five single-theme files (mkt/size/value/profitability/investment) and the merged FF3/FF5 files;
  - Daily USD per local-currency exchange rates.
- **Price source**: Yahoo Finance adjusted closes.

## 3. 研究框架与方法 / Framework and Methods

| 章节 Chapter | 内容 | Content |
|---|---|---|
| 第3章 Ch.3 | 样本覆盖、日收益描述统计、ARCH-LM；按各市场本地交易日历构造收益 | Coverage, return statistics, ARCH-LM; returns built on each market's own calendar |
| 第4章 Ch.4 | 市场模型事件研究，估计窗 [−210,−30]，8 个检验窗口；公司层面用 Corrado、自助法和日历安慰剂并做 BH-FDR；聚合层面用 Plain-CS、Patell、BMP 和 ADJ-BMP；HC3 暴露梯度回归；规模置换检验；替代估计窗；FF3/FF5 美元超额收益多因子模型 | Market-model event study with estimation window [−210,−30] and 8 test windows; firm-level Corrado, bootstrap and calendar placebo with BH-FDR; aggregate Plain-CS, Patell, BMP and ADJ-BMP; HC3 exposure regression; size permutation tests; alternative windows; FF3/FF5 USD excess-return models |
| 第5章 Ch.5 | 19 个核心资产对的两两 DCC-GARCH(1,1) 与 Forbes-Rigobon 异方差校正检验；事件前相关目标；跨越 MH17 的 90 日灾难簇窗口 | Pairwise DCC-GARCH(1,1) and Forbes-Rigobon tests on 19 core pairs; pre-event correlation targeting; 90-day cluster window spanning MH17 |
| 第6章 Ch.6 | 特质波动放大比 AVR；处理组与对照组的 DiD 置换检验；前窗稳健性；规模交互回归 | Abnormal volatility ratio (AVR); treated-vs-control DiD permutation test; pre-window robustness; size interactions |
| 第7章 Ch.7 | GARCH(1,1)，在正态、t、偏 t 中按惩罚 BIC 选择分布；扩展窗口样本外 VaR（95/97.5/99/99.5%）；Kupiec、Christoffersen 检验及 Kratz 多项式多水平检验（Nass 校正） | GARCH(1,1) with normal/t/skewed-t chosen by penalised BIC; expanding-window out-of-sample VaR (95/97.5/99/99.5%); Kupiec, Christoffersen and Kratz multinomial tests with the Nass correction |
| 第8章 Ch.8 | 80% 功效下的最小可检测效应 MDE，与文献中的合理效应量比较 | Minimum detectable effects at 80% power compared with literature-based plausible effects |
| 第9章 Ch.9 | 主结果汇总 | Headline results |

**关键设计 / Key design choices**

**中文**

- **日历安慰剂**：伪事件窗口不得与真实事件窗口重叠，也不得覆盖 MH17±5 日。
- **ADJ-BMP**：用估计窗残差的平均两两相关 r̄ 校正截面相关；MDE 采用 CAR 尺度上经 Kolari-Pynnönen 因子膨胀的截面标准误。
- **资产对齐**：同一区域内的资产对使用同步收益，跨区域的资产对使用两日重叠收益，以缓解收盘时差。
- **多因子模型**：汇率在股票交易日上前向填充，SCAR 含预测误差项；新兴市场使用 JKP 国别因子，不用发达市场因子代替。
- **功效判定**：先看观测效应是否显著，再区分“有信息的零结果”与“功效不足”。
- **随机种子**：统一为 3702014，各检验使用相互独立的随机数流。

**English**

- **Calendar placebo**: pseudo-event windows may not overlap the true event window or cover MH17 ± 5 days.
- **ADJ-BMP**: corrects for cross-sectional correlation using the mean pairwise correlation r̄ of estimation-window residuals. MDEs use the Kolari-Pynnönen-inflated cross-sectional standard error on the CAR scale.
- **Asset alignment**: pairs within the same region use contemporaneous returns; cross-region pairs use two-day overlapping returns to soften closing-time differences.
- **Multi-factor models**: FX rates are forward-filled onto stock trading days, and the SCAR includes the prediction-error term. Emerging markets use JKP country factors rather than developed-market proxies.
- **Power verdicts**: significance of the observed effect is checked first; only then is an informative null distinguished from low power.
- **Random seed**: 3702014 throughout, with independent random streams for each test.

## 4. 主要结果 / Main Results

### 第3章 数据特征 / Ch.3 Data properties

**中文**：日收益普遍呈厚尾，峰度中位数为 7.3；75 家公司中有 58 家在 5 阶 ARCH-LM 检验下拒绝“无条件异方差”，这支持后续采用 GARCH 类模型。

**English**: Daily returns are fat-tailed (median kurtosis 7.3), and 58 of 75 firms reject no-ARCH at 5 lags, which motivates GARCH-type models.

### 第4章 价格冲击 / Ch.4 Price impact

**中文**

- 长窗口 [−5,+30] 的平均 CAR 为 **−2.37%**。
  - 不校正截面相关时显著（Plain-CS p=0.023，BMP p=0.038）；
  - 经截面相关校正后不显著（**ADJ-BMP p=0.203**），MDE 为 4.7%，对 3% 量级的效应功效不足。
- 短窗口内（[−1,+1] 至 [−5,+5]）没有可检出的反应。
- 事件后 120 日的 CAR 为 −5.94%，ADJ-BMP 下同样不显著（p=0.156）。
- 公司层面经 FDR 校正后，所有窗口的拒绝数均为 0。
- 暴露梯度斜率为 0.007（p=0.73）；规模对照与渠道效应均不显著。
- 替代估计窗下，长窗口 CAR 为 −3.0% 和 −2.7%，方向与主结果一致。
- FF3/FF5 美元超额收益模型下，长窗口 CAAR 分别为 −2.08% 和 −1.55%，ADJ-BMP p 分别为 0.31 和 0.45，FDR 拒绝数均为 0，结论不依赖正常收益模型的设定。

**English**

- The long-window [−5,+30] mean CAR is **−2.37%**.
  - It is significant before correcting for cross-sectional correlation (Plain-CS p=0.023, BMP p=0.038);
  - It is not significant after the correction (**ADJ-BMP p=0.203**). The MDE is 4.7%, so the test is underpowered for a 3% effect.
- There is no detectable reaction in the short windows ([−1,+1] to [−5,+5]).
- The 120-day post-event CAR is −5.94%, also insignificant under ADJ-BMP (p=0.156).
- Firm-level FDR rejections are zero in every window.
- The exposure-gradient slope is 0.007 (p=0.73); size contrasts and channel effects are insignificant.
- Alternative estimation windows give long-window CARs of −3.0% and −2.7%, the same direction as the main result.
- FF3/FF5 USD excess-return models give long-window CAARs of −2.08% and −1.55% (ADJ-BMP p=0.31 and 0.45) with zero FDR rejections, so the conclusion does not depend on the normal-return model.

### 第5章 传染 / Ch.5 Contagion

**中文**

- 在不受 MH17 干扰的窗口（w40、w60）中，Forbes-Rigobon 检验 **0/38 拒绝**。异方差校正后的相关变化为负（w40 −0.168，w60 −0.062），表现为相依而非传染。
- DCC 三个窗口的条件相关变化均值为 −0.008 至 −0.016，FDR 拒绝数为 **0/57**；改用事件前相关目标后仍为 0。
- 只有在跨越 MH17 的 90 日窗口中，亚航与慕尼黑再保险（5099.KL–MUV2.DE）这一对被拒绝（FDR p=0.029）。该结果对应的是马航的灾难簇，而非 MH370 本身。

**English**

- In the MH17-free windows (w40, w60), Forbes-Rigobon tests reject **0/38**. Heteroskedasticity-adjusted correlation changes are negative (w40 −0.168, w60 −0.062), indicating interdependence rather than contagion.
- Mean DCC shifts are −0.008 to −0.016 across the three windows with **0/57** FDR rejections, and still 0 under pre-event targeting.
- Only the 90-day window spanning MH17 rejects one pair, AirAsia–Munich Re (5099.KL–MUV2.DE, FDR p=0.029). This reflects the Malaysia Airlines disaster cluster rather than MH370 alone.

### 第6章 波动率 / Ch.6 Volatility

**中文**

- 事件后 120 日的特质波动相对事件前整体下降：平均 logAVR 为 **−0.156**（t≈−4.7），仅 25% 的公司 AVR 大于 1。前窗缩短后，下降幅度收窄到 −0.06 至 −0.14，说明这反映的是 2013 年下半年较高波动的基期，而非事件效应。
- 处理组（13 家高暴露航空）与对照组的 DiD 为 **−0.038**（置换 p=0.665），相对市场的 DiD 为 0.130（p=0.175），均无 MH370 特有的波动放大。
- 规模对照与交互项均不显著。

**English**

- Idiosyncratic volatility over the 120 post-event days falls relative to the pre-event baseline: mean logAVR is **−0.156** (t≈−4.7), and only 25% of firms have AVR > 1. Shorter pre-windows shrink the decline to between −0.06 and −0.14, so this reflects a high-volatility baseline in late 2013 rather than an event effect.
- The treated (13 high-exposure airlines) minus control DiD is **−0.038** (permutation p=0.665), and the market-relative DiD is 0.130 (p=0.175). There is no MH370-specific volatility amplification.
- Size contrasts and interactions are insignificant.

### 第7章 尾部风险 / Ch.7 Tail risk

**中文**

- 分布选择结果：Student-t 62 家、正态 8 家、偏 t 5 家；α+β 中位数为 0.946；4 家公司触及持续性边界。
- 样本外 VaR 的汇总突破率在所有水平上都高于名义值：95% 水平为 5.6%，97.5% 为 3.0%，99% 为 1.34%，99.5% 为 0.76%。
- Kratz 多水平检验（Nass 校正）下，**21/75** 家公司被拒绝，表明标准 GARCH 对部分公司的尾部覆盖不足。

**English**

- Distribution choice: Student-t for 62 firms, normal for 8 and skewed-t for 5; median α+β is 0.946; 4 firms hit the persistence boundary.
- Pooled out-of-sample exceedance rates exceed nominal at every level: 5.6% at 95%, 3.0% at 97.5%, 1.34% at 99% and 0.76% at 99.5%.
- The Nass-corrected Kratz multi-level test rejects for **21/75** firms, so standard GARCH under-covers the tails of a subset of firms.

### 第8章 统计功效 / Ch.8 Power

**中文**

- 聚合 CAR 的 ADJ-BMP 口径 MDE 为 4.7%（长窗口）和 11.5%（事件后 120 日），均高于 3% 的合理效应量，属于功效不足。
- 暴露梯度与规模调节的 MDE 是合理效应量的 3 到 7 倍。
- 区域内 DCC 相关变化的 MDE 为 0.099，低于合理效应量 0.10，构成有信息的零结果。
- 99% VaR 的 Kupiec 检验对“真实突破率 1.5%”的检出功效仅为 0.33。

**English**

- Aggregate-CAR MDEs under ADJ-BMP are 4.7% (long window) and 11.5% (120 post-event days), both above the 3% plausible effect, so these tests are underpowered.
- MDEs for the exposure gradient and size moderation are 3 to 7 times the plausible effects.
- The intra-region DCC shift has an MDE of 0.099, below the plausible 0.10, which makes it an informative null.
- The 99% Kupiec test has only 0.33 power against a true 1.5% exceedance rate.

### 第9章 总体结论 / Ch.9 Overall conclusion

**中文**：市场把 MH370 视为一次局部、温和的冲击。价格下跌约 2.4%，但不能区别于截面相关带来的噪声；资产间没有传染，也没有 MH370 特有的波动放大。尾部风险模型对部分公司存在覆盖不足。由于单一日历事件的有效独立样本有限（N_eff≈28），对中小规模效应的零结果应解读为“功效不足”，而不是“效应不存在”。

**English**: Markets treated MH370 as a local, moderate shock. Prices fell by about 2.4%, but this cannot be distinguished from cross-sectionally correlated noise; there was no contagion and no MH370-specific volatility amplification. Tail-risk models under-cover a subset of firms. Because a single calendar-date event leaves only about 28 effectively independent firms (N_eff≈28), null results for small-to-moderate effects should be read as low power rather than as evidence of no effect.

## 5. 项目结构 / Project Structure

```
MH370_capital_market_study/
├── README.md
├── requirements.txt
├── 文件清单_file_inventory.csv          全部文件的中英文说明 / bilingual description of every file
├── mh370_study/                         分析代码 / analysis code
│   ├── config.py                        参数与路径 / parameters and paths
│   ├── stats_utils.py                   公共统计函数 / shared statistics
│   ├── data_alignment.py                第3章 / Ch.3
│   ├── event_study.py                   第4、6章 / Ch.4, 6
│   ├── factor_models.py                 第4章 表4-4 / Ch.4 Table 4-4
│   ├── contagion_tests.py               第5章 / Ch.5
│   ├── garch_var.py                     第7章 / Ch.7
│   ├── robustness.py                    表4-3、5-3 / Tables 4-3, 5-3
│   ├── power_mde.py                     第8章 / Ch.8
│   ├── headline_summary.py              第9章 / Ch.9
│   ├── figures.py                       图形 / figures
│   └── run_all.py                       一键运行 / one-command run
├── data/
│   ├── metadata/                        样本清单、配对表、基准清单、因子区域映射 / sample, pairing, benchmarks, factor regions
│   ├── prices/companies/                75 家公司日度价格 / daily prices of 75 firms
│   ├── prices/benchmarks/               17 个基准指数 / 17 benchmark indices
│   ├── factors/french/                  French 区域因子 / French regional factors
│   ├── factors/jkp/                     JKP 国别因子（单因子与合成表）/ JKP country factors (single themes and merged)
│   └── fx/                              汇率 / exchange rates
├── results/
│   ├── results_index.csv                结果表索引（中英文）/ bilingual index of result tables
│   ├── tables/                          35 张结果表 / 35 result tables
│   ├── figures/                         6 幅图 / 6 figures
│   └── logs/run_manifest.json           运行环境与耗时 / environment and timings
├── tests/                               48 个单元测试 / 48 unit tests
└── docs/literature/                     外文文献原文与中文译文 / reference paper and Chinese translation
```

## 6. 运行 / How to Run

```bash
pip install -r requirements.txt
python -m mh370_study.run_all          # 约 7 分钟，结果写入 results/ / about 7 minutes; writes to results/
python -m pytest -q                    # 单元测试 / unit tests
```

运行 `python -m mh370_study.run_all --run-all-pair-screening` 可附加全部资产对的 DCC/FR 探索性筛查，耗时较长。

Running `python -m mh370_study.run_all --run-all-pair-screening` adds an exploratory DCC/FR screen over all asset pairs; this is slow.

## 7. 命名规则 / Naming Conventions

**中文**

- **公司价格文件**：`<代码>_<公司名>.csv`，代码中的“.”替换为“_”，例如 `0753_HK_Air_China.csv`。
- **基准指数文件**：`<指数代码>_<指数名>.csv`，例如 `HSI_Hang_Seng_Index.csv`。
- **因子文件**：`<区域>_<FF3|FF5>_daily_decimal.csv`；JKP 单因子文件为 `JKP_<国家>_<主题>_daily_vw_cap.csv`。
- **结果表**：以论文表号或章节开头，例如 `table4_1_…`、`ch5_…`、`appendix_…`。

**English**

- **Firm price files**: `<ticker>_<company>.csv`, with “.” in the ticker replaced by “_”, e.g. `0753_HK_Air_China.csv`.
- **Benchmark files**: `<index code>_<index name>.csv`, e.g. `HSI_Hang_Seng_Index.csv`.
- **Factor files**: `<region>_<FF3|FF5>_daily_decimal.csv`; JKP single-theme files are `JKP_<country>_<theme>_daily_vw_cap.csv`.
- **Result tables**: prefixed with the thesis table number or chapter, e.g. `table4_1_…`, `ch5_…`, `appendix_…`.
