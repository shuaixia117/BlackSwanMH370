# The Impact of the MH370 Disappearance on Global Capital Markets

[中文版](README_cn.md) · [Methods and models](METHODS_en.md)

## 1. Research Task

Malaysia Airlines flight MH370 disappeared on 8 March 2014. Using 10 March 2014, the first trading day after the disappearance, as the event date, this project assesses the shock to listed firms worldwide. The study answers five questions:

- Did markets significantly re-price exposed firms?
- Did the shock produce contagion across assets, beyond ordinary interdependence?
- Was there persistent amplification of idiosyncratic volatility?
- Do standard GARCH-VaR models cover tail risk?
- Does the sample have enough statistical power to detect plausibly sized effects?

The sample runs from 2012-01-03 to 2016-12-30. The MH17 crash of 2014-07-17, a second disaster for the same airline, is treated as a confounding event: it is excluded from all placebo sampling, and confirmatory windows are limited to +60 trading days.

## 2. Data

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

## 3. Framework and Methods

Formulas, parameters and test details for every model are given in [METHODS_en.md](METHODS_en.md).

| Chapter | Content |
|---|---|
| Ch.3 | Coverage, return statistics, ARCH-LM; returns built on each market's own calendar |
| Ch.4 | Market-model event study with estimation window [−210,−30] and 8 test windows; firm-level Corrado, bootstrap and calendar placebo with BH-FDR; aggregate Plain-CS, Patell, BMP and ADJ-BMP; HC3 exposure regression; size permutation tests; alternative windows; FF3/FF5 USD excess-return models |
| Ch.5 | Pairwise DCC-GARCH(1,1) and Forbes-Rigobon tests on 19 core pairs; pre-event correlation targeting; 90-day cluster window spanning MH17 |
| Ch.6 | Abnormal volatility ratio (AVR); treated-vs-control DiD permutation test; pre-window robustness; size interactions |
| Ch.7 | GARCH(1,1) with normal/t/skewed-t chosen by penalised BIC; expanding-window out-of-sample VaR (95/97.5/99/99.5%); Kupiec, Christoffersen and Kratz multinomial tests with the Nass correction |
| Ch.8 | Minimum detectable effects at 80% power compared with literature-based plausible effects |
| Ch.9 | Headline results |

**Key design choices**

- **Calendar placebo**: pseudo-event windows may not overlap the true event window or cover MH17 ± 5 days.
- **ADJ-BMP**: corrects for cross-sectional correlation using the mean pairwise correlation r̄ of estimation-window residuals. MDEs use the Kolari-Pynnönen-inflated cross-sectional standard error on the CAR scale.
- **Asset alignment**: pairs within the same region use contemporaneous returns; cross-region pairs use two-day overlapping returns to soften closing-time differences.
- **Multi-factor models**: FX rates are forward-filled onto stock trading days, and the SCAR includes the prediction-error term. Emerging markets use JKP country factors rather than developed-market proxies.
- **Power verdicts**: significance of the observed effect is checked first; only then is an informative null distinguished from low power.
- **Random seed**: 3702014 throughout, with independent random streams for each test.

## 4. Main Results

### Ch.3 Data properties

Daily returns are fat-tailed (median kurtosis 7.3), and 58 of 75 firms reject no-ARCH at 5 lags, which motivates GARCH-type models.

### Ch.4 Price impact

- The long-window [−5,+30] mean CAR is **−2.37%**.
  - It is significant before correcting for cross-sectional correlation (Plain-CS p=0.023, BMP p=0.038);
  - It is not significant after the correction (**ADJ-BMP p=0.203**). The MDE is 4.7%, so the test is underpowered for a 3% effect.
- There is no detectable reaction in the short windows ([−1,+1] to [−5,+5]).
- The 120-day post-event CAR is −5.94%, also insignificant under ADJ-BMP (p=0.156).
- Firm-level FDR rejections are zero in every window.
- The exposure-gradient slope is 0.007 (p=0.73); size contrasts and channel effects are insignificant.
- Alternative estimation windows give long-window CARs of −3.0% and −2.7%, the same direction as the main result.
- FF3/FF5 USD excess-return models give long-window CAARs of −2.08% and −1.55% (ADJ-BMP p=0.31 and 0.45) with zero FDR rejections, so the conclusion does not depend on the normal-return model.

### Ch.5 Contagion

- In the MH17-free windows (w40, w60), Forbes-Rigobon tests reject **0/38**. Heteroskedasticity-adjusted correlation changes are negative (w40 −0.168, w60 −0.062), indicating interdependence rather than contagion.
- Mean DCC shifts are −0.008 to −0.016 across the three windows with **0/57** FDR rejections, and still 0 under pre-event targeting.
- Only the 90-day window spanning MH17 rejects one pair, AirAsia–Munich Re (5099.KL–MUV2.DE, FDR p=0.029). This reflects the Malaysia Airlines disaster cluster rather than MH370 alone.

### Ch.6 Volatility

- Idiosyncratic volatility over the 120 post-event days falls relative to the pre-event baseline: mean logAVR is **−0.156** (t≈−4.7), and only 25% of firms have AVR > 1. Shorter pre-windows shrink the decline to between −0.06 and −0.14, so this reflects a high-volatility baseline in late 2013 rather than an event effect.
- The treated (13 high-exposure airlines) minus control DiD is **−0.038** (permutation p=0.665), and the market-relative DiD is 0.130 (p=0.175). There is no MH370-specific volatility amplification.
- Size contrasts and interactions are insignificant.

### Ch.7 Tail risk

- Distribution choice: Student-t for 62 firms, normal for 8 and skewed-t for 5; median α+β is 0.946; 4 firms hit the persistence boundary.
- Pooled out-of-sample exceedance rates exceed nominal at every level: 5.6% at 95%, 3.0% at 97.5%, 1.34% at 99% and 0.76% at 99.5%.
- The Nass-corrected Kratz multi-level test rejects for **21/75** firms, so standard GARCH under-covers the tails of a subset of firms.

### Ch.8 Power

- Aggregate-CAR MDEs under ADJ-BMP are 4.7% (long window) and 11.5% (120 post-event days), both above the 3% plausible effect, so these tests are underpowered.
- MDEs for the exposure gradient and size moderation are 3 to 7 times the plausible effects.
- The intra-region DCC shift has an MDE of 0.099, below the plausible 0.10, which makes it an informative null.
- The 99% Kupiec test has only 0.33 power against a true 1.5% exceedance rate.

### Ch.9 Overall conclusion

Markets treated MH370 as a local, moderate shock. Prices fell by about 2.4%, but this cannot be distinguished from cross-sectionally correlated noise; there was no contagion and no MH370-specific volatility amplification. Tail-risk models under-cover a subset of firms. Because a single calendar-date event leaves only about 28 effectively independent firms (N_eff≈28), null results for small-to-moderate effects should be read as low power rather than as evidence of no effect.

## 5. Project Structure

```
BlackSwanMH370/
├── README_cn.md                  project description (Chinese)
├── README_en.md                  project description (English)
├── METHODS_cn.md                 methods and models (Chinese)
├── METHODS_en.md                 methods and models (English)
├── requirements.txt
├── file_inventory_cn.csv         inventory of all files (Chinese)
├── file_inventory_en.csv         inventory of all files (English)
├── mh370_study/                  analysis code (bilingual comments)
│   ├── config.py                 parameters and paths
│   ├── stats_utils.py            shared statistics
│   ├── data_alignment.py         Ch.3
│   ├── event_study.py            Ch.4, 6
│   ├── factor_models.py          Ch.4 Table 4-4
│   ├── contagion_tests.py        Ch.5
│   ├── garch_var.py              Ch.7
│   ├── robustness.py             Tables 4-3, 5-3
│   ├── power_mde.py              Ch.8
│   ├── headline_summary.py       Ch.9
│   ├── figures.py                figures
│   └── run_all.py                one-command run
├── data/
│   ├── metadata/                 sample, pairing, benchmarks, factor regions
│   ├── prices/companies/         daily prices of 75 firms
│   ├── prices/benchmarks/        17 benchmark indices
│   ├── factors/french/           French regional factors
│   ├── factors/jkp/              JKP country factors (single themes and merged)
│   └── fx/                       exchange rates
├── results/
│   ├── results_index_cn.csv      index of result tables (Chinese)
│   ├── results_index_en.csv      index of result tables (English)
│   ├── tables/                   35 result tables
│   ├── figures/                  6 figures
│   └── logs/run_manifest.json    environment and timings
└── tests/                        48 unit tests
```

## 6. How to Run

```bash
pip install -r requirements.txt
python -m mh370_study.run_all          # about 7 minutes; writes to results/
python -m pytest -q                    # unit tests
```

Running `python -m mh370_study.run_all --run-all-pair-screening` adds an exploratory DCC/FR screen over all asset pairs; this is slow.

## 7. Naming Conventions

- **Firm price files**: `<ticker>_<company>.csv`, with “.” in the ticker replaced by “_”, e.g. `0753_HK_Air_China.csv`.
- **Benchmark files**: `<index code>_<index name>.csv`, e.g. `HSI_Hang_Seng_Index.csv`.
- **Factor files**: `<region>_<FF3|FF5>_daily_decimal.csv`; JKP single-theme files are `JKP_<country>_<theme>_daily_vw_cap.csv`.
- **Result tables**: prefixed with the thesis table number or chapter, e.g. `table4_1_…`, `ch5_…`, `appendix_…`.
- **Documentation files**: Chinese versions end in `_cn`, English versions in `_en`.
