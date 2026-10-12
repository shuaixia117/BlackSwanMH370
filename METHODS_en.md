# Methods and Models

[中文版](METHODS_cn.md) · [Back to project README](README_en.md)

This document sets out the econometric methods, model specifications and parameter values used in the project; each section maps directly to the implementation in `mh370_study/`. All returns are log returns in decimal units. The event date $e$ is the position of 10 March 2014 in the relevant trading calendar; if that day is not a trading day, the next trading day is used.

---

## 1. Data Construction and Alignment (Ch. 3, `data_alignment.py`)

**Log returns.** For each asset, on its own local trading calendar,

$$r_t = \ln\left(P_t / P_{t-1}\right),$$

where $P_t$ is the adjusted close. Assets are outer-joined by date with no global common-date intersection, so a missing value means the market was closed that day.

**Firm–benchmark alignment.** Each firm is aligned only with the benchmark index of its listing market, on the two series' common trading days. The event centre, estimation window and test windows are all counted in trading days on this pairwise calendar.

**Asset-pair alignment.** Pairs are handled in one of two ways:
- Pairs within the same broad region (Americas, Asia, Europe, Oceania) use contemporaneous returns.
- Cross-region pairs use two-day overlapping returns $r_t + r_{t-1}$ (`overlap2`), to soften the bias from non-synchronous closing times.

**Descriptive statistics.** The tables report the mean, standard deviation, skewness, kurtosis (not excess) and Engle (1982) ARCH-LM p-values at 1 and 5 lags, computed on demeaned returns.

---

## 2. Event Study (Ch. 4, `event_study.py`)

### 2.1 Normal-return model

The market model

$$R_{it} = \alpha_i + \beta_i R_{mt} + \varepsilon_{it}$$

is estimated by OLS over the estimation window $[e-210,\ e-30]$ ($L = 181$ trading days).

- Residual variance: $\hat\sigma_i^2 = \sum \hat\varepsilon_{it}^2 / (L-2)$.
- Abnormal return: $AR_{it} = R_{it} - \hat\alpha_i - \hat\beta_i R_{mt}$.

### 2.2 Test windows

| Label | Window | Label | Window |
|---|---|---|---|
| short | [−1, +1] | long | [−5, +30] |
| sym3 | [−3, +3] | post20 | [+1, +20] |
| sym5 | [−5, +5] | post60 | [+1, +60] |
| med | [−5, +10] | post120 | [+1, +120] |

The cumulative abnormal return is $CAR_i(\tau_1,\tau_2) = \sum_{t=\tau_1}^{\tau_2} AR_{it}$.

### 2.3 Standardised CAR

The CAR variance includes the prediction-error term:

$$\operatorname{Var}(CAR_i) = \hat\sigma_i^2 \sum_{t\in W}\left[1 + \frac{1}{L} + \frac{(R_{mt} - \bar R_m)^2}{\sum_{s\in E}(R_{ms} - \bar R_m)^2}\right],$$

where $W$ is the test window, $E$ the estimation window and $\bar R_m$ the estimation-window mean of the market return. The standardised CAR is $SCAR_i = CAR_i / \sqrt{\operatorname{Var}(CAR_i)}$.

### 2.4 Firm-level inference

**Corrado (1989) rank test.** The estimation- and test-window abnormal returns are pooled and ranked ($T$ observations in total):

$$z = \frac{\bar K_W - (T+1)/2}{\sqrt{(T^2-1)/12}}\sqrt{L_W},$$

where $\bar K_W$ is the mean rank of the test-window observations and $L_W$ is the test-window length. This test is a per-firm diagnostic only and is not naively pooled across firms.

**iid residual bootstrap (secondary inference).** Draw $L_W$ estimation-window residuals with replacement and sum them, repeating $B = 1999$ times:

$$p = \frac{\#\{|CAR^*| \ge |CAR|\} + 1}{B + 1}.$$

**Calendar placebo (primary inference).** Up to 999 pseudo-event centres are drawn without replacement from the admissible set, and pseudo-CARs over windows of the same length give a two-sided p-value. A pseudo-centre $c$ must satisfy all four conditions:
1. $|c - e| > 40$;
2. its window does not overlap the true test window, i.e. $|c - e| > \tau_2 - \tau_1$;
3. $c$ is not within ±5 trading days of the MH17 date;
4. its window does not cover MH17 ±5 trading days.

The tables report the number of draws actually made and the minimum resolvable p-value $1/(n+1)$.

**Multiple testing.** Within each test window, the placebo p-values of the 75 firms are controlled by the Benjamini-Hochberg FDR procedure at 5%.

### 2.5 Aggregate inference

Let $N$ be the number of valid firms, $\overline{CAR}$ and $s_{CAR}$ the cross-sectional mean and standard deviation of CAR, and likewise $\overline{SCAR}$ and $s_{SCAR}$.

| Test | Statistic | Reference distribution |
|---|---|---|
| Plain-CS | $t = \overline{CAR}\,/\,(s_{CAR}/\sqrt N)$ | $t_{N-1}$ |
| Patell (1976) | $z = \sum_i SCAR_i\,/\,\sqrt{\sum_i (L_i-2)/(L_i-4)}$ | $N(0,1)$ |
| BMP (1991) | $t = \overline{SCAR}\,/\,(s_{SCAR}/\sqrt N)$ | $t_{N-1}$ |
| ADJ-BMP (Kolari & Pynnönen 2010) | $t_{adj} = t_{BMP}\sqrt{\dfrac{1-\bar r}{1+(N-1)\bar r}}$ | $t_{N-1}$ |

Here $\bar r$ is the mean pairwise correlation of estimation-window standardised residuals (off-diagonal elements only). Because all firms share a single calendar event date, cross-sectional correlation makes the uncorrected tests over-reject, so ADJ-BMP is the primary test.

Two standard errors on the CAR scale are also computed, for the MDEs in Chapter 8:

$$SE_{plain} = s_{CAR}/\sqrt N, \qquad SE_{KP} = SE_{plain}\sqrt{\frac{1+(N-1)\bar r}{1-\bar r}}.$$

### 2.6 Event-time CAAR (Figure 4-1)

For $\tau \in [-10, +120]$ the project computes $CAAR(\tau) = \frac1N\sum_i \sum_{t=-10}^{\tau} AR_{it}$ with a cross-sectional 95% band. The high-exposure group (direct exposure plus Asia-Pacific airlines, 13 firms) is reported separately from the remaining firms.

### 2.7 Mechanism regressions (Tables 4-2 and 6-4)

The long-window CAR is regressed by OLS with HC3 robust standard errors:

$$CAR_i = a + b\cdot \text{ExposureRank}_i + \sum_k c_k\,\text{Channel}_{ik} + d\cdot \text{Large}_i + \sum_g f_g\,\text{Region}_{ig} + u_i.$$

- Exposure rank: direct = 3, Asia-Pacific airline = 2, regional airline = 1, global airline = 0. Non-airline channels are coded 0, so $b$ identifies the exposure gradient within airlines.
- Channel dummies: airport, tourism/hotel and insurance/finance (baseline: airlines).
- Region fixed effects: Asia, Europe and Oceania (baseline: Americas).

Table 6-4 uses $\log AVR_{120}$ as the dependent variable and adds a size × exposure interaction.

**Size contrasts.** The difference in means between large and mid/small firms is tested with 9999 permutations, for the full sample and within the airport and tourism/hotel channels; $p = (\text{count}+1)/(9999+1)$.

### 2.8 Robustness (Table 4-3)

The alternative estimation windows are $[e-100, e-20]$ and $[e-150, e-10]$, with the test window extended to $[-5, +40]$.

---

## 3. Multi-factor Normal-Return Models (Table 4-4, `factor_models.py`)

**USD excess returns.** Local prices are multiplied by the USD-per-local-currency rate to obtain USD prices. FX rates are carried backward onto stock trading days, looking back at most 4 calendar days. USD log returns are then computed and the US risk-free rate RF is subtracted: $y_{it} = r^{USD}_{it} - RF_t$.

**Specifications.**

$$\text{FF3:}\quad y_{it} = a_i + b_i\,MKT_t + s_i\,SMB_t + h_i\,HML_t + \varepsilon_{it}$$

$$\text{FF5:}\quad \text{FF3 plus } RMW_t \text{ and } CMA_t.$$

**Factor sources.**
- US, Canadian and ADR listings use the Kenneth French North America factors.
- Europe, Japan and developed Asia-Pacific markets (Australia, Hong Kong, Singapore, New Zealand) use the matching French regional factors.
- China, Malaysia, South Korea, Turkey and Thailand use JKP country theme factors (value-weighted), with no developed-market proxy.

**Estimation and testing.**
- The estimation window is the same as for the market model, counted on the merged stock–factor calendar, and at least 120 observations are required.
- The residual variance uses $L - k - 1$ degrees of freedom.
- The CAR variance includes the multi-factor prediction error:

$$\operatorname{Var}(CAR_i) = \hat\sigma_i^2\left(L_W + \iota' X_W (X'X)^{-1} X_W' \iota\right),$$

  where $X$ is the estimation-window design matrix (with a constant), $X_W$ the test-window design matrix and $\iota$ a vector of ones.
- Firm-level p-values are two-sided normal p-values, with BH-FDR applied within each model × window family.
- Aggregate tests are the same as in Section 2.5, with $\bar r$ taken from the multi-factor estimation-window residuals.

---

## 4. Volatility Tests (Ch. 6, `event_study.py`)

**Abnormal volatility ratio (AVR).** Using market-model abnormal returns,

$$AVR_i^{(h)} = \frac{\sigma\!\left(AR_{i,[+1,+h]}\right)}{\sigma\!\left(AR_{i,[-180,-61]}\right)}, \qquad h \in \{60, 120\},$$

and $\log AVR$ is reported as well. The pre-window robustness check uses three alternative pre-windows: $[-150,-61]$, $[-120,-61]$ and $[-60,-1]$. If a pre- or post-window falls outside the sample, the firm is recorded as failed rather than truncated.

**Difference-in-differences (DiD).** The treated group is the direct-exposure and Asia-Pacific airlines (13 firms); the control group is the remaining 62 firms:

$$\text{DiD} = \overline{\log AVR_{120}}^{\,treated} - \overline{\log AVR_{120}}^{\,control},$$

with inference from 9999 permutations. A market-relative version is also reported, using $\log AVR_{120,i} - \log AVR_{120,\text{bench}(i)}$ as the outcome; the benchmark term is the volatility ratio of the benchmark index's raw returns.

---

## 5. GARCH and VaR Backtesting (Ch. 7, `garch_var.py`)

### 5.1 Model

A GARCH(1,1) is fitted to percentage returns $100\,r_t$:

$$100\,r_t = \mu + \varepsilon_t,\quad \varepsilon_t = \sigma_t z_t,\quad \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2.$$

Three innovation distributions are fitted for $z_t$: standard normal, standardised Student-t (degrees of freedom $\nu$) and Hansen's skewed-t ($\eta, \lambda$).

**Distribution choice: penalised BIC.**

$$\text{PBIC} = \text{BIC} + 10^9 \cdot \mathbb 1\{\alpha+\beta \ge 0.99 \ \text{or}\ \beta > 0.99\}.$$

The penalty steers selection away from near-unit-root solutions; firms whose selected model still lies on this boundary are flagged `PARAMETER_BOUNDARY_WARNING`. A 5-lag ARCH-LM test is also run on the standardised residuals of the selected model.

### 5.2 Out-of-sample VaR

The scheme uses an expanding window. The first 252 observations are a burn-in period; after that, every 21 trading days all three distributions are refitted on all prior data and PBIC selection is repeated. Between refits, the conditional variance is updated one step at a time with the estimated parameters: $\hat\sigma^2_{t} = \hat\omega + \hat\alpha\,\hat\varepsilon_{t-1}^2 + \hat\beta\,\hat\sigma^2_{t-1}$. The VaR is

$$\text{VaR}_t^{(p)} = -\left(\hat\mu + \hat\sigma_t\, q_p\right),$$

where $q_p$ is the standardised quantile of the selected distribution, at confidence levels of 95%, 97.5%, 99% and 99.5%. An exceedance occurs when $r_t < -\text{VaR}_t$.

### 5.3 Backtests

**Kupiec (1995) unconditional coverage.** With $x$ exceedances in $n$ periods and nominal tail probability $p$:

$$LR_{uc} = -2\left[x\ln p + (n-x)\ln(1-p) - x\ln\hat\pi - (n-x)\ln(1-\hat\pi)\right] \sim \chi^2_1,\quad \hat\pi = x/n.$$

When $x = 0$, $LR_{uc} = -2n\ln(1-p)$.

**Christoffersen (1998) conditional coverage.** The independence statistic $LR_{ind}$ is built from the first-order transition counts $n_{ij}$ of the hit sequence, and $LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2_2$.

**Kratz-Lok-McNeil (2018) multinomial multi-level test.** The $K = 4$ VaR levels sort each period's outcome into $K+1 = 5$ buckets, and the Pearson statistic $X^2 = \sum_j (O_j - E_j)^2 / E_j$ is computed. The Nass (1959) small-sample correction is then applied:

$$c = \frac{2\,\mathbb E[X^2]}{\operatorname{Var}[X^2]},\qquad \mathbb E[X^2] = K,\qquad \operatorname{Var}[X^2] = 2K + \frac{\sum_j 1/p_j - (K+1)^2 - 2(K+1) + 2}{n},$$

$$c\,X^2 \sim \chi^2_{cK}.$$

Firms with a Nass-corrected p < 0.05 are flagged for inadequate multi-level coverage.

---

## 6. Contagion Tests (Ch. 5, `contagion_tests.py`)

### 6.1 Core pairs

There are 19 core pairs, centred on the three major Chinese airlines, AirAsia and Malaysia Airports, and extending to regional airlines, insurers and reinsurers. The list is `CORE_PAIRS` in `run_all.py`.

### 6.2 Pairwise DCC(1,1)

The inputs are the full-sample standardised residuals of the GARCH models selected in Section 5. Each pair is first aligned by the rule in Section 1 and then re-standardised with the fit-sample mean and standard deviation, giving $z_t$. Dynamic correlations follow

$$Q_t = (1-a-b)\,\bar Q + a\, z_{t-1}z_{t-1}' + b\, Q_{t-1},\qquad \rho_t = \frac{q_{12,t}}{\sqrt{q_{11,t}\,q_{22,t}}}.$$

$(a, b)$ are estimated from the correlation part of the log-likelihood (constants omitted):

$$\ell(a,b) = -\tfrac12 \sum_t \left[\ln(1-\rho_t^2) + \frac{z_{1t}^2 - 2\rho_t z_{1t} z_{2t} + z_{2t}^2}{1-\rho_t^2}\right].$$

- Optimisation: L-BFGS-B with $a \in [10^{-6}, 0.4]$, $b \in [10^{-6}, 0.998]$ and $a + b < 0.999$.
- Standard errors: QMLE standard errors from a numerical Hessian (step $10^{-5}$). They do not include first-stage GARCH estimation uncertainty.

**Test.** The statistic compares mean $\rho_t$ before and after the event, $\Delta = \bar\rho_{post} - \bar\rho_{pre}$, over three windows:

| Label | Pre-window | Post-window |
|---|---|---|
| short | [−20, −1] | [0, +20] |
| medium | [−40, −1] | [0, +40] |
| long | [−60, −1] | [0, +60] |

Inference uses a two-sided calendar placebo (999 draws). Pseudo-centres must lie outside the event ±60 band, and pseudo-windows may not overlap the true windows or cover MH17 ±5. BH-FDR is then applied within each window family.

**Pre-event correlation targeting (Table 5-3).** $\bar Q$ and the re-standardisation use pre-event data only, so that post-event information does not enter the long-run correlation.

### 6.3 Forbes-Rigobon (2002) heteroskedasticity-adjusted test

The test runs on raw returns, with pre-window $[e-w, e)$ and post-window $[e, e+w)$.

- Volatility increase: $\delta = \max\left(\frac{\sigma^2_{x,post}}{\sigma^2_{x,pre}} - 1,\ \frac{\sigma^2_{y,post}}{\sigma^2_{y,pre}} - 1,\ 0\right)$.
- Adjusted post-event correlation:

$$\rho^*_{post} = \frac{\rho_{post}}{\sqrt{1 + \delta\,(1-\rho_{post}^2)}},\qquad \Delta^* = \rho^*_{post} - \rho_{pre}.$$

- Inference: a one-sided placebo test, with the alternative that the adjusted correlation rises (contagion): $p = (\#\{\Delta^{*,placebo} \ge \Delta^*\} + 1)/(n+1)$, requiring at least 30 valid placebos. Pseudo-centres must lie outside $e \pm (w+5)$, and their windows may not overlap the true windows or cover MH17 ±5.
- Windows: $w = 40$ and $w = 60$ are confirmatory. The $w = 90$ window spans MH17 and is reported separately as the Malaysia Airlines disaster cluster (Table 5-4).
- Multiple testing: BH-FDR within each window.

---

## 7. Statistical Power and MDE (Ch. 8, `power_mde.py`)

**Minimum detectable effect.** For a 5% two-sided test at 80% power,

$$\text{MDE} = (z_{0.975} + z_{0.80})\cdot SE \approx 2.80\cdot SE.$$

| Target | SE used | Plausible effect |
|---|---|---|
| Aggregate CAR (Plain-CS) | $SE_{plain}$ | 3% |
| Aggregate CAR (BMP) | $|\overline{CAR}/t_{BMP}|$ (approximate conversion to the CAR scale) | 3% |
| Aggregate CAR (ADJ-BMP) | $SE_{KP}$ | 3% |
| Exposure gradient | HC3 SE (per rank) | 1% |
| Size moderation (logAVR) | HC3 SE | 0.10 |
| Within-channel size contrast | Pooled two-sample SE, with t quantiles in place of z | 3% |
| Aggregate logAVR | $s/\sqrt N$ | 0.10 |
| DCC shift | SD of placebo $\Delta$ ($L = 60$) | 0.10 |

**Effective sample size.** $N_{eff} = N / (1 + (N-1)\bar r)$.

**Verdict rule.**
1. If $|\text{observed}|/SE \ge 1.96$, the verdict is a significant effect (SIGNIFICANT_EFFECT).
2. Otherwise, if MDE ≤ the plausible effect, it is an informative null (INFORMATIVE_NULL).
3. Otherwise it is underpowered (UNDERPOWERED).

**Power of the VaR test.** With $n$ set to the median number of out-of-sample periods for the 99% VaR, 3000 binomial simulations at a true exceedance rate of 1.5% give the rejection rate of the Kupiec test at the 5% level.

---

## 8. Randomness and Reproducibility

- A single random seed, 3702014, is used throughout.
- In the event study and DCC, the placebo and bootstrap use independent random streams from `numpy.random.SeedSequence(3702014).spawn(2)`. Forbes-Rigobon and the permutation tests use separate generators initialised with the same seed.
- GARCH fits are cached by series content, so the tables, the multi-level backtest and the figures all share a single estimation.

---

## 9. References

- Boehmer, E., Musumeci, J., & Poulsen, A. B. (1991). Event-study methodology under conditions of event-induced variance. *Journal of Financial Economics*, 30(2), 253–272.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Corrado, C. J. (1989). A nonparametric test for abnormal security-price performance in event studies. *Journal of Financial Economics*, 23(2), 385–395.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- Engle, R. F. (2002). Dynamic conditional correlation. *Journal of Business & Economic Statistics*, 20(3), 339–350.
- Fama, E. F., & French, K. R. (1993; 2015). Common risk factors in the returns on stocks and bonds; A five-factor asset pricing model. *Journal of Financial Economics*.
- Forbes, K. J., & Rigobon, R. (2002). No contagion, only interdependence. *Journal of Finance*, 57(5), 2223–2261.
- Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, 35(3), 705–730.
- Jensen, T. I., Kelly, B., & Pedersen, L. H. (2023). Is there a replication crisis in finance? *Journal of Finance*, 78(5), 2465–2518.
- Kolari, J. W., & Pynnönen, S. (2010). Event study testing with cross-sectional correlation of abnormal returns. *Review of Financial Studies*, 23(11), 3996–4025.
- Kratz, M., Lok, Y. H., & McNeil, A. J. (2018). Multinomial VaR backtests. *Journal of Banking & Finance*, 88, 393–407.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Nass, C. A. G. (1959). The χ² test for small expectations in contingency tables. *Biometrika*, 46(3/4), 365–385.
- Patell, J. M. (1976). Corporate forecasts of earnings per share and stock price behavior. *Journal of Accounting Research*, 14(2), 246–276.
