# 方法与模型说明

[English version](METHODS_en.md) · [返回项目说明](README_cn.md)

本文档逐一说明项目所用的计量方法、模型设定与参数取值，与 `mh370_study/` 中的代码实现一一对应。所有收益均为小数单位的对数收益；事件日记为 $e$，指 2014-03-10 在相应交易日历中的位置；若该日不是交易日，则取其后第一个交易日。

---

## 1. 数据构造与对齐（第 3 章，`data_alignment.py`）

**对数收益。** 每只资产在其本地交易日历上计算

$$r_t = \ln\left(P_t / P_{t-1}\right),$$

其中 $P_t$ 为复权收盘价。各资产按日期外连接，不做全样本共同交易日交集，缺失值表示该市场当日休市。

**资产与基准对齐。** 每家公司只与其上市地基准指数在两者的共同交易日上对齐。事件中心、估计窗与检验窗都在这一两两日历上按交易日计数。

**资产对对齐。** 资产对的处理分两种情况：
- 同一宽口径区域（美洲、亚洲、欧洲、大洋洲）内的资产对，使用同步收益；
- 跨区域的资产对，使用两日重叠收益 $r_t + r_{t-1}$（`overlap2`），以缓解收盘时差带来的非同步交易偏差。

**描述统计。** 报告均值、标准差、偏度、峰度（非超额），以及 1 阶与 5 阶 Engle (1982) ARCH-LM 检验 p 值（对去均值收益）。

---

## 2. 事件研究（第 4 章，`event_study.py`）

### 2.1 正常收益模型

采用市场模型：

$$R_{it} = \alpha_i + \beta_i R_{mt} + \varepsilon_{it},$$

在估计窗 $[e-210,\ e-30]$（$L = 181$ 个交易日）上用 OLS 估计。

- 残差方差：$\hat\sigma_i^2 = \sum \hat\varepsilon_{it}^2 / (L-2)$。
- 异常收益：$AR_{it} = R_{it} - \hat\alpha_i - \hat\beta_i R_{mt}$。

### 2.2 检验窗口

| 名称 | 窗口 | 名称 | 窗口 |
|---|---|---|---|
| short | [−1, +1] | long | [−5, +30] |
| sym3 | [−3, +3] | post20 | [+1, +20] |
| sym5 | [−5, +5] | post60 | [+1, +60] |
| med | [−5, +10] | post120 | [+1, +120] |

累计异常收益为 $CAR_i(\tau_1,\tau_2) = \sum_{t=\tau_1}^{\tau_2} AR_{it}$。

### 2.3 标准化 CAR

CAR 的方差包含预测误差项：

$$\operatorname{Var}(CAR_i) = \hat\sigma_i^2 \sum_{t\in W}\left[1 + \frac{1}{L} + \frac{(R_{mt} - \bar R_m)^2}{\sum_{s\in E}(R_{ms} - \bar R_m)^2}\right],$$

其中 $W$ 为检验窗，$E$ 为估计窗，$\bar R_m$ 为估计窗内的市场收益均值。标准化 CAR 为 $SCAR_i = CAR_i / \sqrt{\operatorname{Var}(CAR_i)}$。

### 2.4 公司层面推断

**Corrado (1989) 秩检验。** 将估计窗与检验窗的异常收益合并排序，共 $T$ 个观测：

$$z = \frac{\bar K_W - (T+1)/2}{\sqrt{(T^2-1)/12}}\sqrt{L_W}.$$

其中 $\bar K_W$ 为检验窗内观测的平均秩，$L_W$ 为检验窗长度。该检验只作为单公司诊断，不做跨公司的简单合并。

**iid 残差自助法（次要推断）。** 从估计窗残差中有放回地抽取 $L_W$ 个并求和，重复 $B = 1999$ 次：

$$p = \frac{\#\{|CAR^*| \ge |CAR|\} + 1}{B + 1}.$$

**日历安慰剂（主推断）。** 从可用的伪事件中心中无放回地抽取至多 999 个，计算同长度窗口的伪 CAR，得到双侧 p 值。伪中心 $c$ 必须同时满足四个条件：
1. $|c - e| > 40$；
2. 伪窗口与真实检验窗口不重叠，即 $|c - e| > \tau_2 - \tau_1$；
3. $c$ 不落在 MH17 日期 ±5 个交易日内；
4. 伪窗口不覆盖 MH17 ±5 个交易日。

表中同时报告实际抽样次数与最小可分辨 p 值 $1/(n+1)$。

**多重检验。** 在每个检验窗口内，对 75 家公司的安慰剂 p 值做 Benjamini-Hochberg FDR 控制，显著性水平为 5%。

### 2.5 聚合推断

设 $N$ 为有效公司数，$\overline{CAR}$ 与 $s_{CAR}$ 为截面均值与标准差，$\overline{SCAR}$ 与 $s_{SCAR}$ 同理。

| 检验 | 统计量 | 参考分布 |
|---|---|---|
| Plain-CS | $t = \overline{CAR}\,/\,(s_{CAR}/\sqrt N)$ | $t_{N-1}$ |
| Patell (1976) | $z = \sum_i SCAR_i\,/\,\sqrt{\sum_i (L_i-2)/(L_i-4)}$ | $N(0,1)$ |
| BMP (1991) | $t = \overline{SCAR}\,/\,(s_{SCAR}/\sqrt N)$ | $t_{N-1}$ |
| ADJ-BMP (Kolari & Pynnönen 2010) | $t_{adj} = t_{BMP}\sqrt{\dfrac{1-\bar r}{1+(N-1)\bar r}}$ | $t_{N-1}$ |

其中 $\bar r$ 为估计窗标准化残差的平均两两相关系数（只取非对角元）。由于所有公司共享同一个日历事件日，截面相关会使未校正的检验过度拒绝，因此以 ADJ-BMP 为主检验。

CAR 尺度上的标准误另外计算两种，供第 8 章计算 MDE 使用：

$$SE_{plain} = s_{CAR}/\sqrt N, \qquad SE_{KP} = SE_{plain}\sqrt{\frac{1+(N-1)\bar r}{1-\bar r}}.$$

### 2.6 事件时间 CAAR（图 4-1）

对 $\tau \in [-10, +120]$ 计算 $CAAR(\tau) = \frac1N\sum_i \sum_{t=-10}^{\tau} AR_{it}$，并给出截面 95% 置信带。高暴露组（直接暴露 + 亚太航空，共 13 家）与其余公司分组报告。

### 2.7 机制回归（表 4-2、6-4）

以长窗口 CAR 为因变量，做 HC3 稳健标准误的 OLS：

$$CAR_i = a + b\cdot \text{ExposureRank}_i + \sum_k c_k\,\text{Channel}_{ik} + d\cdot \text{Large}_i + \sum_g f_g\,\text{Region}_{ig} + u_i.$$

- 暴露等级：直接暴露 = 3，亚太航空 = 2，区域航空 = 1，全球航空 = 0。非航空渠道记为 0，因此 $b$ 识别的是航空公司内部的暴露梯度。
- 渠道哑变量：机场、旅游酒店、保险金融（基准为航空）。
- 区域固定效应：亚洲、欧洲、大洋洲（基准为美洲）。

表 6-4 以 $\log AVR_{120}$ 为因变量，并加入规模 × 暴露交互项。

**规模对照。** 比较大型与中小型公司的均值差：在全样本和机场、旅游酒店渠道内分别做 9999 次置换检验，$p = (\text{count}+1)/(9999+1)$。

### 2.8 稳健性（表 4-3）

替代估计窗取 $[e-100, e-20]$ 与 $[e-150, e-10]$，检验窗扩展为 $[-5, +40]$。

---

## 3. 多因子正常收益模型（表 4-4，`factor_models.py`）

**美元超额收益。** 将本币价格乘以美元/本币汇率换算为美元价格。汇率在股票交易日上向后取最近值，最多回溯 4 个自然日。然后计算美元对数收益，再减去美国无风险利率 RF：$y_{it} = r^{USD}_{it} - RF_t$。

**模型设定。**

$$\text{FF3:}\quad y_{it} = a_i + b_i\,MKT_t + s_i\,SMB_t + h_i\,HML_t + \varepsilon_{it}$$

$$\text{FF5:}\quad \text{在 FF3 基础上加入 } RMW_t \text{ 与 } CMA_t.$$

**因子来源。**
- 美国、加拿大、ADR 使用 Kenneth French 北美因子；
- 欧洲、日本、亚太发达市场（澳、港、新、新西兰）使用对应的 French 区域因子；
- 中国、马来西亚、韩国、土耳其、泰国使用 JKP 国别主题因子（市值加权），不用发达市场因子代替。

**估计与检验。**
- 估计窗与市场模型相同（在股票-因子合并日历上计数），至少需要 120 个观测。
- 残差方差的自由度为 $L - k - 1$。
- CAR 方差包含多因子预测误差：

$$\operatorname{Var}(CAR_i) = \hat\sigma_i^2\left(L_W + \iota' X_W (X'X)^{-1} X_W' \iota\right),$$

  其中 $X$ 为估计窗设计矩阵（含常数项），$X_W$ 为检验窗设计矩阵，$\iota$ 为全 1 向量。
- 公司层面 p 值取正态双侧值，并在模型 × 窗口内做 BH-FDR。
- 聚合检验与第 2.5 节相同，$\bar r$ 取自多因子模型的估计窗残差。

---

## 4. 波动率检验（第 6 章，`event_study.py`）

**异常波动比（AVR）。** 用市场模型异常收益计算：

$$AVR_i^{(h)} = \frac{\sigma\!\left(AR_{i,[+1,+h]}\right)}{\sigma\!\left(AR_{i,[-180,-61]}\right)}, \qquad h \in \{60, 120\},$$

并报告 $\log AVR$。前窗稳健性另取 $[-150,-61]$、$[-120,-61]$、$[-60,-1]$ 三种前窗。若前窗或后窗超出样本，该公司记为失败，不做截断。

**双重差分（DiD）。** 处理组为直接暴露与亚太航空公司（13 家），对照组为其余 62 家：

$$\text{DiD} = \overline{\log AVR_{120}}^{\,treated} - \overline{\log AVR_{120}}^{\,control},$$

用 9999 次置换检验推断。另报告相对市场的版本：以 $\log AVR_{120,i} - \log AVR_{120,\text{bench}(i)}$ 为结果变量，其中基准项用基准指数原始收益的波动比计算。

---

## 5. GARCH 与 VaR 回测（第 7 章，`garch_var.py`）

### 5.1 模型

对百分比收益 $100\,r_t$ 拟合 GARCH(1,1)：

$$100\,r_t = \mu + \varepsilon_t,\quad \varepsilon_t = \sigma_t z_t,\quad \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2.$$

新息 $z_t$ 依次取标准正态、标准化 Student-t（自由度 $\nu$）、Hansen 偏 t（$\eta, \lambda$）三种分布。

**分布选择：惩罚 BIC。**

$$\text{PBIC} = \text{BIC} + 10^9 \cdot \mathbb 1\{\alpha+\beta \ge 0.99 \ \text{或}\ \beta > 0.99\}.$$

罚项的作用是避免选中近单位根的解；最终触及该边界的公司标记为 `PARAMETER_BOUNDARY_WARNING`。对选中模型的标准化残差另做 5 阶 ARCH-LM 检验。

### 5.2 样本外 VaR

采用扩展窗口：前 252 个观测作为预热期，之后每 21 个交易日用此前全部数据重新拟合三种分布，并重新做 PBIC 选择。在两次重估之间，条件方差按估计参数一步递推：$\hat\sigma^2_{t} = \hat\omega + \hat\alpha\,\hat\varepsilon_{t-1}^2 + \hat\beta\,\hat\sigma^2_{t-1}$。VaR 为

$$\text{VaR}_t^{(p)} = -\left(\hat\mu + \hat\sigma_t\, q_p\right),$$

其中 $q_p$ 为所选分布的标准化分位数，置信水平取 95%、97.5%、99%、99.5%。当 $r_t < -\text{VaR}_t$ 时记为一次突破。

### 5.3 回测

**Kupiec (1995) 无条件覆盖。** 设 $x$ 为 $n$ 期中的突破次数，$p$ 为名义尾部概率：

$$LR_{uc} = -2\left[x\ln p + (n-x)\ln(1-p) - x\ln\hat\pi - (n-x)\ln(1-\hat\pi)\right] \sim \chi^2_1,\quad \hat\pi = x/n.$$

当 $x = 0$ 时取 $LR_{uc} = -2n\ln(1-p)$。

**Christoffersen (1998) 条件覆盖。** 由突破序列的一阶转移计数 $n_{ij}$ 构造独立性统计量 $LR_{ind}$，再令 $LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2_2$。

**Kratz-Lok-McNeil (2018) 多项式多水平检验。** 以 $K = 4$ 个 VaR 水平把每期结果划入 $K+1 = 5$ 个区间，计算 Pearson 统计量 $X^2 = \sum_j (O_j - E_j)^2 / E_j$。再用 Nass (1959) 小样本校正：

$$c = \frac{2\,\mathbb E[X^2]}{\operatorname{Var}[X^2]},\qquad \mathbb E[X^2] = K,\qquad \operatorname{Var}[X^2] = 2K + \frac{\sum_j 1/p_j - (K+1)^2 - 2(K+1) + 2}{n},$$

$$c\,X^2 \sim \chi^2_{cK}.$$

Nass 校正后 p < 0.05 的公司记为多水平覆盖不足。

---

## 6. 传染检验（第 5 章，`contagion_tests.py`）

### 6.1 核心资产对

共 19 对，以中国三大航空、亚航、马来西亚机场为中心，延伸到区域航空、保险与再保险公司。名单见 `run_all.py` 中的 `CORE_PAIRS`。

### 6.2 两两 DCC(1,1)

输入为第 5 节所选 GARCH 模型的全样本标准化残差，先按第 1 节的规则两两对齐，再用拟合样本的均值和标准差重新标准化得到 $z_t$。动态相关的递推为

$$Q_t = (1-a-b)\,\bar Q + a\, z_{t-1}z_{t-1}' + b\, Q_{t-1},\qquad \rho_t = \frac{q_{12,t}}{\sqrt{q_{11,t}\,q_{22,t}}}.$$

以相关部分的对数似然估计 $(a, b)$（省略常数项）：

$$\ell(a,b) = -\tfrac12 \sum_t \left[\ln(1-\rho_t^2) + \frac{z_{1t}^2 - 2\rho_t z_{1t} z_{2t} + z_{2t}^2}{1-\rho_t^2}\right].$$

- 优化方法：L-BFGS-B，约束为 $a \in [10^{-6}, 0.4]$，$b \in [10^{-6}, 0.998]$，$a + b < 0.999$。
- 标准误：由数值 Hessian（步长 $10^{-5}$）得到 QMLE 标准误，不包含第一阶段 GARCH 的估计不确定性。

**检验。** 比较事件前后 $\rho_t$ 的均值，$\Delta = \bar\rho_{post} - \bar\rho_{pre}$，三个窗口为：

| 名称 | 前窗 | 后窗 |
|---|---|---|
| short | [−20, −1] | [0, +20] |
| medium | [−40, −1] | [0, +40] |
| long | [−60, −1] | [0, +60] |

推断用双侧日历安慰剂（999 次）：伪中心须在事件日 ±60 以外，伪窗口不得与真实窗口重叠或覆盖 MH17±5；再在各窗口族内做 BH-FDR。

**事件前相关目标（表 5-3）。** $\bar Q$ 与重新标准化只使用事件前的样本，以避免事件后的信息进入长期相关。

### 6.3 Forbes-Rigobon (2002) 异方差校正检验

在原始收益上进行，前窗为 $[e-w, e)$，后窗为 $[e, e+w)$。

- 波动上升幅度：$\delta = \max\left(\frac{\sigma^2_{x,post}}{\sigma^2_{x,pre}} - 1,\ \frac{\sigma^2_{y,post}}{\sigma^2_{y,pre}} - 1,\ 0\right)$。
- 校正后的事件后相关：

$$\rho^*_{post} = \frac{\rho_{post}}{\sqrt{1 + \delta\,(1-\rho_{post}^2)}},\qquad \Delta^* = \rho^*_{post} - \rho_{pre}.$$

- 推断：单侧安慰剂检验（备择假设为校正后相关上升，即“传染”），$p = (\#\{\Delta^{*,placebo} \ge \Delta^*\} + 1)/(n+1)$，有效安慰剂不少于 30 个。伪中心须在 $e \pm (w+5)$ 以外，且窗口不与真实窗口重叠、不覆盖 MH17±5。
- 窗口：确认性窗口为 $w = 40$ 与 $w = 60$；$w = 90$ 窗口跨越 MH17，作为马航灾难簇单独报告（表 5-4）。
- 多重检验：在各窗口内做 BH-FDR。

---

## 7. 统计功效与 MDE（第 8 章，`power_mde.py`）

**最小可检测效应。** 在 5% 双侧检验、80% 功效下：

$$\text{MDE} = (z_{0.975} + z_{0.80})\cdot SE \approx 2.80\cdot SE.$$

| 目标 | SE 的取法 | 合理效应量 |
|---|---|---|
| 聚合 CAR（Plain-CS） | $SE_{plain}$ | 3% |
| 聚合 CAR（BMP） | $|\overline{CAR}/t_{BMP}|$（近似换算到 CAR 尺度） | 3% |
| 聚合 CAR（ADJ-BMP） | $SE_{KP}$ | 3% |
| 暴露梯度 | HC3 标准误（每级） | 1% |
| 规模调节（logAVR） | HC3 标准误 | 0.10 |
| 渠道内规模对照 | 两样本合并方差 SE，用 t 分位数代替 z | 3% |
| 聚合 logAVR | $s/\sqrt N$ | 0.10 |
| DCC 相关变化 | 安慰剂 $\Delta$ 的标准差（$L = 60$） | 0.10 |

**有效样本量。** $N_{eff} = N / (1 + (N-1)\bar r)$。

**判定规则。**
1. 若 $|\text{观测值}|/SE \ge 1.96$，判为显著效应（SIGNIFICANT_EFFECT）；
2. 否则，若 MDE ≤ 合理效应量，判为有信息的零结果（INFORMATIVE_NULL）；
3. 否则判为功效不足（UNDERPOWERED）。

**VaR 检验的功效。** 以 99% VaR 样本外期数的中位数为 $n$，按真实突破率 1.5% 做 3000 次二项模拟，计算 Kupiec 检验在 5% 水平下的拒绝率。

---

## 8. 随机性与可复现性

- 统一随机种子为 3702014。
- 事件研究与 DCC 中，安慰剂与自助法通过 `numpy.random.SeedSequence(3702014).spawn(2)` 生成相互独立的随机数流；Forbes-Rigobon 与各置换检验使用以同一种子初始化的独立生成器。
- GARCH 拟合结果按序列内容缓存，表格、多水平回测与作图共用同一次估计。

---

## 9. 参考文献

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
