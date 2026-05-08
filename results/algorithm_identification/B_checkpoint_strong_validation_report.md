# B 组 comprehension cell 的 Fourier 算法强验证报告

生成日期：2026-05-08  
项目：Mini-Qwen modular-addition grokking / comprehension 机制识别

## 1. 目标

已有 grid search 已经在 B/C 组发现多个 `comprehend` cell。本报告的目标是：

1. 根据 grid search 得到的 `alpha` 与 `lambda`，列出应该保存训练 checkpoint 的实验组；
2. 实际训练 B 组代表性 `comprehend` cell，保存模型 checkpoint；
3. 对 B 组代表性 checkpoint 进行强验证：Fourier 模式保留 / 删除、logit 三维 FFT、循环等变性；
4. 给出数学解释与机制判断。

本轮新增脚本：

- `scripts/train_checkpoint.py`：训练单个 grid cell，并保存可复用 `.pt` checkpoint；
- `scripts/probe_fourier_checkpoint.py`：对 checkpoint 做 Fourier causal probe。

## 2. 需要保存 checkpoint 的实验组清单

优先级依据：

- `phase=comprehend/grok`，说明模型确实泛化；
- `fourier_score` 高，说明 trajectory-level Fourier evidence 强；
- 保留 weak-evidence / memorize / fail control，用于排除“所有泛化都自动 Fourier”或“embedding 有 Fourier 结构但不泛化”的混淆。

### 2.1 本轮最终建议保存 checkpoint 的 4 个实验组

在已经完成两个 B 组 P0 representative checkpoint 后，下一批最有信息量的是以下 4 个实验组。它们不是简单按 `fourier_score` 排序，而是覆盖四类互补假设：weak-evidence comprehension、grok boundary、memorize control、C 组强 Fourier reference。

机器可读清单已生成：

- `results/algorithm_identification/recommended_checkpoint_groups.csv`
- `results/algorithm_identification/recommended_checkpoint_groups.json`

| priority | experiment id | group | split | tied | alpha | lambda | seed | grid-search phase | fourier score | top canonical freq | role | checkpoint tag |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|
| P1 | `B_weakev_comp_a0p2_lam1` | B | S3 | true | 0.2 | 1.0 | 0 | comprehend | 26.76 | 30 | weak-evidence comprehension control | `B_weakev_comp_a0p2_lam1_seed0` |
| P1 | `B_grok_boundary_a0p4_lam0p01` | B | S3 | true | 0.4 | 0.01 | 0 | grok | 50.44 | 41 | grok-boundary mechanism control | `B_grok_boundary_a0p4_lam0p01_seed0` |
| P1 | `B_memorize_fourier_control_a0p1_lam0p316228` | B | S3 | true | 0.1 | 0.316228 | 0 | memorize | 41.46 | 21 | Fourier-like embedding without generalization control | `B_memorize_fourier_control_a0p1_lam0p316228_seed0` |
| Ref | `C_strong_ref_a0p7_lam0p031623` | C | S3 | false | 0.7 | 0.031623 | 0 | comprehend | 100.00 | 5 | untied strong-Fourier reference | `C_strong_ref_a0p7_lam0p031623_seed0` |

选择理由：

1. `B_weakev_comp_a0p2_lam1`：模型已经 comprehension，但 trajectory-level Fourier evidence 弱；这是最关键的“可能非 Fourier / 或 Fourier 更分散”候选。
2. `B_grok_boundary_a0p4_lam0p01`：与已验证 `alpha=0.4, lambda=0.1` 共享主频族，但属于 grok boundary；检验 delayed generalization 是否仍是同一 additive-line Fourier 机制。
3. `B_memorize_fourier_control_a0p1_lam0p316228`：embedding 指标 moderate Fourier-like，但 phase 是 memorize；用于排除“embedding 看起来 Fourier-like 就等于学到算法”的误判。
4. `C_strong_ref_a0p7_lam0p031623`：C 组 untied 的强 Fourier reference，用于校准 B 组 tied 模型的 probe 强度与 additive-line energy 上限。

建议统一保存到 `results/checkpoints/`，并使用 top-10 Fourier probe 输出到 `results/algorithm_identification/`。训练命令与 probe 命令已经写入上述 JSON/CSV manifest。

### 2.2 P0：本轮已训练并强验证的 B 组代表 cell

| priority | group | split | tied | alpha | lambda | seed | grid-search phase | reason | status |
|---|---|---|---|---:|---:|---:|---|---|---|
| P0 | B | S3 | true | 0.4 | 0.1 | 0 | comprehend | B 组最高 Fourier evidence；`fourier_score=50.50`，`circularity=0.1615` | ✅ trained + probed |
| P0 | B | S3 | true | 0.3 | 0.1 | 0 | comprehend | B 组 comprehension plateau 中心；更典型的 S3+tied 泛化 cell | ✅ trained + probed |

### 2.3 P1：建议后续补 checkpoint 的 B 组强 / 中等 Fourier evidence cell

| priority | group | split | tied | alpha | lambda | phase | fourier_score | top canonical freq | reason |
|---|---|---|---|---:|---:|---|---:|---:|---|
| P1 | B | S3 | true | 0.2 | 0.1 | comprehend | 47.10 | 18 | 低 alpha 仍能 comprehension，检验数据稀疏时是否同一 Fourier 机制 |
| P1 | B | S3 | true | 0.2 | 0.316228 | comprehend | 44.99 | 40 | 与上一个 cell 相近但主频不同，检验频率选择稳定性 |
| P1 | B | S3 | true | 0.4 | 0.031623 | comprehend | 44.50 | 41 | 与已验证 `alpha=0.4, lambda=0.1` 同主频族，用于 weight-decay 对照 |
| P1 | B | S3 | true | 0.3 | 0.316228 | comprehend | 39.42 | 30 | plateau 中心附近的 lambda 对照 |

### 2.4 P2：B 组 weak-evidence / boundary controls

| priority | group | split | tied | alpha | lambda | phase | fourier_score | reason |
|---|---|---|---|---:|---:|---|---:|---|
| P2 | B | S3 | true | 0.2 | 1.0 | comprehend | 26.76 | comprehension 但 weak Fourier evidence；检验是否存在非-Fourier 泛化候选 |
| P2 | B | S3 | true | 0.3 | 3.162278 | comprehend | 25.22 | high lambda weak evidence control |
| P2 | B | S3 | true | 0.3 | 0.01 | comprehend | 24.92 | low lambda weak evidence control |
| P2 | B | S3 | true | 0.4 | 0.01 | grok | 50.44 | grok boundary；检验 delayed generalization 的机制是否同 Fourier |
| P2 | B | S3 | true | 0.2 | 0.031623 | grok | 37.29 | low-alpha grok boundary |
| P2 | B | S3 | true | 0.1 | 0.316228 | memorize | 41.46 | 有 Fourier-like embedding 但不泛化，用于排除“Fourier-like embedding 足够说明算法” |
| P2 | B | S3 | true | 0.1 | 3.162278 | fail | 19.55 | no-generalization negative control |

### 2.5 A/C 参考 checkpoint

| priority | group | split | tied | alpha | lambda | phase | fourier_score | reason |
|---|---|---|---|---:|---:|---|---:|---|
| Ref | C | S3 | false | 0.7 | 0.031623 | comprehend | 100.00 | C 组强 Fourier-like reference |
| Ref | C | S3 | false | 0.5 | 0.316228 | comprehend | 99.86 | C 组强 Fourier-like reference，主频不同 |
| Ref | A | S1 | true | 0.3 | 1.0 | grok | 67.85 | A 组 grok reference |
| Ref | A | S1 | true | 0.3 | 3.162278 | comprehend | 33.18 | A 组 weak Fourier comprehension reference |

## 3. 本轮实际调用的训练与验证脚本

### 3.1 已训练 checkpoint

1. B 组最高 Fourier evidence representative：
   - `group=B, split=S3, tied=true, p=113, alpha=0.4, lambda=0.1, seed=0`
   - checkpoint: `results/checkpoints/B_repr_a0p4_lam0p1_seed0.pt`
   - metadata: `results/checkpoints/B_repr_a0p4_lam0p1_seed0.json`
   - wall time: 599.0 s
   - final phase: `comprehend`
   - `t_train=1000`, `t_test=1000`, `final_step=100000`
   - final train acc = 1.0, final test acc = 1.0

2. B 组中心 plateau representative：
   - `group=B, split=S3, tied=true, p=113, alpha=0.3, lambda=0.1, seed=0`
   - checkpoint: `results/checkpoints/B_center_a0p3_lam0p1_seed0.pt`
   - metadata: `results/checkpoints/B_center_a0p3_lam0p1_seed0.json`
   - wall time: 335.8 s
   - final phase: `comprehend`
   - `t_train=1000`, `t_test=1000`, `final_step=100000`
   - final train acc = 1.0, final test acc = 1.0

### 3.2 已生成强验证产物

| artifact | description |
|---|---|
| `results/algorithm_identification/B_repr_a0p4_lam0p1_probe.json` | `alpha=0.4` top-3 Fourier probe |
| `results/algorithm_identification/B_repr_a0p4_lam0p1_probe_top5.json` | `alpha=0.4` top-5 Fourier probe |
| `results/algorithm_identification/B_repr_a0p4_lam0p1_probe_top10.json` | `alpha=0.4` top-10 Fourier probe |
| `results/algorithm_identification/B_repr_a0p4_lam0p1_probe_top15.json` | `alpha=0.4` top-15 Fourier probe |
| `results/algorithm_identification/B_center_a0p3_lam0p1_probe_top10.json` | `alpha=0.3` top-10 Fourier probe |
| `results/algorithm_identification/B_probe_summary.csv` | 汇总 CSV |
| `results/algorithm_identification/B_probe_fourier_ablation_summary.png` | top-k 保留/删除准确率汇总图 |
| `results/algorithm_identification/recommended_checkpoint_groups.csv` | 下一批 4 个建议保存 checkpoint 实验组的表格清单 |
| `results/algorithm_identification/recommended_checkpoint_groups.json` | 下一批 4 个实验组的机器可读 manifest，含 train/probe 命令 |
| `results/algorithm_identification/recommended_checkpoint_runs.log` | 本轮 4 组 checkpoint 训练与 probe 的完整运行日志 |
| `results/algorithm_identification/recommended_checkpoint_probe_summary.csv` | 本轮 4 组原始 top-10 probe 汇总，keep 不额外保留 DC |
| `results/algorithm_identification/recommended_checkpoint_probe_variants.csv` | 本轮 4 组原始 probe 的全部 variant 明细 |
| `results/algorithm_identification/recommended_checkpoint_probe_dc_summary.csv` | 本轮 4 组 DC-preserving top-10 probe 汇总 |
| `results/algorithm_identification/recommended_checkpoint_probe_dc_variants.csv` | 本轮 4 组 DC-preserving probe 的全部 variant 明细 |
| `results/algorithm_identification/recommended_checkpoint_fourier_ablation_summary.png` | 原始 top-10 Fourier ablation 对比图 |
| `results/algorithm_identification/recommended_checkpoint_fourier_ablation_dc_summary.png` | DC-preserving top-10 Fourier ablation 对比图 |
| `results/algorithm_identification/B_fail_negative_a0p1_lam3p162278_train.log` | 真正 fail/no-generalization negative control 的训练日志 |
| `results/algorithm_identification/B_fail_negative_a0p1_lam3p162278_probe_top10.json` | fail negative control 的原始 top-10 Fourier probe |
| `results/algorithm_identification/B_fail_negative_a0p1_lam3p162278_probe_dc_top10.json` | fail negative control 的 DC-preserving top-10 Fourier probe |
| `results/algorithm_identification/B_fail_negative_a0p1_lam3p162278_probe_summary.csv` | fail negative control 的单独汇总表 |
| `results/algorithm_identification/checkpoint_controls_probe_summary.csv` | 4 组 controls + 真正 fail control 的统一汇总表 |
| `results/algorithm_identification/checkpoint_controls_probe_summary.json` | 4 组 controls + 真正 fail control 的统一 JSON 汇总 |
| `results/algorithm_identification/checkpoint_controls_fourier_ablation_dc_summary.png` | 加入真正 fail control 后的 DC-preserving 对比图 |

## 4. 数学机制：为什么 Fourier 是 modular addition 的自然算法

任务是模素数加法：

$$
c \equiv a+b \pmod p, \quad p=113.
$$

群 $\mathbb{Z}_p$ 的 Fourier characters 为：

$$
\chi_k(x)=\exp\left(\frac{2\pi i kx}{p}\right), \quad k=0,1,\dots,p-1.
$$

它们满足加法同态：

$$
\chi_k(a+b)=\chi_k(a)\chi_k(b).
$$

因此一个 Fourier addition circuit 可以用若干频率 $k\in K$ 构造 logit：

$$
L(a,b,c) \approx
\sum_{k\in K} A_k \cos\left(\frac{2\pi k(a+b-c)}{p}+\phi_k\right)+\text{bias}.
$$

当 $c=a+b\pmod p$ 时，上式各频率相位对齐，正确答案 logit 上升。

对完整 logit table 做三维 Fourier transform：

$$
\widehat{L}(u,v,w)=\sum_{a,b,c}L(a,b,c)
\exp\left(-\frac{2\pi i(ua+vb+wc)}{p}\right),
$$

若模型实现的是 Fourier addition，则非 DC 能量应集中在 additive line：

$$
(u,v,w)=(k,k,-k).
$$

本报告使用的 additive-line energy ratio 为：

$$
R_{\mathrm{add}}
=\frac{\sum_{k\ne0}|\widehat{L}(k,k,-k)|^2}
{\sum_{(u,v,w)\ne(0,0,0)}|\widehat{L}(u,v,w)|^2}.
$$

同时，对 embedding 做 Fourier 投影：

$$
(P_K E)(x)=\sum_{k\in K\cup(-K)}\widehat{E}_k
\exp\left(\frac{2\pi i kx}{p}\right).
$$

强验证分为：

- **Sufficiency**：只保留 $P_K E$ 后测试准确率仍高；
- **Necessity**：删除 $P_K E$，即使用 $(I-P_K)E$，测试准确率接近 chance $1/p$。

这里 chance accuracy 为：

$$
\frac{1}{113}\approx 0.00885.
$$

## 5. B 组强验证结果

### 5.1 `alpha=0.4, lambda=0.1`：最高 evidence representative

Baseline：train acc = 1.0，test acc = 1.0。

| Fourier projection | kept canonical freqs | keep test acc | remove test acc |
|---:|---|---:|---:|
| top-3 | 41, 30, 27 | 0.4464 | 0.1615 |
| top-5 | 41, 30, 27, 31, 19 | 0.6356 | 0.2059 |
| top-10 | 41, 30, 27, 31, 19, 18, 14, 53, 56, 36 | 0.8455 | 0.0254 |
| top-15 | 41, 30, 27, 31, 19, 18, 14, 53, 56, 36, 11, 9, 35, 48, 13 | 0.8201 | 0.0259 |

Logit FFT：

- additive-line energy ratio：`0.6913`
- top-12 FFT modes 全部满足 $(u,v,w)=(k,k,-k)$：
  - `(31,31,82)`, `(82,82,31)`
  - `(41,41,72)`, `(72,72,41)`
  - `(19,19,94)`, `(94,94,19)`
  - `(99,99,14)`, `(14,14,99)`
  - `(86,86,27)`, `(27,27,86)`
  - `(57,57,56)`, `(56,56,57)`

解释：

- top-3/top-5 已经携带大量功能，但不足以恢复完整算法；
- top-10 保留后 test acc = 0.8455，说明前 10 对 canonical Fourier modes 已经近似充分；
- top-10 删除后 test acc = 0.0254，接近 chance 0.00885，说明这些 Fourier modes 对泛化几乎必要；
- logit FFT 的 top modes 全在 additive line，排除了“只是 embedding 谱稀疏但算法不是加法 Fourier line”的主要疑虑。

### 5.2 `alpha=0.3, lambda=0.1`：B 组中心 plateau representative

Baseline：train acc = 1.0，test acc = 1.0。

| Fourier projection | kept canonical freqs | keep test acc | remove test acc |
|---:|---|---:|---:|
| top-10 | 38, 30, 14, 41, 46, 53, 37, 28, 31, 2 | 0.8785 | 0.0102 |

Logit FFT：

- additive-line energy ratio：`0.6101`
- top-12 FFT modes 全部满足 $(u,v,w)=(k,k,-k)$：
  - `(60,60,53)`, `(53,53,60)`
  - `(75,75,38)`, `(38,38,75)`
  - `(83,83,30)`, `(30,30,83)`
  - `(82,82,31)`, `(31,31,82)`
  - `(41,41,72)`, `(72,72,41)`
  - `(99,99,14)`, `(14,14,99)`

解释：

- top-10 保留后 test acc = 0.8785，说明 Fourier 子空间近似充分；
- top-10 删除后 test acc = 0.0102，几乎等于 chance 0.00885，说明这些 Fourier modes 对泛化必要；
- 中心 cell 的强验证比最高分 cell 更干净：删除 top-10 后几乎完全 collapse 到 chance。

## 6. 新增 4 组 checkpoint 的实测验证

根据 `recommended_checkpoint_groups.json`，本轮已经在当前服务器上实际完成 4 组训练与 probe。完整日志在：

- `results/algorithm_identification/recommended_checkpoint_runs.log`

### 6.1 训练结果

| experiment id | grid phase | trained phase | alpha | lambda | t_train | t_test | final test acc | wall time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `B_weakev_comp_a0p2_lam1` | comprehend | comprehend | 0.2 | 1.0 | 200 | 1000 | 0.9996 | 253.3 s |
| `B_grok_boundary_a0p4_lam0p01` | grok | comprehend | 0.4 | 0.01 | 200 | 500 | 1.0000 | 425.5 s |
| `B_memorize_fourier_control_a0p1_lam0p316228` | memorize | memorize | 0.1 | 0.316228 | 200 | — | 0.9863 | 258.4 s |
| `C_strong_ref_a0p7_lam0p031623` | comprehend | comprehend | 0.7 | 0.031623 | 1000 | 1000 | 1.0000 | 729.3 s |

注意：`B_grok_boundary_a0p4_lam0p01` 在原 grid search 中被标为 `grok`，但本轮重训很早达到 test threshold，因此本轮 checkpoint 的实际 phase 是 `comprehend`。这说明该 boundary cell 对具体运行存在一定非确定性/边界敏感性；在本报告中应把它解释为 **low-lambda boundary control**，而不是稳定的 delayed-grok control。

`B_memorize_fourier_control_a0p1_lam0p316228` 的 `t_test=None`，因为日志点上 test accuracy 没有达到 0.99 threshold；但最终 test acc 已达到 0.9863，说明它并不是干净的失败负例，而是接近 threshold 的 near-comprehension / unstable control。

### 6.2 原始 top-10 probe 结果

原始 probe 的 `keep` 干预只保留非 DC Fourier modes：

$$
P_K E = \sum_{k\in K\cup(-K)}\widehat E_k\chi_k,
$$

因此会去掉 numeric embedding 的 DC/均值分量。结果如下：

| experiment id | baseline test acc | keep top-10 test acc | remove top-10 test acc | additive-line energy | top-12 additive modes |
|---|---:|---:|---:|---:|---:|
| `B_weakev_comp_a0p2_lam1` | 0.9996 | 0.8133 | 0.0731 | 0.6454 | 12/12 |
| `B_grok_boundary_a0p4_lam0p01` | 1.0000 | 0.0000 | 0.0094 | 0.4149 | 12/12 |
| `B_memorize_fourier_control_a0p1_lam0p316228` | 0.9862 | 0.6293 | 0.0320 | 0.4882 | 12/12 |
| `C_strong_ref_a0p7_lam0p031623` | 1.0000 | 0.9563 | 0.0083 | 0.8387 | 12/12 |

解释：

- `B_weakev_comp_a0p2_lam1`：虽然 trajectory-level evidence 被标为 weak，但 checkpoint-level probe 显示它仍有强 Fourier causal signature；删除 top-10 后 test acc 降到 0.0731。
- `B_grok_boundary_a0p4_lam0p01`：删除 top-10 后几乎 chance，但非 DC keep top-10 为 0。额外 top15/top20/top30/top56 sweep 仍为 0，说明这是去掉 DC/均值后的干预伪影，而不是 top-k 太小。
- `B_memorize_fourier_control_a0p1_lam0p316228`：删除 top-10 后降到 0.0320，但 baseline 只有 0.9862，说明它已经形成了相当强的 Fourier-like 泛化结构，只是未稳定跨过 0.99 threshold。
- `C_strong_ref_a0p7_lam0p031623`：强参考最干净，keep top-10 仍有 0.9563，remove top-10 接近 chance。

### 6.3 DC-preserving top-10 probe 结果

为了排除 DC/均值分量导致的 sufficiency false negative，本轮额外做了 DC-preserving keep probe：

$$
P^{+\mathrm{DC}}_K E = \bar E + \sum_{k\in K\cup(-K)}\widehat E_k\chi_k,
\quad
\bar E=\frac{1}{p}\sum_{x\in\mathbb Z_p}E(x).
$$

这更符合机制问题：DC 分量只提供全局偏置/基线，不携带模加法的输入依赖；算法性输入依赖仍由非零 Fourier modes 承担。

| experiment id | baseline test acc | keep top-10 + DC test acc | remove top-10 test acc | interpretation |
|---|---:|---:|---:|---|
| `B_weakev_comp_a0p2_lam1` | 0.9996 | 0.8784 | 0.0731 | weak-evidence comprehension 仍主要由 top Fourier modes 支撑 |
| `B_grok_boundary_a0p4_lam0p01` | 1.0000 | 0.9793 | 0.0094 | low-lambda boundary 的 Fourier modes 必要且近似充分；原始 keep=0 是 DC removal artifact |
| `B_memorize_fourier_control_a0p1_lam0p316228` | 0.9862 | 0.9158 | 0.0320 | memorize/near-comprehension cell 已形成强但未完全稳定的 Fourier circuit |
| `C_strong_ref_a0p7_lam0p031623` | 1.0000 | 1.0000 | 0.0083 | C 组 untied strong reference，Fourier signature 最强 |

对 C 组 untied checkpoint，额外观察到：

- input embedding 的 top-10 删除：test acc = 0.0083；
- lm head 单独删除同一组频率：test acc = 1.0000；
- embedding + lm head 同时 DC-preserving keep top-10：test acc = 0.9945；
- embedding + lm head 同时删除 top-10：test acc = 0.0094。

这说明 C 组强 reference 的关键 causal bottleneck 主要在输入 number embedding 的 Fourier 表示；输出 head 对同一组 embedding-selected frequency 的单独删除并不构成瓶颈，但两侧同时删除会 collapse 到 chance。

### 6.4 真正 fail/no-generalization negative control

为了得到更干净的负例，本轮额外补充了：

- `group=B, split=S3, tied=true, p=113, alpha=0.1, lambda=3.162278, seed=0`
- checkpoint: `results/checkpoints/B_fail_negative_a0p1_lam3p162278_seed0.pt`
- metadata: `results/checkpoints/B_fail_negative_a0p1_lam3p162278_seed0.json`
- training log: `results/algorithm_identification/B_fail_negative_a0p1_lam3p162278_train.log`

该实验使用 adaptive protocol：`T_min=100000, T_max=1000000`。由于模型到 `100000` step 仍未达到 train threshold，训练自动延长到 `1000000` step。最终结果：

| alpha | lambda | trained phase | t_train | t_test | final step | final train acc | final test acc | wall time |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.162278 | fail | — | — | 1,000,000 | 0.9179 | 0.1502 | 2429.4 s |

这是真正意义上的 fail/no-generalization control：既没有达到 train threshold，也没有达到 test threshold；最终 test acc 远低于 comprehension cells。

对应 Fourier probe：

| baseline test acc | raw keep top-10 test acc | keep top-10 + DC test acc | remove top-10 test acc | additive-line energy | top-12 additive modes | chance |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1502 | 0.0979 | 0.1307 | 0.0128 | 0.0707 | 6/12 | 0.00885 |

解释：

- fail control 的 additive-line energy 只有 `0.0707`，远低于 B comprehension controls 的 `0.4149–0.6454`，也远低于 C strong reference 的 `0.8387`；
- top-12 FFT modes 中只有 `6/12` 落在 additive line，而成功泛化 cells 是 `12/12`；
- DC-preserving keep top-10 只有 `0.1307`，不能恢复泛化；
- remove top-10 后降到 `0.0128`，接近 chance，但由于 baseline 本身只有 `0.1502`，这表示模型中虽有少量可用 Fourier-like 信号，却没有形成可泛化的完整 Fourier addition circuit。

因此，这个负例支持一个更严格的判断：**Fourier-like 频率能量本身不充分；成功泛化需要这些频率组织成 additive-line logit circuit，并在 causal probe 中表现出高 sufficiency 与 high necessity。**

### 6.5 五组 controls 的统一对比

加入真正 fail negative control 后，统一汇总如下：

| experiment id | phase | baseline | keep top-10 + DC | remove top-10 | additive-line energy | top-12 additive modes |
|---|---|---:|---:|---:|---:|---:|
| `B_weakev_comp_a0p2_lam1` | comprehend | 0.9996 | 0.8784 | 0.0731 | 0.6454 | 12/12 |
| `B_grok_boundary_a0p4_lam0p01` | comprehend | 1.0000 | 0.9793 | 0.0094 | 0.4149 | 12/12 |
| `B_memorize_fourier_control_a0p1_lam0p316228` | memorize / near-comprehension | 0.9862 | 0.9158 | 0.0320 | 0.4882 | 12/12 |
| `C_strong_ref_a0p7_lam0p031623` | comprehend | 1.0000 | 1.0000 | 0.0083 | 0.8387 | 12/12 |
| `B_fail_negative_a0p1_lam3p162278` | fail | 0.1502 | 0.1307 | 0.0128 | 0.0707 | 6/12 |

这张表提供了目前最强的正负对照：泛化成功或 near-comprehension 的模型普遍有高 additive-line concentration 和 top-10 Fourier causal dependence；真正 fail control 没有这种结构。

## 7. 更新后的结论

综合两个 P0 representative checkpoint、本轮新增 4 组 checkpoint，以及真正 fail/no-generalization negative control，当前证据支持以下结论：

1. **B 组 2 层 mini-Qwen 在 comprehension 区域主要学到的是 Fourier 型 modular-addition 算法。**
   证据包括：
   - 只保留 top Fourier modes 仍能保持高 test accuracy；
   - 删除 top Fourier modes 后 test accuracy 接近 chance；
   - logit table 的非 DC Fourier 能量高度集中在 additive line $(k,k,-k)$；
   - top FFT modes 全部是 additive-line modes。

2. **算法不是单频 Fourier，而是多频 Fourier circuit，并且 DC/均值分量应在 sufficiency probe 中保留。**
   对 P0 cells，top-10 频率恢复 84.6% / 87.8% test accuracy；对新增 4 组，DC-preserving top-10 恢复 87.8% / 97.9% / 91.6% / 100.0% test accuracy。删除 top-10 后分别降到 0.0731 / 0.0094 / 0.0320 / 0.0083。

3. **“其他算法”解释在当前 B 组代表与扩展 cells 上不占主导。**
   如果模型主要使用 lookup、非 Fourier 插值或局部启发式，删除 embedding 中 top Fourier modes 不应让 test acc 接近 chance；logit FFT 也不应把 top modes 几乎全部放在 $(k,k,-k)$ 上。

4. **weak-evidence comprehension cell 仍然是 Fourier，而不是明显非-Fourier 反例。**
   `B_weakev_comp_a0p2_lam1` 的 trajectory-level score 偏弱，但 checkpoint-level DC-preserving keep/remove 与 additive-line FFT 都支持 Fourier circuit。

5. **memorize control 不够“干净”，更像 near-comprehension Fourier circuit。**
   `B_memorize_fourier_control_a0p1_lam0p316228` 最终 test acc = 0.9862，且 Fourier 删除后 collapse，因此它不是理想 negative control，而是一个 near-comprehension control。

6. **真正 fail/no-generalization negative control 明确缺少完整 Fourier addition circuit。**
   `B_fail_negative_a0p1_lam3p162278` 的 final test acc = 0.1502，DC-preserving keep top-10 只有 0.1307，additive-line energy 只有 0.0707，top-12 FFT modes 只有 6/12 在 additive line。这与成功泛化 cells 的 12/12 additive-line top modes 形成清晰对照。

## 8. 后续建议：从个案强验证升级为区域统计

本轮 4 组 manifest 与追加的真正 fail/no-generalization negative control 均已实际执行完成。下一步建议从以下方向继续：

1. **对真正 fail negative control 做多 seed 复现。**
   `B alpha=0.1, lambda=3.162278` 已经在 seed=0 上复现 fail。建议补 seed=1/2/3，确认低 additive-line energy 与低 keep+DC sufficiency 是否稳定。

2. **对 boundary cells 做多 seed / 多 checkpoint 时间点保存。**
   `B alpha=0.4, lambda=0.01` 本轮从 grid 的 `grok` 变成快速 `comprehend`，说明边界区域对随机性或实现细节敏感。建议保存 step=100/200/500/1000/5000/20000/100000 的 trajectory checkpoints，观察 Fourier circuit 从 memorize 到 comprehension 的形成过程。

3. **将 DC-preserving probe 作为默认 sufficiency 测试。**
   非 DC Fourier modes 负责算法性输入依赖，但 DC/均值分量提供表示基线。后续报告应同时给出 raw keep 与 keep+DC，避免把 DC removal artifact 误判成非 Fourier。

4. **把强验证推广到更多 B 组 cells。**
   优先补 `alpha=0.2, lambda=0.1`、`alpha=0.2, lambda=0.316228`、`alpha=0.3, lambda=0.316228`，形成 B 组 comprehension 区域内的机制统计。

完成上述补充后，可以把当前结论进一步升级为：

- B 组 comprehension 内部：强/弱 evidence cell 的机制边界；
- B 组 phase 边界：comprehend/grok/memorize 的 causal Fourier 差异；
- B vs C：tied 与 untied 表示下 Fourier algorithm 的强度对照。