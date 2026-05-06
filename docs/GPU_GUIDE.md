# GPU 使用手册 — Mini-Qwen Grokking 实验

> **场景**：你已经在 CPU 上完成了所有代码开发，现在切到 GPU 机器（推荐 RTX 4090 24GB / A100 40GB / Blackwell 32GB）跑真正的相图实验。

---

## 0. 前置确认

```bash
nvidia-smi        # 确认 GPU 可见
python3 --version # >= 3.10
git log --oneline | head
```

---

## 1. 环境安装（一次性，约 2–5 分钟）

```bash
cd /path/to/ai_exp
python3 -m pip install --upgrade pip setuptools wheel
pip install -e .
make smoke   # 4/4 通过 = 包安装成功
make test    # 全栈无回归（CPU 上数秒级）
```

**潜在坑**：

| 现象 | 修复 |
|---|---|
| `pip install -e .` 报 PEP 660 错误 | 升 pip ≥ 22；上面命令已含 |
| torch 装了 CPU 版 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `pyarrow` 不可用 | `pip install pyarrow pandas` |

---

## 2. M2 风险闸门（**先做**，约 3–10 分钟 GPU）

跑通**单个 cell**确认 grokking 曲线真的会出现，再花成本做全扫。

```bash
python3 scripts/train_one.py \
    --p 113 --alpha 0.3 --lambda 1.0 \
    --T-min 100000 --T-max 1000000 --device cuda
```

**预期输出**：
```
[setup] p=113 alpha=0.3 lambda=1.0 split=S1 tied=True | train=3831 test=8938
[result] phase=grok t_train=~200 t_test=~50000 final_step=~100000
[saved] results/single_cell/p113_a0.3_l1.0_splitS1_tiedTrue_seed0.json
```

**通过条件**：`phase=grok` 且 `t_test / t_train > 10`。

**未通过排查**：
- 全是 `memorize` → 试 `--lambda 3.0` 或 `--alpha 0.5`（落到 grok 区中心）
- 全是 `fail` → `--T-min 500000` 给更长预算
- 全是 `comprehend` → 这其实是好事，说明该 cell 在 grok+comprehend 边界

---

## 3. Group A 主扫描（baseline：S1 + tied embedding）

**完整扫描**（9×7=63 cells，每 cell 1 seed，推荐命令）：

```bash
python3 scripts/run_scan_instrumented.py \
    --group A --p 113 \
    --T-min 100000 --T-max 5000000 \
    --n-seeds 1 \
    --measures-steps 100 1000 10000 100000 500000 1000000 5000000 \
    --progress-interval-steps 100000 \
    --skip-hessian \
    --device cuda \
    --out results/scans/A_p113_phase1.parquet
```

**中断后续跑**（同一个 `--out`，加 `--resume`）：

```bash
python3 scripts/run_scan_instrumented.py \
    --group A --p 113 \
    --T-min 100000 --T-max 5000000 \
    --n-seeds 1 \
    --measures-steps 100 1000 10000 100000 500000 1000000 5000000 \
    --progress-interval-steps 100000 \
    --skip-hessian \
    --device cuda \
    --out results/scans/A_p113_phase1.parquet \
    --resume
```

**日志说明**：

- `[cell ... log]`：在 evaluation/checkpoint step 打印 train/test acc、`t_train`、`t_test`、当前 `T_target`。
- `[cell ... progress]`：轻量 heartbeat，不额外跑 eval；默认每 100k step 打印一次，可用 `--progress-interval-steps 0` 关闭。
- `[cell ... measures]`：该 step 已写入多少条 progress-measure row。
- `[partial]`：每个 cell 完成后写一次 `*.partial.parquet`，中断也能保留已完成 cell。

如果某个 cell 在 100k 还没达到训练阈值，adaptive-T 会继续跑到 500k、1M 甚至 5M；这时看到 heartbeat 但暂时没有 cell completion 是正常现象。

**预计耗时**（仅供参考，依赖 cell 实际收敛快慢）：

| GPU | 估算 |
|---|---|
| Blackwell 32GB | 低 alpha memorize cell 约 5–6 分钟；完整 63 cell 至少数小时，hard/fail cell 可能显著更久 |
| RTX 4090 24GB | 预计同量级或稍慢 |
| A100/H100 | 取决于小模型 full-batch 利用率，不能只按 FLOPS 线性估算 |

> Hessian 计算（power iteration ~10 次双 backward）会让单 cell 多跑 30-60s × N_seeds × N_measure_steps。N=1 + 7 measure steps + skip_hessian → 已可接受。要包含 Hessian，去掉 `--skip-hessian` 并预期 +60% 总耗时。

---

## 4. Group B 主贡献（**核心**：S3 + tied embedding）

从头跑：

```bash
python3 scripts/run_scan_instrumented.py \
    --group B --p 113 \
    --T-min 100000 --T-max 5000000 \
    --n-seeds 1 \
    --measures-steps 100 1000 10000 100000 500000 1000000 5000000 \
    --progress-interval-steps 100000 \
    --skip-hessian \
    --device cuda \
    --out results/scans/B_p113_phase1.parquet
```

续跑：

```bash
python3 scripts/run_scan_instrumented.py \
    --group B --p 113 \
    --T-min 100000 --T-max 5000000 \
    --n-seeds 1 \
    --measures-steps 100 1000 10000 100000 500000 1000000 5000000 \
    --progress-interval-steps 100000 \
    --skip-hessian \
    --device cuda \
    --out results/scans/B_p113_phase1.parquet \
    --resume
```

**关键观测**：在 S3 split 下，未见过的 b 值在测试时不会作为输入 token 出现 → 真正考验"学到群运算"而非"插值见过的 b"。预期 grok 区会比 Group A 小或不同。

---

## 5. Group C 消融（S3 + untied embedding）

从头跑：

```bash
python3 scripts/run_scan_instrumented.py \
    --group C --p 113 \
    --T-min 100000 --T-max 5000000 \
    --n-seeds 1 \
    --measures-steps 100 1000 10000 100000 500000 1000000 5000000 \
    --progress-interval-steps 100000 \
    --skip-hessian \
    --device cuda \
    --out results/scans/C_p113_phase1.parquet
```

续跑：

```bash
python3 scripts/run_scan_instrumented.py \
    --group C --p 113 \
    --T-min 100000 --T-max 5000000 \
    --n-seeds 1 \
    --measures-steps 100 1000 10000 100000 500000 1000000 5000000 \
    --progress-interval-steps 100000 \
    --skip-hessian \
    --device cuda \
    --out results/scans/C_p113_phase1.parquet \
    --resume
```

**预期**：测试集准确率应稳定在 ≈1/p（理论上限），证明 untied + S3 的"未见 token 测试"是架构性 tautology — 这正是 Group B 比 Group C 更有意义的理论依据。

---

## 6. Phase 2 边界细化（可选，提高相图清晰度）

Phase 1 用 1 seed/cell，边界处可能有噪声。对每个分类不一致的边界 cell 加跑 4 个 seed：

> **当前状态**：`run_phase2` 函数已在 `src/mqg/scan/scan_runner.py` 实现，但 instrumented 版本的 phase2 CLI 包装尚未写。等你 Phase 1 出结果后告诉我，5 分钟可以加一个 `scripts/refine_boundaries.py`。

---

## 7. 输出文件结构

```
results/scans/
├── A_p113_phase1.parquet           # trajectory (long-format, 1 行/(cell,seed,step))
├── A_p113_phase1.cells.parquet     # cell summary (1 行/(cell,seed))
├── A_p113_phase1.partial.parquet   # 中途进度（每完成 1 cell 刷新）
├── A_p113_phase1.partial.cells.parquet
├── B_p113_phase1.parquet
├── B_p113_phase1.cells.parquet
├── C_p113_phase1.parquet
└── C_p113_phase1.cells.parquet
```

**trajectory 列**（约 35–40 列）：

- 元数据：`group, split_strategy, tied_embedding, alpha_idx, lambda_idx, alpha, lam, seed, split_seed, step`
- Track 1：`fourier_sparsity, dom_freq_0..4, circularity_top_freq`
- Track 2：`weight_norm_total, embedding_stable_rank, embedding_effective_rank, lm_head_effective_rank, hessian_top_eig` (如未 skip)
- 每层 norm：`wnorm/embedding.weight, wnorm/blocks.0..., wnorm/norm_f.weight`

**cells summary 列**：
`group, alpha_idx, lambda_idx, alpha, lam, seed, phase, t_train, t_test, final_step`

---

## 8. 数据分析速查（pandas）

```python
import pandas as pd
import matplotlib.pyplot as plt

cells = pd.read_parquet("results/scans/B_p113_phase1.cells.parquet")
traj  = pd.read_parquet("results/scans/B_p113_phase1.parquet")

# 相图：每个 (alpha_idx, lambda_idx) 的 phase
phase_map = cells.pivot(index="lambda_idx", columns="alpha_idx", values="phase")
print(phase_map)

# 单 cell 的 fourier_sparsity 轨迹
cell = traj[(traj.alpha_idx == 4) & (traj.lambda_idx == 3)]
plt.semilogx(cell.step, cell.fourier_sparsity)
plt.xlabel("step"); plt.ylabel("Fourier sparsity (Gini)")

# (Track1, Track2) 解释矩阵：每 cell 取最终一步
last = traj.sort_values("step").drop_duplicates(["alpha_idx","lambda_idx","seed"], keep="last")
plt.scatter(last.fourier_sparsity, last.weight_norm_total,
            c=last.phase.astype("category").cat.codes)
```

---

## 9. 常见问题速查

| 症状 | 排查方向 |
|---|---|
| OOM | 当前实现是全批训练，p=113 → 12769 sample × 5 seq → ~64MB activations，4090 安全。如 OOM 检查是否同时跑了别的进程 |
| 训练 loss NaN | 检查 lr 是否过大（`--lr 1e-4`）；Adam betas 不变 |
| 全 cell phase=fail | T_min 不够，加大 `--T-min 500000` |
| Phase 1 跑到一半中断 | 使用同一个 `--out` 加 `--resume`；脚本会读取 `*.partial.cells.parquet` 并跳过已完成 cells |
| 长时间没有 cell completion | 看 `[cell ... progress]` heartbeat；通常是该 cell 自适应延长到 500k/1M/5M |
| Hessian 收敛慢 / 不稳定 | `--hessian-iters 30`（默认 10），或继续 `--skip-hessian` |
| Parquet 读不出 | 确保 `pip install pyarrow`，且 Python ≥ 3.10 |
| 结果与论文 Liu et al. 2022 相图不一致 | 我们用 ~93k 参数 mini-Qwen，他们用 ~10k 参数 MLP；定性应一致（4 区拓扑），定量边界不同 |

---

## 10. 反馈给开发者（你/我）

如果在 GPU 上发现：

1. **某个 cell 行为反直觉** → 把 `cells.parquet` 中该行 + `traj.parquet` 中该 cell 的轨迹一起贴回来
2. **新需求**（如新的 progress measure / 不同的 split / 不同的 grid 范围）→ 告诉我功能描述，我加到 codebase
3. **性能问题** → 报告 `nvidia-smi` 显存占用、单 cell wall-clock，我评估是否需要 vmap N>1 / `torch.compile` 优化

---

## 附：核心提交历史

```
08231b7  feat: add GPU scan observability and resume
94dd38a  chore: harden codebase reliability and lint
0a1d474  Phase 5b: instrumented scan with measures trajectories
51fb38b  Phase 5a: 8 progress measures (dual-track)
4c14fe0  Phase 4: vmap multi-seed scan infrastructure
c74ca43  Phase 3: training loop, S1/S3 splits, adaptive T
b1ecf5f  Phase 2: Mini-Qwen architecture (RMSNorm, RoPE, GQA, SwiGLU)
1075613  Phase 1: project skeleton, configs, smoke test green
```

测试覆盖：`make test` ✅

---

*更新日期：2026-05-06 · 对应 commit `08231b7`*
