# Mini-Qwen Grokking

Research codebase for "Phase Diagram of Emergent Modular Arithmetic in a Tiny Qwen-Style Transformer".

See `docs/` for the full guides:

- **[`docs/GPU_GUIDE.md`](docs/GPU_GUIDE.md)** — GPU 上跑实验的完整手册（验证步骤 / 预计耗时 / 输出格式 / 排错）
- Obsidian vault `AI-Chats/concepts/` — research design notes:
  - Phase Diagram 研究设计
  - Tokenization 与 Split 哲学
  - Phase Diagram 扫描方案
  - Progress Measures 双轨

## Quickstart

```bash
make setup     # install dependencies
make smoke     # run smoke test (model forward pass)
make test      # run full test suite (87/87 expected)
```

**Single-cell run**:
```bash
python3 scripts/train_one.py --p 113 --alpha 0.3 --lambda 1.0 --T-min 100000 --device cuda
```

**Full instrumented scan** (one experimental group):
```bash
python3 scripts/run_scan_instrumented.py --group B --p 113 --device cuda \
    --measures-steps 100 1000 10000 100000 1000000 --skip-hessian
```

See `docs/GPU_GUIDE.md` for full command reference.

## Layout

```
src/mqg/
  model/          # Mini-Qwen architecture (RMSNorm, RoPE, GQA, SwiGLU) — 93,440 params
  data/           # Modular arithmetic dataset + S1/S3 splits
  train/          # Training loop with adaptive T protocol
  measures/       # Track 1 (Fourier 4) + Track 2 (architecture-agnostic 4) measures
  scan/           # vmap multi-seed + grid + boundary detection + instrumented runs
configs/          # Hydra configs (decisions D1-D8 locked here)
scripts/          # CLIs: train_one, run_scan, run_scan_instrumented
tests/            # 87 unit + smoke + equivalence tests
docs/             # GPU_GUIDE.md
results/          # Scan outputs (gitignored)
```

## Roadmap

| Phase | Status |
|---|---|
| 1. Project skeleton | ✅ |
| 2. Mini-Qwen architecture | ✅ |
| 3. Training loop + S1/S3 + adaptive T | ✅ |
| 4. vmap multi-seed scan infrastructure | ✅ |
| 5a. 8 progress measures | ✅ |
| 5b. Instrumented scan + Parquet output | ✅ |
| 6. GPU experiments + figures + paper | ⏳ — see GPU_GUIDE.md |

See `task_plan.md` (in Obsidian vault under `AI-Chats/plans/mini-qwen-grokking/`) for detailed phase breakdown.
