# Mini-Qwen Grokking

Research codebase for "Phase Diagram of Emergent Modular Arithmetic in a Tiny Qwen-Style Transformer".

See `docs/` (links to Obsidian vault) for the full research design notes:

- Phase Diagram 研究设计
- Tokenization 与 Split 哲学
- Phase Diagram 扫描方案
- Progress Measures 双轨

## Quickstart

```bash
make setup     # install dependencies
make smoke     # run smoke test (model forward pass)
make test      # run full test suite
```

## Layout

```
src/mqg/
  model/      # Mini-Qwen architecture (RMSNorm, RoPE, GQA, SwiGLU)
  data/       # Modular arithmetic dataset + S1/S3 splits
  train/      # Training loop with vmap multi-seed
  measures/   # Track 1 (Fourier) + Track 2 (architecture-agnostic) measures
  scan/       # Phase diagram scan orchestration
configs/      # Hydra configs
tests/        # Unit + smoke tests
results/      # Scan outputs (gitignored)
```

## Roadmap

See `task_plan.md` (in Obsidian vault under `AI-Chats/plans/mini-qwen-grokking/`).

Current phase: **Phase 1 — Project skeleton**
