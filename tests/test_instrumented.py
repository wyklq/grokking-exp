"""Tests for instrumented scan: unstack_seed + run_cell_with_measures."""
from __future__ import annotations

import torch

from mqg.data import TaskSpec, make_split
from mqg.model import MiniQwen, MiniQwenConfig
from mqg.scan import run_cell_with_measures, to_dataframe, unstack_seed
from mqg.scan.grid import GridCell
from mqg.scan.multi_seed import _build_stacked
from mqg.train.trainer import TrainConfig


class TestUnstack:
    def test_round_trip(self):
        cfg = MiniQwenConfig(vocab_size=10, n_layers=1, d_model=16,
                             n_heads=4, head_dim=4, n_kv_heads=1,
                             ffn_hidden=32, max_seq_len=8)
        torch.manual_seed(123)
        a = MiniQwen(cfg)
        torch.manual_seed(456)
        b = MiniQwen(cfg)
        params, buffers, _ = _build_stacked(cfg, [123, 456], device="cpu")

        a_rec = unstack_seed(params, buffers, cfg, 0)
        b_rec = unstack_seed(params, buffers, cfg, 1)

        # Compare every parameter to the original
        for (n1, p1), (n2, p2) in zip(a.named_parameters(), a_rec.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2, atol=1e-6), f"seed-0 param {n1} mismatch"
        for (n1, p1), (n2, p2) in zip(b.named_parameters(), b_rec.named_parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), f"seed-1 param {n1} mismatch"

    def test_unstacked_forward_matches(self):
        cfg = MiniQwenConfig(vocab_size=10, n_layers=2, d_model=64)
        params, buffers, base = _build_stacked(cfg, [7, 11], device="cpu")
        from torch.func import functional_call, vmap

        def fmodel(p, b, x):
            return functional_call(base, (p, b), (x,))

        x = torch.randint(0, 10, (3, 5))
        # vmap reference
        vlogits = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, x)
        # unstacked seed 0
        m0 = unstack_seed(params, buffers, cfg, 0)
        with torch.no_grad():
            ref0 = m0(x)
        assert torch.allclose(vlogits[0], ref0, atol=1e-5)
        m1 = unstack_seed(params, buffers, cfg, 1)
        with torch.no_grad():
            ref1 = m1(x)
        assert torch.allclose(vlogits[1], ref1, atol=1e-5)


class TestInstrumentedRun:
    def test_smoke(self):
        spec = TaskSpec(p=5)
        model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size, n_layers=2, d_model=64)
        train_cfg = TrainConfig(
            lr=1e-3, weight_decay=0.0, T_min=20, T_max=20, seed=0,
        )
        cell = GridCell(alpha_idx=0, lambda_idx=0, alpha=0.6, lam=0.0)
        results, rows = run_cell_with_measures(
            group="A", split_strategy="S1", spec=spec, cell=cell,
            seeds=[0, 1], base_train_cfg=train_cfg, base_model_cfg=model_cfg,
            log_steps=(5, 10, 20),
            measures_steps=(10, 20),
            skip_hessian=True,
        )
        assert len(results) == 2
        # rows: 2 seeds × 2 measures_steps = 4
        assert len(rows) == 4
        seeds_seen = {r["seed"] for r in rows}
        assert seeds_seen == {0, 1}
        steps_seen = sorted({r["step"] for r in rows})
        assert steps_seen == [10, 20]
        # check measure keys present
        sample = rows[0]
        for k in ("fourier_sparsity", "weight_norm_total", "embedding_stable_rank"):
            assert k in sample
        # ensure metadata present
        for k in ("group", "alpha", "lam", "alpha_idx", "lambda_idx", "step", "seed"):
            assert k in sample

    def test_dataframe(self):
        spec = TaskSpec(p=5)
        model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size)
        train_cfg = TrainConfig(lr=1e-3, weight_decay=0.0, T_min=20, T_max=20, seed=0)
        cell = GridCell(alpha_idx=0, lambda_idx=0, alpha=0.6, lam=0.0)
        _, rows = run_cell_with_measures(
            group="A", split_strategy="S1", spec=spec, cell=cell,
            seeds=[0], base_train_cfg=train_cfg, base_model_cfg=model_cfg,
            log_steps=(5, 10, 20), measures_steps=(20,), skip_hessian=True,
        )
        df = to_dataframe(rows)
        assert len(df) == 1
        assert "fourier_sparsity" in df.columns
        assert df.iloc[0]["seed"] == 0
        assert df.iloc[0]["step"] == 20
