"""Tests for Phase 4: vmap multi-seed, grid, boundary detection, scan runner."""
from __future__ import annotations

import torch

from mqg.data import TaskSpec, make_split
from mqg.model import MiniQwenConfig
from mqg.scan import (
    GridSpec,
    default_alpha_grid,
    default_grid_spec,
    default_lambda_grid,
    detect_boundary_cells,
    majority_phase,
    train_multi_seed,
)
from mqg.scan.scan_runner import run_cell, run_phase1, run_phase2
from mqg.train.trainer import TrainConfig, train_one_cell


# ---------------- Grid ----------------

class TestGrid:
    def test_default_alpha_size(self):
        a = default_alpha_grid(9)
        assert len(a) == 9
        assert min(a) == 0.1 and max(a) == 0.9

    def test_default_lambda_logspace(self):
        l = default_lambda_grid(7, 1e-2, 1e1)
        assert len(l) == 7
        assert abs(l[0] - 1e-2) < 1e-6
        assert abs(l[-1] - 1e1) < 1e-3
        # Check log spacing: ratio between consecutive values should be roughly constant
        ratios = [l[i+1] / l[i] for i in range(len(l)-1)]
        for r in ratios:
            assert abs(r - ratios[0]) / ratios[0] < 1e-3

    def test_grid_cells_count(self):
        g = GridSpec(alpha_values=(0.1, 0.5, 0.9), lambda_values=(0.01, 1.0))
        cells = g.cells()
        assert len(cells) == 6
        assert g.shape == (3, 2)
        idx_set = {(c.alpha_idx, c.lambda_idx) for c in cells}
        assert len(idx_set) == 6

    def test_default_grid(self):
        g = default_grid_spec()
        assert g.shape == (9, 7)
        assert len(g.cells()) == 63


# ---------------- Boundary ----------------

class TestBoundary:
    def test_uniform_no_boundary(self):
        ph = {(i, j): "grok" for i in range(3) for j in range(3)}
        assert detect_boundary_cells(ph) == []

    def test_single_outlier_returns_5_cells(self):
        """Center 'fail' surrounded by 'grok' -> center + 4 neighbors = 5 cells."""
        ph = {(i, j): "grok" for i in range(3) for j in range(3)}
        ph[(1, 1)] = "fail"
        b = set(detect_boundary_cells(ph))
        # disagreeing edges: center<->each of 4 neighbors
        # both sides included
        assert b == {(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)}

    def test_phase_split_in_half(self):
        """Top row 'grok', bottom row 'memorize' -> middle two rows are boundary."""
        ph = {}
        for i in range(4):
            for j in range(3):
                ph[(i, j)] = "grok" if i < 2 else "memorize"
        b = set(detect_boundary_cells(ph))
        # rows 1 and 2 are at the interface
        expected = {(1, j) for j in range(3)} | {(2, j) for j in range(3)}
        assert b == expected

    def test_majority_phase(self):
        assert majority_phase(["grok", "grok", "memorize"]) == "grok"
        assert majority_phase([]) == "fail"
        # tie -> alphabetic
        assert majority_phase(["grok", "memorize"]) == "grok"
        assert majority_phase(["memorize", "fail"]) == "fail"


# ---------------- Multi-seed trainer ----------------

class TestMultiSeed:
    def _make(self, p=5):
        spec = TaskSpec(p=p)
        model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size, n_layers=2, d_model=64)
        train_cfg = TrainConfig(
            lr=1e-3, weight_decay=0.0, T_min=30, T_max=30, seed=0,
        )
        train_idx, test_idx = make_split("S1", spec, alpha=0.6, seed=0)
        return spec, model_cfg, train_cfg, train_idx, test_idx

    def test_returns_one_per_seed(self):
        spec, mcfg, tcfg, tr, te = self._make()
        seeds = [0, 1, 2]
        results = train_multi_seed(
            mcfg, tcfg, spec, tr, te, seeds=seeds,
            log_steps=(5, 10, 20, 30),
        )
        assert len(results) == 3
        for r in results:
            assert r.final_step > 0
            assert len(r.history) >= 1

    def test_different_seeds_diverge(self):
        """Different init seeds should give different training trajectories."""
        spec, mcfg, tcfg, tr, te = self._make()
        results = train_multi_seed(
            mcfg, tcfg, spec, tr, te, seeds=[0, 1],
            log_steps=(5, 15, 30),
        )
        l0 = results[0].history[-1].train_loss
        l1 = results[1].history[-1].train_loss
        assert abs(l0 - l1) > 1e-6, "Identical losses suggests seeds didn't differentiate"

    def test_matches_single_seed(self):
        """Seed-0 in a multi-seed run should match seed-0 in train_one_cell.

        This is the critical equivalence test: vmap + stacked AdamW must
        produce numerically identical trajectories to the sequential trainer.
        """
        spec, mcfg, tcfg, tr, te = self._make()
        log_steps = (5, 15, 30)
        # Run sequentially
        single = train_one_cell(mcfg, tcfg, spec, tr, te, log_steps=log_steps)
        # Run as part of multi-seed batch (with another seed alongside)
        multi = train_multi_seed(
            mcfg, tcfg, spec, tr, te, seeds=[0, 1], log_steps=log_steps,
        )[0]
        # Histories must match closely
        assert len(single.history) == len(multi.history)
        for s, m in zip(single.history, multi.history):
            assert s.step == m.step
            # loss equivalence within float tolerance
            assert abs(s.train_loss - m.train_loss) < 1e-4, \
                f"step {s.step}: single={s.train_loss} multi={m.train_loss}"
            assert abs(s.test_loss - m.test_loss) < 1e-4
            assert abs(s.train_acc - m.train_acc) < 1e-3
            assert abs(s.test_acc - m.test_acc) < 1e-3
        assert single.t_train == multi.t_train
        assert single.t_test == multi.t_test
        assert single.phase == multi.phase

    def test_matches_single_seed_with_weight_decay(self):
        """Equivalence must hold under nonzero weight decay too (AdamW path)."""
        spec, mcfg, tcfg, tr, te = self._make()
        tcfg = TrainConfig(lr=1e-3, weight_decay=0.5, T_min=20, T_max=20, seed=0)
        log_steps = (5, 10, 20)
        single = train_one_cell(mcfg, tcfg, spec, tr, te, log_steps=log_steps)
        multi = train_multi_seed(
            mcfg, tcfg, spec, tr, te, seeds=[0, 1], log_steps=log_steps,
        )[0]
        for s, m in zip(single.history, multi.history):
            assert abs(s.train_loss - m.train_loss) < 1e-4, \
                f"WD path differs at step {s.step}: {s.train_loss} vs {m.train_loss}"


# ---------------- scan_runner ----------------

class TestScanRunner:
    def _setup(self, p=5):
        spec = TaskSpec(p=p)
        model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size, n_layers=2, d_model=64)
        train_cfg = TrainConfig(
            lr=1e-3, weight_decay=0.0, T_min=20, T_max=20, seed=0,
        )
        return spec, model_cfg, train_cfg

    def test_run_cell_records(self):
        spec, mcfg, tcfg = self._setup()
        from mqg.scan.grid import GridCell
        cell = GridCell(alpha_idx=0, lambda_idx=0, alpha=0.6, lam=0.0)
        recs = run_cell(
            group="A", split_strategy="S1", spec=spec, cell=cell,
            seeds=[0, 1, 2], base_train_cfg=tcfg, base_model_cfg=mcfg,
            log_steps=(5, 10, 20),
        )
        assert len(recs) == 3
        assert {r.seed for r in recs} == {0, 1, 2}
        for r in recs:
            assert r.group == "A"
            assert r.alpha == 0.6
            assert r.lam == 0.0
            assert r.tied_embedding is True

    def test_run_phase1_then_phase2(self):
        """Smoke: full 2x2 grid, phase1 then phase2."""
        spec, mcfg, tcfg = self._setup()
        grid = GridSpec(alpha_values=(0.4, 0.7), lambda_values=(0.0, 0.5))
        p1 = run_phase1(
            group="A", split_strategy="S1", spec=spec, grid=grid,
            base_train_cfg=tcfg, base_model_cfg=mcfg,
            log_steps=(5, 10, 20),
        )
        # one record per cell at seed=0
        assert len(p1) == 4
        assert all(r.seed == 0 for r in p1)
        # phase2: even if no boundaries detected, should not crash
        p2 = run_phase2(
            group="A", split_strategy="S1", spec=spec, grid=grid,
            phase1_records=p1, n_seeds=3,
            base_train_cfg=tcfg, base_model_cfg=mcfg,
            log_steps=(5, 10, 20),
        )
        # p2 records (if boundary exists) have seed in {1, 2}
        for r in p2:
            assert r.seed in {1, 2}
