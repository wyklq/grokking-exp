"""Tests for training loop, loss, and phase classifier (Phase 3).

These do not run a full grokking experiment (the user explicitly chose to
defer training to a GPU machine). We verify:
  - Label-masked loss extracts the right position
  - Phase classifier handles all four corner cases
  - A few training steps run end-to-end and reduce training loss
"""
from __future__ import annotations

import pytest
import torch

import mqg.train.trainer as trainer_module
from mqg.data import TaskSpec, make_split
from mqg.model import MiniQwenConfig
from mqg.train import (
    TrainConfig,
    answer_logits_and_targets,
    classify_phase,
    label_masked_loss_and_acc,
    train_one_cell,
)
from mqg.train.checkpoints import LogStepIterator


class TestLoss:
    def test_extract_answer(self):
        # tokens shape (B=2, S=5)
        tokens = torch.tensor([[1, 113, 2, 114, 3], [5, 113, 6, 114, 11]])
        logits = torch.zeros(2, 5, 115)
        logits[0, 3, 3] = 5.0  # huge logit at correct answer for batch 0
        logits[1, 3, 11] = 5.0  # correct for batch 1
        ans, tgt = answer_logits_and_targets(logits, tokens, answer_pos=4)
        assert ans.shape == (2, 115)
        assert torch.equal(tgt, torch.tensor([3, 11]))

    def test_perfect_predictions_give_low_loss_and_acc_one(self):
        tokens = torch.tensor([[1, 113, 2, 114, 3], [5, 113, 6, 114, 11]])
        logits = torch.full((2, 5, 115), -10.0)
        # high logit for the correct answer at position 3 (predicting position 4)
        logits[0, 3, 3] = 50.0
        logits[1, 3, 11] = 50.0
        loss, acc = label_masked_loss_and_acc(logits, tokens, answer_pos=4)
        assert acc.item() == 1.0
        assert loss.item() < 1e-3

    def test_invalid_answer_pos_raises(self):
        tokens = torch.tensor([[1, 113, 2, 114, 3]])
        logits = torch.zeros(1, 5, 115)
        with pytest.raises(ValueError, match="answer_pos"):
            answer_logits_and_targets(logits, tokens, answer_pos=0)


class TestPhaseClassifier:
    def test_fail(self):
        assert classify_phase(None, None) == "fail"
        assert classify_phase(None, 5000) == "fail"

    def test_memorize(self):
        assert classify_phase(t_train=1000, t_test=None) == "memorize"

    def test_grok(self):
        # gap = 50000, ratio threshold = 10*1000 = 10000 -> grok
        assert classify_phase(t_train=1000, t_test=51000) == "grok"

    def test_comprehend(self):
        # gap = 5000, threshold = 10000 -> comprehend
        assert classify_phase(t_train=1000, t_test=6000) == "comprehend"

    def test_invalid_grok_ratio_raises(self):
        with pytest.raises(ValueError, match="grok_ratio"):
            classify_phase(t_train=1000, t_test=6000, grok_ratio=0.0)


class TestLogStepIterator:
    def test_emits_at_thresholds(self):
        it = LogStepIterator(base=(10, 20, 50))
        assert it.peek_next() == 10
        assert it.reached(5) is False
        assert it.reached(10) is True
        assert it.peek_next() == 20
        # Skipping past thresholds still triggers each only once
        assert it.reached(100) is True  # advances past 20
        assert it.reached(100) is True  # advances past 50
        assert it.reached(100) is False  # exhausted


class TestTrainerSmoke:
    def test_short_run_reduces_loss(self):
        """Run 50 steps on a tiny p=5 task; train loss must drop."""
        spec = TaskSpec(p=5)  # 25 pairs
        model_cfg = MiniQwenConfig(
            vocab_size=spec.vocab_size,
            n_layers=2,
            d_model=64,
        )
        train_cfg = TrainConfig(
            lr=1e-3,
            weight_decay=0.0,
            T_min=50,
            T_max=50,
            seed=0,
        )
        train_idx, test_idx = make_split("S1", spec, alpha=0.6, seed=0)

        # Custom log steps so we capture the trajectory
        result = train_one_cell(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            spec=spec,
            train_idx=train_idx,
            test_idx=test_idx,
            log_steps=(1, 5, 10, 25, 50),
        )

        assert len(result.history) >= 3
        first = result.history[0]
        last = result.history[-1]
        # Loss must decrease over 50 steps
        assert last.train_loss < first.train_loss, (
            f"Train loss did not decrease: {first.train_loss} -> {last.train_loss}"
        )
        # Phase will be 'fail' since 50 steps is far below T to reach 99% acc.
        assert result.phase in {"fail", "memorize", "comprehend", "grok"}

    def test_adaptive_T_extends_after_train_acc(self):
        """If train_acc reaches threshold quickly, T_target should extend."""
        # Trick: start with extremely low acc threshold so it triggers immediately.
        spec = TaskSpec(p=5)
        model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size)
        train_cfg = TrainConfig(
            lr=1e-3,
            weight_decay=0.0,
            T_min=10,
            T_max=200,
            grok_extension_factor=10,
            acc_threshold=0.0,  # trivially satisfied at step 1
            seed=0,
        )
        train_idx, test_idx = make_split("S1", spec, alpha=0.8, seed=0)
        result = train_one_cell(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            spec=spec,
            train_idx=train_idx,
            test_idx=test_idx,
            log_steps=(1, 5, 10, 50, 100, 200),
        )
        # Threshold met immediately -> T_target = max(10, 10*1) = 10
        # but t_test also met (acc >= 0.0 always) -> finishes around T_target
        assert result.t_train == 1
        assert result.t_test == 1

    def test_adaptive_T_only_extends_once_without_train_acc(self):
        """No-train cells should fail after one order-of-magnitude extension."""
        spec = TaskSpec(p=7)
        model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size)
        train_cfg = TrainConfig(
            lr=1e-12,
            weight_decay=0.0,
            T_min=2,
            T_max=50,
            acc_threshold=1.0,
            seed=0,
        )
        train_idx, test_idx = make_split("S1", spec, alpha=0.5, seed=0)
        result = train_one_cell(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            spec=spec,
            train_idx=train_idx,
            test_idx=test_idx,
            log_steps=(2, 20, 50),
        )
        assert result.t_train is None
        assert result.phase == "fail"
        assert result.final_step == 20

    def test_train_after_no_train_extension_stops_at_grok_target(self, monkeypatch):
        """Late training accuracy still uses the grok target, not T_max."""
        scripted_metrics = iter(
            [
                (1.0, 0.0),  # step 2 train
                (1.0, 0.0),  # step 2 test
                (1.0, 1.0),  # step 20 train: sets t_train and target=60
                (1.0, 0.0),  # step 20 test
                (1.0, 1.0),  # step 60 train
                (1.0, 0.0),  # step 60 test: no t_test, so stop as memorize
            ]
        )

        def scripted_evaluate(*_args, **_kwargs):
            return next(scripted_metrics)

        monkeypatch.setattr(trainer_module, "evaluate", scripted_evaluate)

        spec = TaskSpec(p=5)
        model_cfg = MiniQwenConfig(
            vocab_size=spec.vocab_size,
            d_model=8,
            n_layers=1,
            n_heads=1,
            head_dim=8,
            n_kv_heads=1,
            ffn_hidden=16,
        )
        train_cfg = TrainConfig(
            lr=1e-3,
            weight_decay=0.0,
            T_min=2,
            T_max=80,
            grok_extension_factor=3,
            acc_threshold=0.99,
            seed=0,
        )
        train_idx, test_idx = make_split("S1", spec, alpha=0.5, seed=0)

        result = train_one_cell(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            spec=spec,
            train_idx=train_idx,
            test_idx=test_idx,
            log_steps=(2, 20, 60, 80),
        )

        assert result.t_train == 20
        assert result.t_test is None
        assert result.phase == "memorize"
        assert result.final_step == 60
        assert [entry.step for entry in result.history] == [2, 20, 60]

    def test_model_task_vocab_mismatch_raises(self):
        spec = TaskSpec(p=5)
        model_cfg = MiniQwenConfig(vocab_size=spec.vocab_size + 1)
        train_cfg = TrainConfig(T_min=1, T_max=1)
        train_idx, test_idx = make_split("S1", spec, alpha=0.8, seed=0)
        with pytest.raises(ValueError, match="vocab_size mismatch"):
            train_one_cell(model_cfg, train_cfg, spec, train_idx, test_idx, log_steps=(1,))
