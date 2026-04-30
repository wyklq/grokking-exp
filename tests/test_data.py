"""Tests for dataset and split logic (Phase 3)."""
from __future__ import annotations

import pytest
import torch

from mqg.data import TaskSpec, build_full_dataset, make_split, split_S1, split_S3


class TestDataset:
    def test_full_dataset_shape(self):
        spec = TaskSpec(p=7)
        tokens, c = build_full_dataset(spec)
        assert tokens.shape == (49, 5)
        assert c.shape == (49,)
        assert tokens.dtype == torch.long

    def test_answers_correct(self):
        spec = TaskSpec(p=11)
        tokens, c = build_full_dataset(spec)
        a = tokens[:, 0]
        b = tokens[:, 2]
        # operator and equals tokens
        assert torch.all(tokens[:, 1] == spec.plus_id)
        assert torch.all(tokens[:, 3] == spec.eq_id)
        # answers
        assert torch.equal(c, (a + b) % spec.p)
        assert torch.equal(tokens[:, 4], c)

    def test_mul_op(self):
        spec = TaskSpec(p=7, op="mul")
        tokens, c = build_full_dataset(spec)
        a, b = tokens[:, 0], tokens[:, 2]
        assert torch.equal(c, (a * b) % spec.p)

    def test_invalid_spec_raises(self):
        with pytest.raises(ValueError, match="p must be >= 2"):
            TaskSpec(p=1)
        with pytest.raises(ValueError, match="Unknown op"):
            TaskSpec(p=7, op="sub")


class TestSplitS1:
    def test_alpha_size(self):
        train_idx, test_idx = split_S1(100, alpha=0.3, seed=0)
        assert len(train_idx) == 30
        assert len(test_idx) == 70
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0

    def test_determinism(self):
        a = split_S1(100, 0.5, seed=42)
        b = split_S1(100, 0.5, seed=42)
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_disjoint_full_cover(self):
        train_idx, test_idx = split_S1(50, 0.4, seed=1)
        union = torch.cat([train_idx, test_idx]).sort().values
        assert torch.equal(union, torch.arange(50))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            split_S1(100, alpha=0.0, seed=0)


class TestSplitS3:
    def test_b_columns_partition(self):
        spec = TaskSpec(p=11)
        train_idx, test_idx = split_S3(spec, alpha=0.3, seed=0)
        # Reconstruct (a, b) for each pair
        b_all = torch.arange(11).repeat(11)

        train_b = set(b_all[train_idx].tolist())
        test_b = set(b_all[test_idx].tolist())

        # b-values are partitioned: train b's never appear in test (and vice versa)
        assert len(train_b & test_b) == 0
        assert train_b | test_b == set(range(11))

        # Round(0.3 * 11) = 3 b-values seen, so train has 3 * 11 = 33 pairs
        assert len(train_idx) == 3 * 11
        assert len(test_idx) == (11 - 3) * 11

    def test_each_a_appears_for_each_train_b(self):
        spec = TaskSpec(p=7)
        train_idx, _ = split_S3(spec, alpha=0.5, seed=0)
        a_all = torch.arange(7).repeat_interleave(7)
        b_all = torch.arange(7).repeat(7)

        train_pairs = list(zip(a_all[train_idx].tolist(), b_all[train_idx].tolist()))
        train_b = set(b for _, b in train_pairs)

        # for each chosen b, all a in 0..6 appear
        for b in train_b:
            assert {a for a, bp in train_pairs if bp == b} == set(range(7))

    def test_at_least_one_b(self):
        spec = TaskSpec(p=11)
        # Even with very small alpha, at least one b column is selected
        train_idx, _ = split_S3(spec, alpha=0.01, seed=0)
        assert len(train_idx) >= 11  # >= p (one b column)

    def test_invalid_alpha_raises(self):
        spec = TaskSpec(p=11)
        with pytest.raises(ValueError, match="alpha"):
            split_S3(spec, alpha=1.0, seed=0)


class TestMakeSplit:
    def test_dispatch(self):
        spec = TaskSpec(p=7)
        s1 = make_split("S1", spec, 0.5, seed=0)
        s3 = make_split("S3", spec, 0.5, seed=0)
        # Both return tensor pair
        assert len(s1) == 2 and len(s3) == 2

    def test_unknown_strategy_raises(self):
        spec = TaskSpec(p=7)
        with pytest.raises(ValueError, match="Unknown split strategy"):
            make_split("S99", spec, 0.5, seed=0)
