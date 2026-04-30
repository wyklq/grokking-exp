"""Tests for Phase 5 progress measures.

Strategy:
- Verify each measure on a synthetic input where the answer is known.
- Avoid running real grokking; the user will validate on GPU.
"""
from __future__ import annotations

import math

import pytest
import torch

from mqg.measures import (
    circularity,
    collect_activations,
    compute_all_measures,
    dominant_frequencies,
    effective_rank,
    fourier_sparsity,
    hvp,
    linear_cka,
    per_layer_weight_norms,
    restricted_embedding,
    stable_rank,
    top_hessian_eigenvalue,
    weight_norm,
)
from mqg.measures.fourier import _gini


# ---------------- Fourier ----------------

class TestFourier:
    def test_gini_uniform(self):
        x = torch.ones(10)
        assert _gini(x).item() < 1e-6

    def test_gini_concentrated(self):
        x = torch.zeros(10)
        x[0] = 1.0
        # max Gini for n elements with single nonzero is (n-1)/n = 0.9
        assert abs(_gini(x).item() - 0.9) < 1e-6

    def test_gini_rejects_negative_values(self):
        with pytest.raises(ValueError, match="non-negative"):
            _gini(torch.tensor([1.0, -1.0]))

    def test_sparsity_random_low(self):
        torch.manual_seed(0)
        E = torch.randn(31, 16)
        # random Gaussian → spectrum roughly uniform → low Gini
        assert fourier_sparsity(E) < 0.5

    def test_sparsity_pure_freq_high(self):
        """Build E whose every dim is a pure cosine at freq=3 → sparsity ≈ max."""
        p, d = 31, 16
        n = torch.arange(p, dtype=torch.float32)
        cos3 = torch.cos(2 * math.pi * 3 * n / p)
        E = cos3.unsqueeze(1).expand(p, d).clone()
        s = fourier_sparsity(E)
        assert s > 0.9, f"pure-freq Gini should be near 1, got {s}"

    def test_dominant_freq_recovers_truth(self):
        p, d = 31, 16
        n = torch.arange(p, dtype=torch.float32)
        E = torch.cos(2 * math.pi * 7 * n / p).unsqueeze(1).expand(p, d).clone()
        doms = dominant_frequencies(E, k=2)
        # cos at freq k contributes to FFT bins k AND p-k (conjugate)
        assert set(doms[:2]) == {7, p - 7}

    def test_restricted_embedding_keeps_only_freq(self):
        p, d = 31, 8
        torch.manual_seed(0)
        E = torch.randn(p, d)
        rec = restricted_embedding(E, freqs=[3])
        # rec has support only at frequencies ±3
        F = torch.fft.fft(rec, dim=0)
        power = (F.real ** 2 + F.imag ** 2).sum(dim=1)
        # Only bins 3 and p-3 should have nonzero power
        nonzero = (power > 1e-6).nonzero().flatten().tolist()
        assert set(nonzero) == {3, p - 3}

    def test_restricted_full_recovers_E(self):
        p, d = 17, 8
        torch.manual_seed(0)
        E = torch.randn(p, d)
        rec = restricted_embedding(E, freqs=list(range(p)))
        # Keeping all freqs should reconstruct E exactly
        assert torch.allclose(rec, E, atol=1e-5)

    def test_circularity_perfect(self):
        """If E rows lie exactly on a circle for freq=k, CV should be ~0."""
        p = 31
        k = 5
        n = torch.arange(p, dtype=torch.float32)
        # Construct E so each token n maps to (cos, sin) on unit circle in 2D
        # repeated across all d dims
        cos = torch.cos(2 * math.pi * k * n / p)
        sin = torch.sin(2 * math.pi * k * n / p)
        # E shape (p, d) where d=4 = (cos, sin, cos, sin)
        E = torch.stack([cos, sin, cos, sin], dim=1)
        cv = circularity(E, freq=k)
        assert cv < 0.01, f"perfectly circular should have CV~0, got {cv}"

    def test_circularity_is_scale_invariant(self):
        p = 31
        k = 5
        n = torch.arange(p, dtype=torch.float32)
        cos = torch.cos(2 * math.pi * k * n / p)
        sin = torch.sin(2 * math.pi * k * n / p)
        E = torch.stack([cos, sin, cos, sin], dim=1)
        assert abs(circularity(E, freq=k) - circularity(1e-8 * E, freq=k)) < 1e-4


# ---------------- Weights ----------------

class TestWeights:
    def test_weight_norm_matches_manual(self):
        m = torch.nn.Linear(10, 5, bias=True)
        manual = (m.weight ** 2).sum() + (m.bias ** 2).sum()
        manual = manual.sqrt().item()
        assert abs(weight_norm(m) - manual) < 1e-5

    def test_per_layer_weight_norms_keys(self):
        m = torch.nn.Linear(3, 2)
        d = per_layer_weight_norms(m)
        assert set(d.keys()) == {"weight", "bias"}

    def test_stable_rank_identity(self):
        identity = torch.eye(5)
        # All singular values 1 → stable rank = 5/1 = 5
        assert abs(stable_rank(identity) - 5.0) < 1e-5

    def test_stable_rank_scaled_identity(self):
        identity = 1e-8 * torch.eye(5)
        assert abs(stable_rank(identity) - 5.0) < 1e-5

    def test_stable_rank_rank1(self):
        u = torch.randn(10, 1)
        v = torch.randn(1, 8)
        W = u @ v
        assert abs(stable_rank(W) - 1.0) < 1e-4

    def test_effective_rank_identity(self):
        identity = torch.eye(7)
        # Equal singular values → entropy = log(7), exp() = 7
        assert abs(effective_rank(identity) - 7.0) < 1e-4

    def test_effective_rank_scaled_identity(self):
        identity = 1e-8 * torch.eye(7)
        assert abs(effective_rank(identity) - 7.0) < 1e-4

    def test_effective_rank_rank1(self):
        u = torch.randn(10, 1)
        v = torch.randn(1, 8)
        W = u @ v
        assert abs(effective_rank(W) - 1.0) < 1e-4


# ---------------- CKA ----------------

class TestCKA:
    def test_cka_self_is_one(self):
        torch.manual_seed(0)
        X = torch.randn(50, 16)
        assert abs(linear_cka(X, X) - 1.0) < 1e-5

    def test_cka_invariant_to_orthogonal(self):
        torch.manual_seed(0)
        X = torch.randn(50, 16)
        Q, _ = torch.linalg.qr(torch.randn(16, 16))
        Y = X @ Q
        assert abs(linear_cka(X, Y) - 1.0) < 1e-4

    def test_cka_invariant_to_scale(self):
        torch.manual_seed(0)
        X = torch.randn(40, 8)
        assert abs(linear_cka(X, 7.5 * X) - 1.0) < 1e-5

    def test_cka_unrelated_low(self):
        torch.manual_seed(0)
        X = torch.randn(200, 4)
        Y = torch.randn(200, 4)
        # Random uncorrelated Gaussians → CKA should be small
        assert linear_cka(X, Y) < 0.3

    def test_cka_rejects_mismatched_n(self):
        with pytest.raises(ValueError, match="same n"):
            linear_cka(torch.randn(3, 2), torch.randn(4, 2))


# ---------------- Hessian ----------------

class TestHessian:
    def test_quadratic_top_eigenvalue(self):
        """L(theta) = 0.5 * theta^T A theta with A diag(1, 5, 2). Top eig = 5."""
        torch.manual_seed(0)
        theta = torch.zeros(3, requires_grad=True)
        theta.data = torch.randn(3)
        A = torch.diag(torch.tensor([1.0, 5.0, 2.0]))

        def loss_fn():
            return 0.5 * theta @ A @ theta

        eig = top_hessian_eigenvalue(loss_fn, [theta], n_iter=30, seed=0)
        assert abs(eig - 5.0) < 1e-3, f"expected 5.0, got {eig}"

    def test_negative_dominant_eigenvalue_preserves_sign(self):
        """Power iteration converges by magnitude but returns signed Rayleigh quotient."""
        theta = torch.randn(3, requires_grad=True)
        A = torch.diag(torch.tensor([-5.0, 1.0, 2.0]))

        def loss_fn():
            return 0.5 * theta @ A @ theta

        eig = top_hessian_eigenvalue(loss_fn, [theta], n_iter=50, seed=0)
        assert abs(eig + 5.0) < 1e-3, f"expected -5.0, got {eig}"

    def test_hvp_correctness(self):
        """For L = 0.5 ||theta||^2, H = I, so Hv == v."""
        theta = torch.randn(5, requires_grad=True)

        def loss_fn():
            return 0.5 * (theta ** 2).sum()

        v = torch.randn(5)
        out = hvp(loss_fn, [theta], v)
        assert torch.allclose(out, v, atol=1e-5)


# ---------------- compute_all_measures end-to-end ----------------

class TestComputeAll:
    def test_runs_on_mini_qwen(self):
        from mqg.data import TaskSpec, build_full_dataset
        from mqg.model import MiniQwen, MiniQwenConfig

        spec = TaskSpec(p=7)
        cfg = MiniQwenConfig(vocab_size=spec.vocab_size, n_layers=2, d_model=64)
        model = MiniQwen(cfg)
        tokens, _ = build_full_dataset(spec)
        # use a small batch to keep Hessian fast
        train_tokens = tokens[:16]
        out = compute_all_measures(
            model, train_tokens, answer_pos=spec.answer_pos, p=spec.p,
            hessian_iters=3,
        )
        # required keys present
        for k in [
            "fourier_sparsity",
            "circularity_top_freq",
            "weight_norm_total",
            "embedding_stable_rank",
            "embedding_effective_rank",
            "hessian_top_eig",
        ]:
            assert k in out, f"missing key: {k}"
        # sanity ranges
        assert 0.0 <= out["fourier_sparsity"] <= 1.0
        assert out["weight_norm_total"] > 0.0
        assert out["embedding_stable_rank"] >= 1.0
        assert abs(out["hessian_top_eig"]) > 0.0  # dominant eig (signed) magnitude > 0

    def test_skip_hessian(self):
        from mqg.data import TaskSpec, build_full_dataset
        from mqg.model import MiniQwen, MiniQwenConfig

        spec = TaskSpec(p=7)
        cfg = MiniQwenConfig(vocab_size=spec.vocab_size)
        model = MiniQwen(cfg)
        tokens, _ = build_full_dataset(spec)
        out = compute_all_measures(
            model, tokens[:16], answer_pos=spec.answer_pos, p=spec.p,
            skip_hessian=True,
        )
        assert "hessian_top_eig" not in out

    def test_collect_activations_shape(self):
        from mqg.data import TaskSpec, build_full_dataset
        from mqg.model import MiniQwen, MiniQwenConfig

        spec = TaskSpec(p=5)
        cfg = MiniQwenConfig(vocab_size=spec.vocab_size)
        model = MiniQwen(cfg)
        tokens, _ = build_full_dataset(spec)
        x = tokens[:8]
        # collect output of first block
        acts = collect_activations(model, model.blocks[0], x)
        assert acts.shape[0] == 8
        # CKA self check on real activations
        assert abs(linear_cka(acts, acts) - 1.0) < 1e-4
