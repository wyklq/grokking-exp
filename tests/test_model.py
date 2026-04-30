"""Unit tests for Mini-Qwen architecture (Phase 2)."""
from __future__ import annotations

import math

import pytest
import torch

from mqg.model import (
    GQAAttention,
    MiniQwen,
    MiniQwenBlock,
    MiniQwenConfig,
    RMSNorm,
    RoPECache,
    SwiGLU,
    apply_rope,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def test_shape_preserved(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 5, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_rms_after_norm_with_unit_gain(self):
        norm = RMSNorm(64, eps=0.0)
        x = torch.randn(2, 5, 64) * 7.3
        out = norm(x)
        rms = out.float().pow(2).mean(dim=-1).sqrt()
        # Should be ~1 since gain init = 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_param_count(self):
        norm = RMSNorm(64)
        assert sum(p.numel() for p in norm.parameters()) == 64


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
class TestRoPE:
    def test_cache_shape(self):
        rope = RoPECache(head_dim=16, max_seq_len=8)
        cos, sin = rope.get(5)
        assert cos.shape == (5, 8)
        assert sin.shape == (5, 8)

    def test_rope_preserves_norm(self):
        rope = RoPECache(head_dim=16, max_seq_len=8)
        cos, sin = rope.get(5)
        x = torch.randn(2, 5, 4, 16)  # (B, S, H, D)
        y = apply_rope(x, cos, sin)
        # rotation preserves L2 norm per token
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

    def test_rope_distinguishes_positions(self):
        rope = RoPECache(head_dim=16, max_seq_len=8)
        cos, sin = rope.get(5)
        # Same vector at different positions should rotate differently
        x = torch.randn(1, 1, 1, 16).expand(1, 5, 1, 16).contiguous()
        y = apply_rope(x, cos, sin)
        # Position 0 unchanged (cos=1, sin=0)
        assert torch.allclose(y[:, 0], x[:, 0], atol=1e-5)
        # Other positions should differ
        for p in range(1, 5):
            assert not torch.allclose(y[:, p], x[:, p], atol=1e-3)

    def test_seq_len_overflow_raises(self):
        rope = RoPECache(head_dim=16, max_seq_len=4)
        with pytest.raises(ValueError):
            rope.get(10)


# ---------------------------------------------------------------------------
# GQA
# ---------------------------------------------------------------------------
class TestGQA:
    def test_shape(self):
        rope = RoPECache(16, 8)
        attn = GQAAttention(d_model=64, n_heads=4, head_dim=16, n_kv_heads=1, rope=rope)
        x = torch.randn(2, 5, 64)
        out = attn(x)
        assert out.shape == (2, 5, 64)

    def test_param_count(self):
        rope = RoPECache(16, 8)
        attn = GQAAttention(d_model=64, n_heads=4, head_dim=16, n_kv_heads=1, rope=rope)
        n = sum(p.numel() for p in attn.parameters())
        # q: 64*64=4096, k: 64*16=1024, v: 64*16=1024, o: 64*64=4096
        assert n == 4096 + 1024 + 1024 + 4096 == 10240

    def test_causal_mask_applied(self):
        """Output at position t must not depend on inputs at positions > t."""
        rope = RoPECache(16, 8)
        attn = GQAAttention(d_model=64, n_heads=4, head_dim=16, n_kv_heads=1, rope=rope)
        attn.eval()

        torch.manual_seed(0)
        x1 = torch.randn(1, 5, 64)
        x2 = x1.clone()
        x2[:, 3:] = torch.randn(1, 2, 64)  # change positions 3 and 4

        with torch.no_grad():
            y1 = attn(x1)
            y2 = attn(x2)

        # Positions 0..2 should be unaffected by changes at 3+ (causal)
        assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-5)
        # Position 3 onward should differ
        assert not torch.allclose(y1[:, 3], y2[:, 3], atol=1e-4)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------
class TestSwiGLU:
    def test_shape(self):
        ffn = SwiGLU(64, 170)
        x = torch.randn(2, 5, 64)
        assert ffn(x).shape == (2, 5, 64)

    def test_param_count(self):
        ffn = SwiGLU(64, 170)
        n = sum(p.numel() for p in ffn.parameters())
        # gate: 64*170, up: 64*170, down: 170*64
        assert n == 3 * 64 * 170 == 32640


# ---------------------------------------------------------------------------
# MiniQwenBlock
# ---------------------------------------------------------------------------
class TestBlock:
    def test_shape(self):
        cfg = MiniQwenConfig()
        rope = RoPECache(cfg.head_dim, cfg.max_seq_len)
        block = MiniQwenBlock(cfg, rope)
        x = torch.randn(2, 5, cfg.d_model)
        assert block(x).shape == (2, 5, cfg.d_model)

    def test_residual_path(self):
        """With zeroed sublayer outputs, block should be identity."""
        cfg = MiniQwenConfig()
        rope = RoPECache(cfg.head_dim, cfg.max_seq_len)
        block = MiniQwenBlock(cfg, rope)
        # Zero the output projections
        torch.nn.init.zeros_(block.attn.o_proj.weight)
        torch.nn.init.zeros_(block.ffn.down.weight)
        x = torch.randn(2, 5, cfg.d_model)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# MiniQwen full model
# ---------------------------------------------------------------------------
class TestMiniQwen:
    def test_forward_shape(self):
        cfg = MiniQwenConfig()
        model = MiniQwen(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = model(tokens)
        assert logits.shape == (2, 5, cfg.vocab_size)

    def test_param_count_tied(self):
        """Verify locked decision D4: ~93,440 params with tied embedding."""
        cfg = MiniQwenConfig(tied_embedding=True)
        model = MiniQwen(cfg)
        n = model.num_params()
        # Per layer: 10240 (attn) + 32640 (ffn) + 128 (2x RMSNorm) = 43008
        # 2 layers: 86016
        # Embedding (tied = no extra head): 115*64 = 7360
        # Final RMSNorm: 64
        expected = 2 * (10240 + 32640 + 128) + 115 * 64 + 64
        assert n == expected == 93440, f"Expected {expected}, got {n}"

    def test_param_count_untied(self):
        """Untied = tied + extra LM head 64*115 = 7360."""
        cfg = MiniQwenConfig(tied_embedding=False)
        model = MiniQwen(cfg)
        n = model.num_params()
        expected = 93440 + 64 * 115
        assert n == expected == 100800, f"Expected {expected}, got {n}"

    def test_tied_vs_untied_logits_differ_at_init(self):
        """Sanity: tied and untied have structurally different output paths."""
        torch.manual_seed(42)
        cfg_tied = MiniQwenConfig(tied_embedding=True)
        torch.manual_seed(42)
        cfg_untied = MiniQwenConfig(tied_embedding=False)

        torch.manual_seed(42)
        m_tied = MiniQwen(cfg_tied)
        torch.manual_seed(42)
        m_untied = MiniQwen(cfg_untied)

        tokens = torch.randint(0, 115, (1, 5))
        with torch.no_grad():
            l_tied = m_tied(tokens)
            l_untied = m_untied(tokens)
        # Different logits because untied LM head is independently initialized
        assert not torch.allclose(l_tied, l_untied)

    def test_backward_tied(self):
        """Gradients flow through both embedding routes when tied."""
        cfg = MiniQwenConfig(tied_embedding=True)
        model = MiniQwen(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()

        emb_grad = model.embedding.weight.grad
        assert emb_grad is not None
        assert emb_grad.abs().sum().item() > 0

    def test_backward_untied(self):
        """Gradients flow through embedding (input path) and lm_head (output)."""
        cfg = MiniQwenConfig(tied_embedding=False)
        model = MiniQwen(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 5))
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()

        assert model.embedding.weight.grad is not None
        assert model.lm_head.weight.grad is not None
        assert model.embedding.weight.grad.abs().sum().item() > 0
        assert model.lm_head.weight.grad.abs().sum().item() > 0

    def test_causal_full_model(self):
        """End-to-end causality: changing token at pos t doesn't affect logits at pos < t."""
        cfg = MiniQwenConfig()
        torch.manual_seed(0)
        model = MiniQwen(cfg)
        model.eval()

        torch.manual_seed(1)
        tokens = torch.randint(0, cfg.vocab_size, (1, 5))
        tokens2 = tokens.clone()
        tokens2[0, 3] = (tokens[0, 3] + 7) % cfg.vocab_size

        with torch.no_grad():
            l1 = model(tokens)
            l2 = model(tokens2)

        # Logits at positions 0..2 should be identical
        assert torch.allclose(l1[:, :3], l2[:, :3], atol=1e-5)
        # Logits at position 3 onward should differ
        assert not torch.allclose(l1[:, 3], l2[:, 3], atol=1e-4)


# ---------------------------------------------------------------------------
# vmap-readiness sanity (Phase 4 will rely on this)
# ---------------------------------------------------------------------------
def test_model_works_under_vmap_eval():
    """Smoke check that torch.func.vmap can stack multiple model copies.

    This does NOT yet train multiple seeds; just verifies the architecture
    is functional-API friendly.
    """
    from torch.func import functional_call, stack_module_state, vmap

    cfg = MiniQwenConfig()
    n_seeds = 3
    models = [MiniQwen(cfg) for _ in range(n_seeds)]
    params, buffers = stack_module_state(models)

    base = MiniQwen(cfg).to("meta")

    def fmodel(p, b, t):
        return functional_call(base, (p, b), (t,))

    tokens = torch.randint(0, cfg.vocab_size, (2, 5))
    # Same tokens across seeds; vmap over the seed dimension only.
    logits = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, tokens)
    assert logits.shape == (n_seeds, 2, 5, cfg.vocab_size)
    # Different seeds must give different logits (random init differs)
    assert not torch.allclose(logits[0], logits[1])
