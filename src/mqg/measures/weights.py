"""Track 2a: Weight-based architecture-agnostic measures."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def weight_norm(model: nn.Module, p: float = 2.0) -> float:
    """Total Lp norm of all model parameters (default L2)."""
    norms = [param.detach().float().norm(p=p) for param in model.parameters()]
    if not norms:
        return 0.0
    return float(torch.stack(norms).pow(p).sum().pow(1.0 / p).item())


def per_layer_weight_norms(model: nn.Module, p: float = 2.0) -> dict[str, float]:
    """L2 norm per parameter tensor (named)."""
    return {
        name: float(param.detach().norm(p=p).item())
        for name, param in model.named_parameters()
    }


def stable_rank(W: Tensor) -> float:
    """Stable rank: ||W||_F^2 / ||W||_2^2.

    Always >= 1 for nonzero W; equals true rank when all singular values equal.
    """
    W = W.detach().float()
    if W.dim() == 1:
        W = W.unsqueeze(0)
    fro = (W ** 2).sum()
    op = torch.linalg.matrix_norm(W, ord=2)
    if op.item() == 0.0:
        return 0.0
    return float((fro / (op ** 2)).item())


def effective_rank(W: Tensor) -> float:
    """Entropy-based effective rank: exp(H(p)) where p_i = sigma_i / sum sigma_j.

    Equals true rank for orthonormal columns; smoothly degrades when
    singular values decay.
    """
    W = W.detach().float()
    if W.dim() == 1:
        W = W.unsqueeze(0)
    s = torch.linalg.svdvals(W)
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H).item())
