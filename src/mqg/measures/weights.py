"""Track 2a: Weight-based architecture-agnostic measures."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def weight_norm(model: nn.Module, p: float = 2.0) -> float:
    """Total Lp norm of all model parameters (default L2)."""
    sq = 0.0
    for param in model.parameters():
        sq += float(param.detach().norm(p=p).item() ** p)
    return float(sq ** (1.0 / p))


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
    if W.dim() == 1:
        W = W.unsqueeze(0)
    fro = float((W.detach() ** 2).sum().item())
    op = float(torch.linalg.matrix_norm(W.detach(), ord=2).item())
    if op == 0:
        return 0.0
    return fro / (op ** 2)


def effective_rank(W: Tensor) -> float:
    """Entropy-based effective rank: exp(H(p)) where p_i = sigma_i / sum sigma_j.

    Equals true rank for orthonormal columns; smoothly degrades when
    singular values decay.
    """
    if W.dim() == 1:
        W = W.unsqueeze(0)
    s = torch.linalg.svdvals(W.detach())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * torch.log(p + 1e-30)).sum()
    return float(torch.exp(H).item())
