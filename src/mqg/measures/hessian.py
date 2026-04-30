"""Track 2c: Loss landscape — dominant Hessian eigenvalue via power iteration.

Sharpness ≈ |λ| of the loss Hessian H = ∇²L(θ). Power iteration converges
to the eigenvalue with **largest magnitude** (not necessarily positive),
which is standard in the deep-learning sharpness literature (PyHessian
etc.). Negative dominant eigenvalues indicate genuinely non-convex
regions of the loss landscape.

Compute via HVP with double-backward through autograd:
    v_{k+1} = H v_k / ||H v_k||
    λ_dom  ≈ v^T H v
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import Tensor


def _flatten(tensors: Iterable[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten_like(flat: Tensor, ref: list[Tensor]) -> list[Tensor]:
    out = []
    offset = 0
    for r in ref:
        n = r.numel()
        out.append(flat[offset:offset + n].view_as(r))
        offset += n
    return out


def hvp(
    loss_fn: Callable[[], Tensor],
    params: list[Tensor],
    v: Tensor,
) -> Tensor:
    """Hessian-vector product: H v, where H = ∇²(loss_fn) wrt params.

    Args:
        loss_fn: zero-arg callable that returns a scalar loss; must depend on `params`.
        params: list of leaf parameters (requires_grad=True).
        v: flat vector of same total dim as params.

    Returns:
        H v as a flat tensor.
    """
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = _flatten(grads)
    Hv = torch.autograd.grad(flat_grad, params, grad_outputs=v, retain_graph=False)
    return _flatten(Hv).detach()


def top_hessian_eigenvalue(
    loss_fn: Callable[[], Tensor],
    params: list[Tensor],
    n_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 0,
) -> float:
    """Estimate the top eigenvalue of ∇²(loss_fn) via power iteration.

    Returns the Rayleigh quotient v^T H v after convergence (or n_iter steps).
    """
    g = torch.Generator(device=params[0].device).manual_seed(seed)
    n_total = sum(p.numel() for p in params)
    v = torch.randn(n_total, generator=g, device=params[0].device, dtype=params[0].dtype)
    v = v / v.norm()
    last = float("inf")
    for _ in range(n_iter):
        Hv = hvp(loss_fn, params, v)
        norm = Hv.norm()
        if norm.item() < 1e-30:
            return 0.0
        v_new = Hv / norm
        eig = float((v_new @ Hv).item())  # Rayleigh quotient v_new^T H v_new ≈ v_new^T (norm * v_new)
        if abs(eig - last) < tol * max(abs(eig), 1.0):
            v = v_new
            last = eig
            break
        v = v_new
        last = eig
    # Final Rayleigh quotient
    Hv = hvp(loss_fn, params, v)
    return float((v @ Hv).item())
