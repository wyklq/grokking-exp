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
        out.append(flat[offset : offset + n].view_as(r))
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
    if not params:
        raise ValueError("params must be non-empty")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    devices = {p.device for p in params}
    if len(devices) != 1:
        raise ValueError(f"all params must be on the same device, got {devices}")
    dtypes = {p.dtype for p in params}
    if len(dtypes) != 1:
        raise ValueError(f"all params must have the same dtype, got {dtypes}")

    g = torch.Generator(device=params[0].device).manual_seed(seed)
    n_total = sum(p.numel() for p in params)
    v = torch.randn(n_total, generator=g, device=params[0].device, dtype=params[0].dtype)
    v = v / v.norm()
    zero_threshold = torch.finfo(v.dtype).tiny
    last_mag = float("inf")
    for _ in range(n_iter):
        Hv = hvp(loss_fn, params, v)
        norm = Hv.norm()
        eig_mag = float(norm.item())
        if eig_mag < zero_threshold:
            return 0.0
        v_new = Hv / norm
        if abs(eig_mag - last_mag) < tol * max(eig_mag, 1.0):
            v = v_new
            break
        v = v_new
        last_mag = eig_mag
    # Final signed Rayleigh quotient. The iteration converges by magnitude,
    # but this preserves negative dominant eigenvalues.
    Hv = hvp(loss_fn, params, v)
    return float((v @ Hv).item())
