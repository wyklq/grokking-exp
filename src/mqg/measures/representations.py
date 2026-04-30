"""Track 2b: Representation-based measures (CKA, NTK alignment).

Linear CKA between two activation matrices X, Y of shape (n, d_x) and (n, d_y):
    HSIC_linear(X, Y) = ||X_c^T Y_c||_F^2
    CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
where X_c, Y_c are mean-centered. Invariant to orthogonal rotations and
isotropic scaling. CKA(X, X) = 1; CKA(X, Y) for unrelated X, Y → 0.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _center(X: Tensor) -> Tensor:
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: Tensor, Y: Tensor) -> float:
    """Linear CKA between two activation matrices on the same n inputs.

    Args:
        X: (n, d_x)
        Y: (n, d_y)

    Returns:
        CKA in [0, 1]; 1 = identical representations up to rotation/scale.
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same n"
    Xc = _center(X.detach().float())
    Yc = _center(Y.detach().float())
    hsic_xy = (Xc.T @ Yc).pow(2).sum()
    hsic_xx = (Xc.T @ Xc).pow(2).sum()
    hsic_yy = (Yc.T @ Yc).pow(2).sum()
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom.item() < 1e-30:
        return 0.0
    return float((hsic_xy / denom).item())


def collect_activations(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    inputs: Tensor,
) -> Tensor:
    """Run `model(inputs)` and capture the output of `layer` via a forward hook.

    Returns:
        Activations as a (n, ...) tensor flattened to (n, -1).
    """
    captured: list[Tensor] = []

    def hook(_mod, _inp, out):
        captured.append(out.detach())

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("No activation captured; layer not run.")
    act = captured[0]
    n = act.shape[0]
    return act.reshape(n, -1)
