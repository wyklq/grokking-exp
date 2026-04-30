"""Track 1: Fourier-based progress measures (anchored to embedding structure).

These measures detect "circuit-level grokking" — whether the embedding
matrix has organized into the Fourier basis of Z/pZ, which is the
mechanism Nanda et al. (2023) and Liu et al. (2022) identified as the
underlying algorithm for modular addition.
"""
from __future__ import annotations

import torch
from torch import Tensor

# ---------- helpers ----------

def _fft_power_spectrum(E: Tensor) -> Tensor:
    """Compute per-frequency total power of an embedding matrix.

    Args:
        E: (p, d) real embedding matrix (one row per number token).

    Returns:
        (p,) tensor of total power per frequency: P[k] = sum_d |F[k, d]|^2
    """
    F = torch.fft.fft(E, dim=0)  # (p, d) complex
    return (F.real ** 2 + F.imag ** 2).sum(dim=1)


def _gini(x: Tensor) -> Tensor:
    """Gini coefficient of a non-negative 1-D tensor.

    G = 0 means perfectly uniform; G -> 1 means concentrated on one entry.
    """
    x = x.flatten()
    if bool((x < 0).any().item()):
        raise ValueError("Gini requires non-negative values")
    n = x.numel()
    if n == 0 or x.sum().item() == 0:
        return x.new_tensor(0.0)
    sorted_x, _ = torch.sort(x)
    idx = torch.arange(1, n + 1, dtype=sorted_x.dtype, device=sorted_x.device)
    return ((2 * idx - n - 1) * sorted_x).sum() / (n * sorted_x.sum())


# ---------- measures ----------

def fourier_sparsity(E: Tensor, exclude_dc: bool = True) -> float:
    """Gini coefficient of the per-frequency power spectrum of E.

    A grokked embedding concentrates power on a few frequencies → high Gini.
    DC component (k=0) is excluded by default since it captures the mean.

    Args:
        E: (p, d) embedding matrix
        exclude_dc: drop the k=0 component before computing Gini

    Returns:
        Gini coefficient in [0, 1]; higher = sparser/more grokked.
    """
    power = _fft_power_spectrum(E)
    if exclude_dc:
        power = power[1:]
    return float(_gini(power).item())


def dominant_frequencies(E: Tensor, k: int = 5, exclude_dc: bool = True) -> list[int]:
    """Return the top-k frequencies (excluding DC) by power, sorted descending.

    For real-valued E of length p, frequencies k and p-k are conjugates
    with identical power; we keep both.
    """
    power = _fft_power_spectrum(E)
    if exclude_dc:
        # set DC to 0 power so it doesn't appear
        power = power.clone()
        power[0] = 0.0
    top = torch.topk(power, k=min(k, power.numel())).indices
    return [int(i.item()) for i in top]


def restricted_embedding(E: Tensor, freqs: list[int]) -> Tensor:
    """Reconstruct E using only the given Fourier frequencies (zeroing out the rest).

    Useful for computing "restricted loss" — feeding the model an
    embedding stripped to only its dominant Fourier modes.

    Args:
        E: (p, d) real embedding
        freqs: list of frequency indices to keep (and their conjugates if real)

    Returns:
        (p, d) real-valued reconstruction.
    """
    p, d = E.shape
    F = torch.fft.fft(E, dim=0)
    mask = torch.zeros(p, dtype=torch.bool, device=E.device)
    for k in freqs:
        mask[k % p] = True
        mask[(-k) % p] = True  # conjugate, ensures real reconstruction
    F_masked = torch.where(mask.unsqueeze(1), F, torch.zeros_like(F))
    return torch.fft.ifft(F_masked, dim=0).real


def circularity(E: Tensor, freq: int) -> float:
    """Measure how well rows of E lie on a circle in the 2-D plane spanned by
    the (cos, sin) basis at frequency `freq`.

    Returns coefficient of variation (CV) of the per-row radius:
        CV = std(r) / mean(r)
    Lower CV = more circular = more grokked.
    """
    if not E.is_floating_point():
        E = E.float()
    p = E.shape[0]
    tiny = torch.finfo(E.dtype).tiny
    n = torch.arange(p, dtype=E.dtype, device=E.device)
    angle = 2 * torch.pi * freq * n / p
    cos_b = torch.cos(angle)  # (p,)
    sin_b = torch.sin(angle)
    # project: amplitude along each direction = E^T basis (per dim)
    # but we want a single (cos, sin) pair per token, not per d.
    # Following Nanda: take the dot products of each token's embedding
    # with cos_b/sin_b; each dim contributes a (cos, sin) coordinate.
    # The 2D point per token n is (E[n] @ a_cos, E[n] @ a_sin) for some
    # learned amplitude vectors a_cos, a_sin in R^d. Best a_cos, a_sin
    # come from regressing E ≈ outer(cos_b, a_cos) + outer(sin_b, a_sin).
    # Closed form: a_cos = (cos_b @ E) / (cos_b @ cos_b), similarly a_sin.
    a_cos = (cos_b @ E) / (cos_b @ cos_b)  # (d,)
    a_sin = (sin_b @ E) / (sin_b @ sin_b)
    x = E @ a_cos / (a_cos @ a_cos + tiny)  # (p,) — coordinate along cos axis
    y = E @ a_sin / (a_sin @ a_sin + tiny)
    r = torch.sqrt(x ** 2 + y ** 2 + tiny)
    return float((r.std() / (r.mean() + tiny)).item())
