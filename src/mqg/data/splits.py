"""Train/test splits.

Two strategies are supported (decision D6):

  S1: random split over the p^2 (a, b) pairs.
      train_fraction alpha controls fraction of pairs in train.

  S3: 'b-column' split. We randomly select a fraction alpha of b-values to be
      seen during training. ALL pairs (a, b_seen) go into train. The remaining
      (a, b_unseen) form the test set. This tests true algebraic / token-OOD
      generalization (the unseen b's are never input tokens at training time).
"""
from __future__ import annotations

import torch

from .dataset import TaskSpec


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha}")


def split_S1(
    n_pairs: int, alpha: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random split. Returns (train_idx, test_idx) into the p^2 pair list."""
    _validate_alpha(alpha)
    if n_pairs < 2:
        raise ValueError(f"n_pairs must be >= 2, got {n_pairs}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_pairs, generator=g)
    n_train = min(max(1, int(round(alpha * n_pairs))), n_pairs - 1)
    return perm[:n_train].sort().values, perm[n_train:].sort().values


def split_S3(
    spec: TaskSpec, alpha: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick a fraction alpha of b-values; all pairs with those b's go to train.

    Returns (train_idx, test_idx) into the p^2 pair list.
    Pair index = a * p + b (matches build_full_dataset's enumeration order).
    """
    _validate_alpha(alpha)
    p = spec.p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p, generator=g)
    n_train_b = min(max(1, int(round(alpha * p))), p - 1)
    train_b = set(perm[:n_train_b].tolist())

    b_all = torch.arange(p).repeat(p)
    train_mask = torch.tensor([b.item() in train_b for b in b_all])

    train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
    test_idx = torch.nonzero(~train_mask, as_tuple=True)[0]
    return train_idx, test_idx


def make_split(
    strategy: str,
    spec: TaskSpec,
    alpha: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_pairs = spec.p * spec.p
    if strategy == "S1":
        return split_S1(n_pairs, alpha, seed)
    if strategy == "S3":
        return split_S3(spec, alpha, seed)
    raise ValueError(f"Unknown split strategy: {strategy}")
