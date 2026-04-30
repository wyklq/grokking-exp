"""Phase 5: Progress measures (dual track).

Track 1 (Fourier-anchored, mechanism-specific):
    - fourier_sparsity: Gini of per-frequency power
    - dominant_frequencies: top-k freqs by power
    - restricted_embedding: rebuild E from selected freqs only
    - circularity: how circular E rows are at a given freq

Track 2 (architecture-agnostic, mechanism-free):
    - weight_norm / per_layer_weight_norms: L2 control
    - stable_rank / effective_rank: representation capacity
    - linear_cka / collect_activations: representational similarity
    - top_hessian_eigenvalue: loss-landscape sharpness

`compute_all_measures` runs the model-level measures on a single
MiniQwen checkpoint + dataset, returning a flat dict.
"""
from __future__ import annotations

from torch import Tensor

from .fourier import (
    circularity,
    dominant_frequencies,
    fourier_sparsity,
    restricted_embedding,
)
from .hessian import hvp, top_hessian_eigenvalue
from .representations import collect_activations, linear_cka
from .weights import effective_rank, per_layer_weight_norms, stable_rank, weight_norm

__all__ = [
    "fourier_sparsity",
    "dominant_frequencies",
    "restricted_embedding",
    "circularity",
    "weight_norm",
    "per_layer_weight_norms",
    "stable_rank",
    "effective_rank",
    "linear_cka",
    "collect_activations",
    "top_hessian_eigenvalue",
    "hvp",
    "compute_all_measures",
]


def compute_all_measures(
    model,
    train_tokens: Tensor,
    answer_pos: int,
    p: int,
    *,
    n_dom: int = 5,
    hessian_iters: int = 10,
    hessian_seed: int = 0,
    skip_hessian: bool = False,
) -> dict[str, float]:
    """Compute the suite of single-checkpoint measures for a MiniQwen.

    Returns a flat dict suitable for logging into a DataFrame row.

    The Fourier measures operate on the *number-token* slice of the
    embedding matrix (rows 0..p-1). CKA between checkpoints must be
    computed externally (it needs a reference checkpoint).
    """
    out: dict[str, float] = {}

    E_full = model.embedding.weight.detach().float()
    E_num = E_full[:p]
    out["fourier_sparsity"] = fourier_sparsity(E_num, exclude_dc=True)
    doms = dominant_frequencies(E_num, k=n_dom, exclude_dc=True)
    for i, f in enumerate(doms):
        out[f"dom_freq_{i}"] = float(f)
    if doms:
        out["circularity_top_freq"] = circularity(E_num, freq=doms[0])

    out["weight_norm_total"] = weight_norm(model)
    pw = per_layer_weight_norms(model)
    for k, v in pw.items():
        out[f"wnorm/{k}"] = v

    out["embedding_stable_rank"] = stable_rank(E_full)
    out["embedding_effective_rank"] = effective_rank(E_full)
    if hasattr(model, "lm_head") and model.lm_head is not None:
        out["lm_head_effective_rank"] = effective_rank(model.lm_head.weight.detach().float())

    if not skip_hessian:
        from ..train.loss import label_masked_loss_and_acc
        params = [p_ for p_ in model.parameters() if p_.requires_grad]

        def loss_fn():
            logits = model(train_tokens)
            loss, _ = label_masked_loss_and_acc(logits, train_tokens, answer_pos)
            return loss

        out["hessian_top_eig"] = top_hessian_eigenvalue(
            loss_fn, params, n_iter=hessian_iters, seed=hessian_seed,
        )

    return out
