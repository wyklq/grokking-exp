"""Label-masked cross-entropy at the answer position (decision D5).

We feed the model the full sequence [a, op, b, =, c] and compute logits for
every position. For training, we only care about the prediction made at the
'=' token (position 3), whose target is c (the token at position 4).

This is the standard 'next-token prediction at one specific position' setup
used in Nanda et al.'s grokking experiments. It is mathematically identical
to feeding [a, op, b, =] and predicting the next token, but conforms to the
modern causal-LM training pipeline (decision: F3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def answer_logits_and_targets(
    logits: Tensor, tokens: Tensor, answer_pos: int
) -> tuple[Tensor, Tensor]:
    """Extract the logits used to predict the answer, and the targets.

    Args:
        logits: (B, S, V)
        tokens: (B, S)
        answer_pos: position of the answer token c (default 4 for [a,op,b,=,c])

    Returns:
        ans_logits: (B, V) — logits at position answer_pos - 1
        targets:    (B,)   — token at position answer_pos
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (B, S, V), got {tuple(logits.shape)}")
    if tokens.ndim != 2:
        raise ValueError(f"tokens must have shape (B, S), got {tuple(tokens.shape)}")
    if logits.shape[:2] != tokens.shape:
        raise ValueError(
            f"logits and tokens batch/sequence dims must match, got "
            f"{tuple(logits.shape[:2])} and {tuple(tokens.shape)}"
        )
    if not 1 <= answer_pos < tokens.shape[1]:
        raise ValueError(
            f"answer_pos={answer_pos} out of range; expected 1 <= answer_pos < {tokens.shape[1]}"
        )
    ans_logits = logits[:, answer_pos - 1, :]
    targets = tokens[:, answer_pos]
    return ans_logits, targets


def label_masked_loss_and_acc(
    logits: Tensor, tokens: Tensor, answer_pos: int = 4
) -> tuple[Tensor, Tensor]:
    """Cross-entropy + accuracy on the answer token only."""
    ans_logits, targets = answer_logits_and_targets(logits, tokens, answer_pos)
    loss = F.cross_entropy(ans_logits, targets)
    with torch.no_grad():
        preds = ans_logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
    return loss, acc
