"""Modular arithmetic dataset.

For p prime and op in {add, mul}, we enumerate all (a, b) pairs and produce
sequences [a, +, b, =, c] where c = (a op b) mod p.

Token ids:
  0 .. p-1     -> numbers
  p            -> '+' (or operator)
  p + 1        -> '='

Sequence length is always 5, and loss is computed only at position 4 (the
answer), per decision D5.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TaskSpec:
    p: int
    op: str = "add"  # 'add' | 'mul'

    @property
    def vocab_size(self) -> int:
        return self.p + 2

    @property
    def plus_id(self) -> int:
        return self.p

    @property
    def eq_id(self) -> int:
        return self.p + 1

    @property
    def seq_len(self) -> int:
        return 5

    @property
    def answer_pos(self) -> int:
        # position of c in [a, op, b, =, c]
        return 4


def compute_answer(a: torch.Tensor, b: torch.Tensor, spec: TaskSpec) -> torch.Tensor:
    if spec.op == "add":
        return (a + b) % spec.p
    if spec.op == "mul":
        return (a * b) % spec.p
    raise ValueError(f"Unknown op: {spec.op}")


def build_full_dataset(spec: TaskSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate all p^2 (a, b) pairs.

    Returns:
        tokens: (p^2, 5) int64 sequence [a, op, b, =, c]
        targets: (p^2,) int64 answer c (also tokens[:, -1] for convenience)
    """
    p = spec.p
    a = torch.arange(p).repeat_interleave(p)  # (p^2,)
    b = torch.arange(p).repeat(p)
    c = compute_answer(a, b, spec)

    op_id = torch.full_like(a, spec.plus_id)
    eq_id = torch.full_like(a, spec.eq_id)

    tokens = torch.stack([a, op_id, b, eq_id, c], dim=1)  # (p^2, 5)
    return tokens.long(), c.long()
