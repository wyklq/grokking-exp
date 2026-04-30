"""Modular arithmetic dataset and S1/S3 splits."""
from .dataset import TaskSpec, build_full_dataset, compute_answer
from .splits import make_split, split_S1, split_S3

__all__ = [
    "TaskSpec",
    "build_full_dataset",
    "compute_answer",
    "make_split",
    "split_S1",
    "split_S3",
]
