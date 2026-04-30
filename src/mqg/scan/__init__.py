"""Phase-diagram scanning infrastructure (Phase 4)."""
from .boundary import detect_boundary_cells, majority_phase
from .grid import GridCell, GridSpec, default_alpha_grid, default_grid_spec, default_lambda_grid
from .instrumented import run_cell_with_measures, to_dataframe, unstack_seed
from .multi_seed import train_multi_seed
from .scan_runner import CellRunRecord, run_cell, run_phase1, run_phase2

__all__ = [
    "GridCell",
    "GridSpec",
    "default_alpha_grid",
    "default_lambda_grid",
    "default_grid_spec",
    "detect_boundary_cells",
    "majority_phase",
    "train_multi_seed",
    "CellRunRecord",
    "run_cell",
    "run_phase1",
    "run_phase2",
    "run_cell_with_measures",
    "to_dataframe",
    "unstack_seed",
]
