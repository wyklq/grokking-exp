"""Smoke test — verify package imports and config loads."""
import torch
from omegaconf import OmegaConf
from pathlib import Path

import mqg


def test_package_imports():
    assert mqg.__version__ == "0.1.0"


def test_torch_available():
    x = torch.zeros(2, 3)
    assert x.shape == (2, 3)


def test_base_config_loads():
    cfg_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    cfg = OmegaConf.load(cfg_path)

    # Locked architecture decisions
    assert cfg.model.d_model == 64
    assert cfg.model.n_layers == 2
    assert cfg.model.n_heads == 4
    assert cfg.model.n_kv_heads == 1
    assert cfg.model.vocab_size == 115

    # Locked task
    assert cfg.task.p == 113
    assert cfg.task.vocab.plus_token_id == 113
    assert cfg.task.vocab.eq_token_id == 114


def test_subpackages_importable():
    from mqg import model, data, train, measures, scan  # noqa: F401


def test_configure_matmul_precision_accepts_none_and_high():
    from mqg.perf import configure_matmul_precision

    configure_matmul_precision(None)
    configure_matmul_precision("high")
