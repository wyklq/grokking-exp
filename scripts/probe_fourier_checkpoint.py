"""Causal Fourier-algorithm probes for a saved Mini-Qwen checkpoint.

Use this after saving/rerunning representative comprehension-region cells. The
script tests whether Fourier modes are sufficient/necessary and whether the
model's full logit table has the additive Fourier signature.

Example:

    python scripts/probe_fourier_checkpoint.py \
        --checkpoint results/checkpoints/B_a0.3_l0.1_seed0.pt \
        --p 113 --split S3 --alpha 0.3 --tied \
        --device cuda \
        --out results/algorithm_identification/B_a0.3_l0.1_probe.json
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from mqg.data import TaskSpec, build_full_dataset, make_split
from mqg.measures.fourier import restricted_embedding
from mqg.model import MiniQwen, MiniQwenConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a checkpoint for Fourier addition algorithms.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/algorithm_identification/checkpoint_probe.json"))
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--op", choices=["add", "mul"], default="add")
    parser.add_argument("--split", choices=["S1", "S3"], default="S3")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-k-pairs", type=int, default=3)
    parser.add_argument("--shifts", type=int, nargs="*", default=[1, 2, 5, 17])

    # Architecture overrides. Defaults match configs/base.yaml for p=113.
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=1)
    parser.add_argument("--ffn-hidden", type=int, default=170)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--tied", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace, checkpoint_obj: Any, spec: TaskSpec) -> MiniQwenConfig:
    cfg_kwargs = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "head_dim": args.head_dim,
        "n_kv_heads": args.n_kv_heads,
        "ffn_hidden": args.ffn_hidden,
        "vocab_size": spec.vocab_size,
        "max_seq_len": args.max_seq_len,
        "rope_base": args.rope_base,
        "norm_eps": args.norm_eps,
        "tied_embedding": args.tied,
    }
    if isinstance(checkpoint_obj, dict):
        raw_cfg = checkpoint_obj.get("model_cfg") or checkpoint_obj.get("cfg")
        if isinstance(raw_cfg, MiniQwenConfig):
            cfg_kwargs.update(asdict(raw_cfg))
        elif isinstance(raw_cfg, dict):
            valid = set(MiniQwenConfig.__dataclass_fields__.keys())
            cfg_kwargs.update({k: v for k, v in raw_cfg.items() if k in valid})
    cfg_kwargs["vocab_size"] = spec.vocab_size
    return MiniQwenConfig(**cfg_kwargs)


def extract_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return normalize_state_dict(value)
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return normalize_state_dict(checkpoint_obj)
    raise ValueError(
        "Could not find a model state_dict. Expected a plain state_dict or one of "
        "model_state_dict/state_dict/model/model_state."
    )


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def evaluate(
    model: MiniQwen,
    tokens: torch.Tensor,
    answer_pos: int,
    *,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size].to(device)
            logits = model(batch)[:, answer_pos - 1, :]
            targets = batch[:, answer_pos]
            total_loss += float(loss_fn(logits, targets).item())
            total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
            total += int(batch.shape[0])
    return {"loss": total_loss / total, "acc": total_correct / total}


def top_canonical_freqs(E_num: torch.Tensor, p: int, n_pairs: int) -> list[int]:
    spectrum = torch.fft.fft(E_num.float(), dim=0)
    power = (spectrum.real.square() + spectrum.imag.square()).sum(dim=1)
    half = (p - 1) // 2
    pair_power = torch.stack([power[k] + power[(-k) % p] for k in range(1, half + 1)])
    top = torch.topk(pair_power, k=min(n_pairs, half)).indices + 1
    return [int(k.item()) for k in top]


def clone_with_fourier_variant(
    model: MiniQwen,
    p: int,
    freqs: list[int],
    *,
    target: str,
    mode: str,
) -> MiniQwen:
    variant = copy.deepcopy(model)
    variant.eval()
    with torch.no_grad():
        if target in {"embedding", "both"}:
            E = variant.embedding.weight[:p].detach().clone()
            keep = restricted_embedding(E, freqs)
            replacement = keep if mode == "keep" else E - keep
            variant.embedding.weight[:p].copy_(replacement)
        if target in {"lm_head", "both"} and variant.lm_head is not None:
            W = variant.lm_head.weight[:p].detach().clone()
            keep = restricted_embedding(W, freqs)
            replacement = keep if mode == "keep" else W - keep
            variant.lm_head.weight[:p].copy_(replacement)
    return variant


def logits_table(
    model: MiniQwen,
    spec: TaskSpec,
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    tokens, _ = build_full_dataset(spec)
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tokens.shape[0], batch_size):
            batch = tokens[start : start + batch_size].to(device)
            logits = model(batch)[:, spec.answer_pos - 1, : spec.p]
            chunks.append(logits.detach().cpu())
    return torch.cat(chunks, dim=0).reshape(spec.p, spec.p, spec.p)


def triadic_fft_signature(L: torch.Tensor, p: int) -> dict[str, float | list[list[int | float]]]:
    centered = L.float() - L.float().mean()
    fft = torch.fft.fftn(centered, dim=(0, 1, 2))
    power = (fft.real.square() + fft.imag.square())
    total_non_dc = power.sum() - power[0, 0, 0]
    additive = torch.stack([power[k, k, (-k) % p] for k in range(1, p)]).sum()
    line_fraction = float((additive / total_non_dc.clamp_min(torch.finfo(power.dtype).tiny)).item())

    flat = power.flatten()
    flat[0] = 0.0
    top_vals, top_idx = torch.topk(flat, k=min(12, flat.numel() - 1))
    top_modes: list[list[int | float]] = []
    for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
        ka = idx // (p * p)
        rem = idx % (p * p)
        kb = rem // p
        kc = rem % p
        top_modes.append([int(ka), int(kb), int(kc), float(val)])
    return {"additive_line_fraction": line_fraction, "top_modes": top_modes}


def cyclic_equivariance(L: torch.Tensor, p: int, shifts: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    base = L.float()
    denom = base.flatten().norm().clamp_min(torch.finfo(base.dtype).tiny)
    for s in shifts:
        shift = int(s) % p
        if shift == 0:
            continue
        a_shift = torch.roll(base, shifts=-shift, dims=0)
        b_shift = torch.roll(base, shifts=-shift, dims=1)
        c_shift = torch.roll(base, shifts=shift, dims=2)
        out[f"a_shift_{shift}_relative_mse"] = float(((a_shift - c_shift).square().mean()).item())
        out[f"b_shift_{shift}_relative_mse"] = float(((b_shift - c_shift).square().mean()).item())
        out[f"a_shift_{shift}_cosine"] = float(
            torch.nn.functional.cosine_similarity(a_shift.flatten(), c_shift.flatten(), dim=0).item()
        )
        out[f"b_shift_{shift}_cosine"] = float(
            torch.nn.functional.cosine_similarity(b_shift.flatten(), c_shift.flatten(), dim=0).item()
        )
        out[f"a_shift_{shift}_mse_over_norm"] = float(((a_shift - c_shift).flatten().norm() / denom).item())
        out[f"b_shift_{shift}_mse_over_norm"] = float(((b_shift - c_shift).flatten().norm() / denom).item())
    return out


def plot_variant_accuracies(out_path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["variant"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]
    test_acc = [row["test_acc"] for row in rows]
    x = torch.arange(len(labels)).numpy()
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 4.8))
    ax.bar(x - width / 2, train_acc, width, label="train")
    ax.bar(x + width / 2, test_acc, width, label="test")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Fourier ablation probe")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.top_k_pairs < 1:
        raise ValueError("--top-k-pairs must be >= 1")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    spec = TaskSpec(p=args.p, op=args.op)
    checkpoint_obj = torch.load(args.checkpoint, map_location="cpu")
    cfg = build_cfg(args, checkpoint_obj, spec)
    model = MiniQwen(cfg).to(device)
    state_dict = extract_state_dict(checkpoint_obj)
    load_info = model.load_state_dict(state_dict, strict=False)
    model.eval()

    tokens, _ = build_full_dataset(spec)
    train_idx, test_idx = make_split(args.split, spec, args.alpha, args.split_seed)
    train_tokens = tokens[train_idx]
    test_tokens = tokens[test_idx]

    baseline_train = evaluate(
        model, train_tokens, spec.answer_pos, batch_size=args.batch_size, device=device
    )
    baseline_test = evaluate(
        model, test_tokens, spec.answer_pos, batch_size=args.batch_size, device=device
    )

    freqs = top_canonical_freqs(model.embedding.weight[: spec.p].detach().cpu(), spec.p, args.top_k_pairs)
    variants = [{"variant": "baseline", "train_acc": baseline_train["acc"], "test_acc": baseline_test["acc"]}]
    probe_targets = ["embedding"]
    if model.lm_head is not None:
        probe_targets.extend(["lm_head", "both"])
    for target in probe_targets:
        for mode in ("keep", "remove"):
            variant_model = clone_with_fourier_variant(model, spec.p, freqs, target=target, mode=mode).to(device)
            train_eval = evaluate(
                variant_model,
                train_tokens,
                spec.answer_pos,
                batch_size=args.batch_size,
                device=device,
            )
            test_eval = evaluate(
                variant_model,
                test_tokens,
                spec.answer_pos,
                batch_size=args.batch_size,
                device=device,
            )
            variants.append(
                {
                    "variant": f"{target}_{mode}_top{args.top_k_pairs}",
                    "train_loss": train_eval["loss"],
                    "train_acc": train_eval["acc"],
                    "test_loss": test_eval["loss"],
                    "test_acc": test_eval["acc"],
                }
            )

    L = logits_table(model, spec, batch_size=args.batch_size, device=device)
    fft_signature = triadic_fft_signature(L, spec.p)
    equivariance = cyclic_equivariance(L, spec.p, args.shifts)

    result = {
        "checkpoint": str(args.checkpoint),
        "config": asdict(cfg),
        "task": {"p": args.p, "op": args.op, "split": args.split, "alpha": args.alpha},
        "load_info": {
            "missing_keys": list(load_info.missing_keys),
            "unexpected_keys": list(load_info.unexpected_keys),
        },
        "top_canonical_freqs": freqs,
        "chance_accuracy": 1.0 / args.p,
        "baseline": {"train": baseline_train, "test": baseline_test},
        "variants": variants,
        "triadic_fft": fft_signature,
        "cyclic_equivariance": equivariance,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_variant_accuracies(args.out.with_suffix(".accuracy.png"), variants)

    print(f"[saved] {args.out}")
    print(f"[saved] {args.out.with_suffix('.accuracy.png')}")
    print(f"[baseline] train_acc={baseline_train['acc']:.4f} test_acc={baseline_test['acc']:.4f}")
    print(f"[freqs] {freqs}")
    print(f"[triadic_fft] additive_line_fraction={fft_signature['additive_line_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())