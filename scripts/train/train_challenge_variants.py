#!/usr/bin/env python3
"""
Train MNIST challenge-task variants (none/noise/wd/full) and save checkpoints.

This is intentionally separate from generate_vision_figures.py: analysis should not train.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from shared.components import Bilinear, Linear
from src.data.challenge_dataset import create_challenge_datasets, ChallengeDatasetWrapper
from src.utils import track_emissions, get_device


def train_challenge_variant(
    *,
    device: str,
    seed: int,
    noise_std: float,
    weight_decay: float,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 2048,
    threshold: float = 0.6,
    target_digit: int = 1,
    target_index: int = 0,
) -> Dict:
    torch.manual_seed(seed)

    train_data, test_data, target = create_challenge_datasets(
        device=device,
        threshold=threshold,
        target_digit=target_digit,
        target_index=target_index,
    )
    train_wrapped = ChallengeDatasetWrapper(train_data)
    test_wrapped = ChallengeDatasetWrapper(test_data)

    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch]).float()
        y = torch.stack([item[1] for item in batch])
        return x, y

    loader = DataLoader(
        train_wrapped._dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    d_hidden = 256

    class ChallengeModel(torch.nn.Module):
        def __init__(self, d_input: int = 784, d_hidden_: int = 256, d_output: int = 2, bias: bool = True):
            super().__init__()
            self.embed = Linear(d_input, d_hidden_, bias=False)
            self.bilinear = Bilinear(d_hidden_, d_hidden_, bias=bias)
            self.head = Linear(d_hidden_, d_output, bias=False)
            self.criterion = torch.nn.CrossEntropyLoss()

        def forward(self, x):
            x = x.flatten(start_dim=1)
            x = self.embed(x)
            x = self.bilinear(x)
            return self.head(x)

        def step(self, x, y):
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
            return loss, acc

        @property
        def w_e(self) -> torch.Tensor:
            return self.embed.weight.data

        @property
        def w_u(self) -> torch.Tensor:
            return self.head.weight.data

        @property
        def w_l(self) -> torch.Tensor:
            return self.bilinear.w_l

        @property
        def w_r(self) -> torch.Tensor:
            return self.bilinear.w_r

        def decompose_difference(self):
            w_diff = self.w_u[1] - self.w_u[0]  # [d_hidden]
            b = torch.einsum("o,oi,oj->ij", w_diff, self.w_l, self.w_r)
            b = 0.5 * (b + b.T)
            vals, vecs = torch.linalg.eigh(b.cpu())
            vecs = torch.einsum("ec,ei->ci", vecs, self.w_e.cpu())  # [comp, inp]
            sort_idx = torch.argsort(vals.abs(), descending=True)
            return vals[sort_idx], vecs[sort_idx]

    model = ChallengeModel(d_hidden_=d_hidden, bias=True).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_std > 0:
                xb = (xb + noise_std * torch.randn_like(xb)).clamp(0, 1)
            loss, _ = model.step(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = model.step(test_wrapped.x.to(device), test_wrapped.y.to(device))
        history.append({"val_loss": float(val_loss.item()), "val_acc": float(val_acc.item())})

    eigenvalues, eigenvectors = model.decompose_difference()

    return {
        "config": {
            "d_hidden": d_hidden,
            "threshold": threshold,
            "epochs": epochs,
            "noise_std": noise_std,
            "weight_decay": weight_decay,
        },
        "model_state_dict": model.state_dict(),
        "eigenvalues": eigenvalues.cpu(),
        "eigenvectors": eigenvectors.cpu(),
        "target_image": target.cpu(),
        "history": history,
        "seed": seed,
    }


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Single seed (default: all 5 seeds)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="checkpoints/vision/challenge")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use all 5 seeds by default, or single seed if specified
    seeds = [args.seed] if args.seed is not None else DEFAULT_SEEDS
    print(f"Training with seeds: {seeds}")

    variants_spec = {
        "none": (0.0, 0.0),
        "noise": (0.5, 0.0),
        "wd": (0.0, 1.0),
        "full": (0.5, 1.0),
    }

    total_trained = 0
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print(f"{'='*50}")
        
        for tag, (noise_std, wd) in variants_spec.items():
            path = out_dir / f"mnist_challenge_{tag}_seed{seed}.pt"
            print(f"Training {tag}: noise_std={noise_std}, weight_decay={wd} -> {path}")
            
            # Track emissions for each variant
            with track_emissions("bilinear-mlp-reproduction-challenge") as tracker:
                ckpt = train_challenge_variant(
                    device=device,
                    seed=seed,
                    noise_std=noise_std,
                    weight_decay=wd,
                    epochs=args.epochs,
                )
            
            # Add emissions data to checkpoint
            ckpt["emissions"] = {
                "wall_time_hours": tracker.result.wall_time_hours,
                "wall_time_seconds": tracker.result.wall_time_seconds,
                "gpu_hours": tracker.result.gpu_hours,
                "co2_kg": tracker.result.emissions_kg,
            }
            
            torch.save(ckpt, path)
            print(f"  Wall time: {tracker.result.wall_time_seconds:.1f}s, GPU hours: {tracker.result.gpu_hours:.4f}")
            total_trained += 1

    print(f"\nDone. Trained {total_trained} checkpoints ({len(seeds)} seeds × {len(variants_spec)} variants).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

