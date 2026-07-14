#!/usr/bin/env python3
"""Recompute the 5-seed MNIST<->EMNIST-Digits transfer accuracies from checkpoints.

Backs the cross-dataset transfer table (97.53 / 97.91). The single-seed
preliminary value in mechanism_stability_results.json
(emnist_on_mnist=0.9836) predates the 5-seed protocol; the values written
here are the ones to use. Uses existing checkpoints only (no new training).

Inputs: checkpoints/extension_cross_dataset (mnist_dense_full_com and
emnist_digits_regularized, seeds 42-46).
Output: results/extension_cross_dataset/transfer_accuracy_check.json
Env: conda from environment_cpu.yml (torch, torchvision). Run from repo root.

Usage:
    python scripts/figures/verify_transfer_accuracies.py
"""
import sys
import json
import types
import importlib.util
import importlib.machinery
import pathlib

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bilinear-decomposition-main"))
for n in ("wandb", "codecarbon"):
    mod = types.ModuleType(n)
    mod.__spec__ = importlib.machinery.ModuleSpec(n, None)
    sys.modules.setdefault(n, mod)

spec = importlib.util.spec_from_file_location(
    "om", REPO / "bilinear-decomposition-main/image/model.py")
om = importlib.util.module_from_spec(spec)
spec.loader.exec_module(om)
from src.data import MNIST, EMNISTDigits

SEEDS = range(42, 47)

mnist_te = MNIST(train=False, device="cpu", apply_com=True)
emnist_te = EMNISTDigits(train=False, device="cpu", apply_com=True)


def logits(m, xx):
    out = m(xx)
    return out.logits if hasattr(out, "logits") else out


def accuracy(model, data):
    with torch.no_grad():
        return (logits(model, data.x).argmax(-1) == data.y).float().mean().item()


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = om.Model(om.Config(epochs=100, d_hidden=256, d_output=10,
                           wd=1.0, lr=1e-3, seed=0))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


m2e, e2m = [], []
for s in SEEDS:
    m2e.append(accuracy(load(
        REPO / f"checkpoints/extension_cross_dataset/mnist_dense_full_com_seed{s}.pt"),
        emnist_te))
    e2m.append(accuracy(load(
        REPO / f"checkpoints/extension_cross_dataset/emnist_digits_regularized_seed{s}.pt"),
        mnist_te))

out = {
    "mnist_to_emnist_digits": {"per_seed": m2e, "mean": float(np.mean(m2e)),
                               "std": float(np.std(m2e, ddof=1))},
    "emnist_digits_to_mnist": {"per_seed": e2m, "mean": float(np.mean(e2m)),
                               "std": float(np.std(e2m, ddof=1))},
    "note": ("5-seed recomputation from checkpoints; supersedes the single-seed "
             "preliminary accuracies in mechanism_stability_results.json"),
}
print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
dest = REPO / "results/extension_cross_dataset/transfer_accuracy_check.json"
with open(dest, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", dest)
