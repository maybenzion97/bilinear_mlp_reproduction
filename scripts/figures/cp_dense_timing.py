#!/usr/bin/env python3
"""Matched-protocol timing of dense vs CP bilinear models.

Backing the Section 6.5 analysis: the observed ~28x wall-clock gap between
CP and dense training runs is a protocol artifact (full-batch, 100 optimizer
steps vs mini-batch, ~2,900 steps with a shuffling loader), not an
architectural speed advantage: at matched protocol the CP layer is
marginally slower per epoch and has slightly more parameters.

Input: data/MNIST/raw training files (timing only; no checkpoints written).
Output: results/extension2/cp_dense_timing.json
Env: conda from environment_cpu.yml (torch, numpy). CPU timing; ratios,
not absolute times, are the quantity of interest. Run from repo root.

Usage:
    python scripts/figures/cp_dense_timing.py
"""
import sys
import time
import json
import struct
import types
import importlib.util
import pathlib

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bilinear-decomposition-main"))
# stubs: timing does not use experiment tracking
for name in ("wandb", "codecarbon"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
torch.manual_seed(0)

spec = importlib.util.spec_from_file_location(
    "orig_model", REPO / "bilinear-decomposition-main/image/model.py")
om = importlib.util.module_from_spec(spec)
spec.loader.exec_module(om)

raw = REPO / "data/MNIST/raw"
with open(raw / "train-images-idx3-ubyte", "rb") as f:
    _, n, r, c = struct.unpack(">IIII", f.read(16))
    x = torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8)
                         .reshape(n, r * c).copy()).float() / 255.0
with open(raw / "train-labels-idx1-ubyte", "rb") as f:
    _, n = struct.unpack(">II", f.read(8))
    y = torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8).copy()).long()

dense = om.Model(om.Config(epochs=100, d_hidden=256, d_output=10,
                           wd=1.0, lr=1e-3, seed=0))
from src.models.cp_model import CPImageModel
cp = CPImageModel(d_hidden=256, rank=256, n_classes=10, cp_init_mode="lambda")

crit = torch.nn.CrossEntropyLoss()


def logits(model, xx):
    out = model(xx)
    return out.logits if hasattr(out, "logits") else out


def time_fullbatch(model, n_steps=10):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = crit(logits(model, x), y)  # warmup
    opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loss = crit(logits(model, x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return (time.perf_counter() - t0) / n_steps


def time_minibatch_epoch(model, bs=2048):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randperm(x.shape[0])
    t0 = time.perf_counter()
    for i in range(0, x.shape[0] - bs + 1, bs):
        b = idx[i:i + bs]
        loss = crit(logits(model, x[b]), y[b])
        opt.zero_grad(); loss.backward(); opt.step()
    return time.perf_counter() - t0


n_params = lambda m: sum(p.numel() for p in m.parameters())
out = {
    "params": {"dense": n_params(dense), "cp_lambda_r256": n_params(cp)},
    "fullbatch_step_s": {"dense": time_fullbatch(dense), "cp": time_fullbatch(cp)},
    "minibatch_epoch_s": {"dense": time_minibatch_epoch(dense), "cp": time_minibatch_epoch(cp)},
}
# rank sweep: training-step time scales with R (linear interaction cost);
# saving is capped by the shared 784->256 embedding
out["rank_sweep_fullbatch_step_s"] = {}
for R in (8, 32, 64, 128, 256):
    torch.manual_seed(0)
    m = CPImageModel(d_hidden=256, rank=R, n_classes=10, cp_init_mode="lambda")
    out["rank_sweep_fullbatch_step_s"][f"cp_r{R}"] = time_fullbatch(m)
out["ratios"] = {
    "cp_over_dense_fullbatch": out["fullbatch_step_s"]["cp"] / out["fullbatch_step_s"]["dense"],
    "cp_over_dense_minibatch": out["minibatch_epoch_s"]["cp"] / out["minibatch_epoch_s"]["dense"],
    "dense_minibatch_over_fullbatch": out["minibatch_epoch_s"]["dense"] / out["fullbatch_step_s"]["dense"],
}
print(json.dumps(out, indent=2))
dest = REPO / "results/extension2/cp_dense_timing.json"
with open(dest, "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote", dest)
