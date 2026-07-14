#!/usr/bin/env python3
"""Budget-matched control for the Section 6.4 QFS bridge experiment.

Trains dense models under the CP protocol (full-batch, 100 optimizer steps,
wd=1.0) and compares their per-class decision surfaces, via rank-20 QFS in
input space, against the fully trained dense-wd checkpoints. Answers whether
the CP-vs-dense QFS gap reflects the CP constraint or the optimization
budget: if undertrained dense models diverge from fully trained dense models
as much as CP models do, the gap is a budget effect.

Inputs: data/MNIST/raw and checkpoints/vision/mnist (dense-wd, seeds 42-46).

Outputs:
    results/extension2/qfs_budget_control.json
    bilinear_mlp_reproduction_report/figures/extension_cp/budget_matched_dense_eigvecs.pdf

Env: conda from environment_cpu.yml. Run from repo root. Deterministic
(seeds 42-46; torch.manual_seed per model).

Usage:
    python scripts/figures/qfs_budget_control.py
"""
import sys
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
for name in ("wandb", "codecarbon"):
    sys.modules.setdefault(name, types.ModuleType(name))

spec = importlib.util.spec_from_file_location(
    "orig_model", REPO / "bilinear-decomposition-main/image/model.py")
om = importlib.util.module_from_spec(spec)
spec.loader.exec_module(om)

SEEDS = [42, 43, 44, 45, 46]
K = 20
STEPS = 100
WD = 1.0


def load_mnist(split):
    raw = REPO / "data/MNIST/raw"
    stem = "train" if split == "train" else "t10k"
    with open(raw / f"{stem}-images-idx3-ubyte", "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        x = torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8)
                             .reshape(n, r * c).copy()).float() / 255.0
    with open(raw / f"{stem}-labels-idx1-ubyte", "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        y = torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8).copy()).long()
    return x, y


xtr, ytr = load_mnist("train")
xte, yte = load_mnist("test")


def logits(m, xx):
    out = m(xx)
    return out.logits if hasattr(out, "logits") else out


def dense_input_matrices(model):
    # input-space symmetrized per-class interaction matrices, float64
    # (matches Model.decompose's einsum convention, conjugated by w_e)
    l, r = model.w_lr[0].unbind()
    w_u = model.w_u.double()
    w_e = model.w_e.double()
    L = l.double() @ w_e
    R = r.double() @ w_e
    A = torch.einsum("ch,hi,hj->cij", w_u, L, R)
    return 0.5 * (A + A.transpose(1, 2))


def qfs_k(A, B, k=K):
    def trunc(M):
        lam, V = torch.linalg.eigh(M)
        idx = lam.abs().argsort(descending=True)[:k]
        return (V[:, idx] * lam[idx]) @ V[:, idx].T
    Ak, Bk = trunc(A), trunc(B)
    return (torch.sum(Ak * Bk) / (Ak.norm() * Bk.norm())).item()


def train_fullbatch_dense(seed, steps=STEPS, wd=WD):
    torch.manual_seed(seed)
    m = om.Model(om.Config(epochs=steps, d_hidden=256, d_output=10,
                           wd=wd, lr=1e-3, seed=seed))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        loss = crit(logits(m, xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    with torch.no_grad():
        acc = (logits(m, xte).argmax(-1) == yte).float().mean().item()
    return m, acc


print(f"training {len(SEEDS)} budget-matched dense models "
      f"(full-batch, {STEPS} steps, wd={WD})...")
fb_mats, fb_accs = {}, {}
for s in SEEDS:
    m, acc = train_fullbatch_dense(s)
    fb_mats[s], fb_accs[s] = dense_input_matrices(m), acc
    print(f"  seed {s}: test acc {acc:.4f}")

pub_mats = {}
for s in SEEDS:
    ck = torch.load(REPO / f"checkpoints/vision/mnist/mnist_dense_wd_seed{s}.pt",
                    map_location="cpu", weights_only=False)
    m = om.Model(om.Config(epochs=100, d_hidden=256, d_output=10,
                           wd=1.0, lr=1e-3, seed=s))
    m.load_state_dict(ck["model_state_dict"])
    pub_mats[s] = dense_input_matrices(m)


def diag_stats(ms1, ms2, exclude_same_seed=False):
    vals = []
    for s1, A in ms1.items():
        for s2, B in ms2.items():
            if exclude_same_seed and s1 == s2:
                continue
            vals.append(float(np.mean([qfs_k(A[c], B[c]) for c in range(10)])))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
            "n": len(vals)}


# effective rank of the budget-matched models (mean over classes, then seeds)
def eff_rank(mats):
    vals = []
    for A in mats.values():
        lam = torch.linalg.eigvalsh(A)          # [10, 784]
        l = lam.abs()
        vals.append(((l.sum(1) ** 2) / (l ** 2).sum(1)).mean().item())
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}


# eigenvector grid figure for the qualitative (localization) control
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A42 = fb_mats[42]
fig, axes = plt.subplots(3, 5, figsize=(7.5, 4.8))
for col in range(5):
    lam, V = torch.linalg.eigh(A42[col])
    idx = lam.abs().argsort(descending=True)[:3]
    for row in range(3):
        ax = axes[row, col]
        ax.imshow(V[:, idx[row]].reshape(28, 28), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"digit {col}", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"eig {row + 1}", fontsize=8)
plt.tight_layout()
FIGDIR = REPO.parent / "bilinear_mlp_reproduction_report/figures/extension_cp"
fig.savefig(FIGDIR / "budget_matched_dense_eigvecs.pdf", bbox_inches="tight")
plt.close(fig)

out = {
    "protocol": {"steps": STEPS, "wd": WD, "k": K, "seeds": SEEDS,
                 "note": "dense trained under the CP full-batch protocol"},
    "fb_test_acc": fb_accs,
    "fb_eff_rank": eff_rank(fb_mats),
    "sanity_pub_cross_seed": diag_stats(pub_mats, pub_mats, exclude_same_seed=True),
    "control_fb_x_pub": diag_stats(fb_mats, pub_mats),
    "anchor_fb_cross_seed": diag_stats(fb_mats, fb_mats, exclude_same_seed=True),
    "published_reference": {"dense_wd_x_cp_lambda": 0.175,
                            "cp_cross_seed": 0.686,
                            "dense_wd_cross_seed": 0.855},
}
print(json.dumps({k: v for k, v in out.items() if k != "fb_test_acc"}, indent=2))
dest = REPO / "results/extension2/qfs_budget_control.json"
with open(dest, "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote", dest)
