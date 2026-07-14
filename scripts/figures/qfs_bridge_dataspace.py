#!/usr/bin/env python3
"""Data-manifold variants of the Section 6.4 dense-vs-CP QFS comparison.

Asks whether the dense-vs-CP weight-space QFS gap lives ON or OFF the data
manifold. Transformations of the per-class matrices (functional CP
reconstruction):
  proj-p : A -> P A P, P = projector onto top-p eigvecs of E[xx^T] (test set)
  whiten : A -> S^{1/2} A S^{1/2}, S = E[xx^T]  (data-metric QFS; equals the
           second-moment-weighted similarity of the quadratic forms)

Decision criteria: if the cross-family diag under the data metric rises to
>= 0.7x the same-transform dense x dense anchor, the raw QFS gap is
substantially an off-manifold artifact; if it stays < 0.5x, the surfaces
differ on-manifold too.

Inputs: data/MNIST/raw test images, checkpoints/vision/mnist (dense
none/wd), and checkpoints/extension_cp (Lambda-CP R=256). No new training.
Output: results/extension2/qfs_bridge_dataspace.json
Env: conda from environment_cpu.yml.

Usage:
    python scripts/figures/qfs_bridge_dataspace.py
"""
import pathlib
import itertools, json
import numpy as np
import torch

torch.set_grad_enabled(False)
REPO = str(pathlib.Path(__file__).resolve().parents[2])
SP = str(pathlib.Path(__file__).resolve().parents[2] / "results/extension2")
SEEDS = [42, 43, 44, 45, 46]

import importlib.util
spec = importlib.util.spec_from_file_location("qf", str(pathlib.Path(__file__).resolve().parent / "qfs_bridge_robustness.py"))
# qfs_bridge_robustness.py executes at import; re-implement the few pieces needed
import struct
raw = REPO + "/data/MNIST/raw"
with open(raw + "/t10k-images-idx3-ubyte", "rb") as f:
    _, n, r, c = struct.unpack(">IIII", f.read(16))
    X = torch.from_numpy(
        np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c).copy()
    ).double() / 255.0

def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.double() for k, v in ck["model_state_dict"].items()}

def dense_mats(sd):
    w_e = sd["embed.weight"]
    w_l, w_r = sd["blocks.0.weight"].chunk(2, dim=0)
    return torch.einsum("ch,hi,hj->cij", sd["head.weight"], w_l @ w_e, w_r @ w_e)

def cp_mats_functional(sd):
    A, B, C = sd["bilinear.A"], sd["bilinear.B"], sd["bilinear.C"]
    lam = sd["bilinear.lambdas"]
    A_n = A / (A.norm(dim=0, keepdim=True) + 1e-8)
    B_n = B / (B.norm(dim=0, keepdim=True) + 1e-8)
    glam = lam * (lam.abs() > lam.abs().max() * 0.05).double()
    w_e, w_h = sd["embed.weight"], sd["head.weight"]
    scale = (w_h @ C) * glam
    return torch.einsum("cr,ri,rj->cij", scale, A_n.T @ w_e, B_n.T @ w_e)

def sym(m): return 0.5 * (m + m.transpose(1, 2))

stacks = {}
for cfg in ["none", "wd"]:
    for s in SEEDS:
        stacks[(f"dense_{cfg}", s)] = sym(dense_mats(
            load_sd(f"{REPO}/checkpoints/vision/mnist/mnist_dense_{cfg}_seed{s}.pt")))
for s in SEEDS:
    stacks[("cp", s)] = sym(cp_mats_functional(
        load_sd(f"{REPO}/checkpoints/extension_cp/mnist_cp_r256_lambda_seed{s}.pt")))

# data second moment and its eigendecomposition
S = (X.T @ X) / X.shape[0]           # [784, 784]
evals, evecs = torch.linalg.eigh(S)  # ascending
evals, evecs = evals.flip(0), evecs.flip(1)
frac = (evals.cumsum(0) / evals.sum())
print("data 2nd-moment spectrum: p=50 captures %.4f, p=100 %.4f of trace"
      % (frac[49], frac[99]))
S_half = evecs @ torch.diag(evals.clamp(min=0).sqrt()) @ evecs.T

def transform_all(fn):
    return {k: fn(v) for k, v in stacks.items()}

def qfs_grid(a, b):
    cross = torch.einsum("cij,dij->cd", a, b)
    return (cross / a.flatten(1).norm(dim=1).unsqueeze(1)
            / b.flatten(1).norm(dim=1).unsqueeze(0)).numpy()

def combo(src, va, vb):
    pairs = (list(itertools.combinations(SEEDS, 2)) if va == vb
             else [(a, b) for a in SEEDS for b in SEEDS])
    grids = [qfs_grid(src[(va, sa)], src[(vb, sb)]) for sa, sb in pairs]
    diags = np.concatenate([np.diag(g) for g in grids])
    offs = np.concatenate([g[~np.eye(10, dtype=bool)] for g in grids])
    return float(diags.mean()), float(offs.mean())

OUT = {}
variants = {"raw_fullrank": lambda m: m}
for p in [50, 100]:
    U = evecs[:, :p]
    P = U @ U.T
    variants[f"proj{p}"] = lambda m, P=P: torch.einsum("ij,cjk,kl->cil", P, m, P)
variants["whitened"] = lambda m: torch.einsum("ij,cjk,kl->cil", S_half, m, S_half)

COMBOS = [("dense_wd", "dense_wd"), ("cp", "cp"), ("dense_none", "dense_wd"),
          ("dense_wd", "cp"), ("dense_none", "cp")]
for vname, fn in variants.items():
    src = transform_all(fn)
    OUT[vname] = {}
    print(f"\n--- {vname} ---")
    for va, vb in COMBOS:
        d, o = combo(src, va, vb)
        OUT[vname][f"{va}__x__{vb}"] = {"diag_mean": d, "offdiag_mean": o}
        print(f"  {va:12s} x {vb:12s} diag {d:+.4f} off {o:+.3f}")
    ratio = (OUT[vname]["dense_wd__x__cp"]["diag_mean"]
             / OUT[vname]["dense_wd__x__dense_wd"]["diag_mean"])
    OUT[vname]["cross_over_within_ratio"] = ratio
    print(f"  cross/within ratio = {ratio:.4f}")

with open(SP + "/qfs_bridge_dataspace.json", "w") as fh:
    json.dump(OUT, fh, indent=2)
print("\nwrote", SP + "/qfs_bridge_dataspace.json")
