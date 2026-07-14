#!/usr/bin/env python3
"""Fully-trained (mini-batch) CP models and the QFS bridge to dense models.

Backing the Section 6.4 analysis: trains Lambda-CP (R=256) under the DENSE
mini-batch protocol (batch 2048, shuffled, drop_last, AdamW lr 1e-3, cosine
annealing stepped per epoch, 100 epochs ~ 2,900 optimizer steps), two arms:
wd=0.1 (the published CP config, PRIMARY) and wd=1.0 (matching dense-wd,
SECONDARY, closing the weight-decay asymmetry). Then computes the QFS bridge
(input-space recomputation, k=20, float64) against fully trained dense-wd
models, with fixed anchors, gates, and one confirmatory permutation test.

Checkpoints go to checkpoints/extension_cp/minibatch/ (subdirectory keeps them
out of the emissions aggregation globs), schema-compatible with existing CP
checkpoints. Emissions recorded as wall time (wandb/codecarbon stubbed;
noted in the checkpoint).

Outputs:
    checkpoints/extension_cp/minibatch/mnist_cp_r256_lambda_minibatch_{wd01,wd10}_seed{42..46}.pt
    results/extension2/qfs_cp_bridge_minibatch.json

Env: conda from environment_cpu.yml. Run from repo root. Deterministic.

Usage:
    python scripts/figures/qfs_cp_bridge_minibatch.py
"""
import sys
import json
import time
import types
import importlib.util
import importlib.machinery
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
from src.models.cp_model import CPImageModel
from src.data import MNIST

SEEDS = [42, 43, 44, 45, 46]
K = 20
EPOCHS = 100
BATCH = 2048
CKPT_DIR = REPO / "checkpoints/extension_cp/minibatch"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

train = MNIST(train=True, device="cpu", apply_com=False)
test = MNIST(train=False, device="cpu", apply_com=False)


def train_minibatch_cp(seed, wd):
    torch.manual_seed(seed)
    model = CPImageModel(d_hidden=256, rank=256, n_classes=10,
                         cp_init_mode="lambda")
    torch.manual_seed(seed)  # mirror dense fit's re-seed before training
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = torch.nn.CrossEntropyLoss()
    loader = DataLoader(torch.utils.data.TensorDataset(train.x, train.y),
                        batch_size=BATCH, shuffle=True, drop_last=True)
    t0 = time.time()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    wall = time.time() - t0
    model.eval()
    with torch.no_grad():
        acc = (model(test.x).argmax(-1) == test.y).float().mean().item()
    return model, acc, wall


def cp_input_matrices(model):
    ev, evec = model.decompose()  # input-space eigh per class
    return ev, evec


def dense_input_matrices_from_ckpt(seed):
    ck = torch.load(REPO / f"checkpoints/vision/mnist/mnist_dense_wd_seed{seed}.pt",
                    map_location="cpu", weights_only=False)
    m = om.Model(om.Config(epochs=100, d_hidden=256, d_output=10,
                           wd=1.0, lr=1e-3, seed=seed))
    m.load_state_dict(ck["model_state_dict"])
    l, r = m.w_lr[0].unbind()
    A = torch.einsum("ch,hi,hj->cij", m.w_u.double(),
                     (l.double() @ m.w_e.double()),
                     (r.double() @ m.w_e.double()))
    return 0.5 * (A + A.transpose(1, 2))


def matrices_from_eigpairs(ev, evec):
    # reconstruct input-space matrices (float64) from decompose() eigenpairs.
    # NOTE: CPImageModel.decompose returns eigenvectors as ROWS
    # (eigenvectors[c, r, :] is the r-th eigenvector), hence "cr,cri,crj".
    ev = ev.detach().double(); evec = evec.detach().double()
    return torch.einsum("cr,cri,crj->cij", ev, evec, evec)


def cp_matrices_from_factors(model):
    # direct reconstruction from CP factors (mirrors decompose(), float64)
    bl = model.bilinear
    w_e = model.embed.weight.detach().double()
    w_h = model.head.weight.detach().double()
    A = bl.A.detach().double(); B = bl.B.detach().double()
    C = bl.C.detach().double(); lam = bl.lambdas.detach().double()
    A_proj = A.T @ w_e  # [rank, 784]
    B_proj = B.T @ w_e
    Cw = w_h @ C        # [n_classes, rank]
    M = torch.einsum("r,cr,ri,rj->cij", lam, Cw, A_proj, B_proj)
    return 0.5 * (M + M.transpose(1, 2))


def qfs_k(A, B, k=K):
    def trunc(M):
        lam, V = torch.linalg.eigh(M)
        idx = lam.abs().argsort(descending=True)[:k]
        return (V[:, idx] * lam[idx]) @ V[:, idx].T
    Ak, Bk = trunc(A), trunc(B)
    return (torch.sum(Ak * Bk) / (Ak.norm() * Bk.norm())).item()


def grid(msA, msB, exclude_same_seed=False):
    grids = []
    for sa, A in msA.items():
        for sb, B in msB.items():
            if exclude_same_seed and sa == sb:
                continue
            grids.append(np.array([[qfs_k(A[c], B[d]) for d in range(10)]
                                   for c in range(10)]))
    return np.stack(grids)


def diag_stats(g):
    d = np.stack([np.diag(x) for x in g])
    off = np.stack([x[~np.eye(10, dtype=bool)] for x in g])
    return {"diag_mean": float(d.mean()), "diag_std_over_grids": float(d.mean(1).std(ddof=1)) if len(g) > 1 else 0.0,
            "offdiag_mean": float(off.mean()),
            "n_complete_separation": int(sum(x.diagonal().min() > x[~np.eye(10, dtype=bool)].max() for x in g)),
            "n_grids": len(g)}


def permutation_test(mean_grid, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = float(np.diag(mean_grid).mean() - mean_grid[~np.eye(10, dtype=bool)].mean())
    b = 0
    for _ in range(n):
        p = rng.permutation(10)
        g = mean_grid[:, p]
        stat = float(np.diag(g).mean() - g[~np.eye(10, dtype=bool)].mean())
        if stat >= obs:
            b += 1
    return {"statistic": obs, "n_permutations": n, "n_geq": b,
            "p_value": (b + 1) / (n + 1), "seed": seed}


out = {"protocol": {"epochs": EPOCHS, "batch": BATCH, "k": K, "seeds": SEEDS,
                    "arms": {"primary": "wd=0.1", "secondary": "wd=1.0"}}}

# --- train both arms, save schema-compatible checkpoints ---
mats = {"cp_mb_wd01": {}, "cp_mb_wd10": {}}
accs = {"cp_mb_wd01": {}, "cp_mb_wd10": {}}
for wd, key, tag in ((0.1, "cp_mb_wd01", "wd01"), (1.0, "cp_mb_wd10", "wd10")):
    for s in SEEDS:
        ckpath = CKPT_DIR / f"mnist_cp_r256_lambda_minibatch_{tag}_seed{s}.pt"
        if ckpath.exists():
            ck = torch.load(ckpath, map_location="cpu", weights_only=False)
            mats[key][s] = matrices_from_eigpairs(ck["eigenvalues"], ck["eigenvectors"])
            accs[key][s] = ck["metrics"]["val_acc"]
            print(f"{key} seed {s}: reused checkpoint (acc {accs[key][s]:.4f})")
            continue
        model, acc, wall = train_minibatch_cp(s, wd)
        ev, evec = model.decompose()
        torch.save({
            "config": {"mode": "cp", "rank": 256, "cp_init_mode": "lambda",
                       "protocol": "minibatch", "epochs": EPOCHS,
                       "batch_size": BATCH, "weight_decay": wd,
                       "apply_com": False, "l1_coeff": 0.0,
                       "lambda_l1_coeff": 0.0, "lambda_l0_coeff": 0.0},
            "model_state_dict": model.state_dict(),
            "metrics": {"val_acc": acc},
            "seed": s, "eigenvalues": ev, "eigenvectors": evec,
            "emissions": {"wall_time_seconds": wall,
                          "wall_time_hours": wall / 3600, "gpu_hours": 0,
                          "co2_kg": 0,
                          "note": "CPU run; codecarbon/wandb disabled"},
        }, CKPT_DIR / f"mnist_cp_r256_lambda_minibatch_{tag}_seed{s}.pt")
        mats[key][s] = matrices_from_eigpairs(ev, evec)
        accs[key][s] = acc
        print(f"{key} seed {s}: acc {acc:.4f} wall {wall:.1f}s")
out["test_acc"] = {k: {str(s): v for s, v in a.items()} for k, a in accs.items()}

# --- acceptance check: eigenpair reconstruction must equal the
# direct factor reconstruction to <= 1e-5 for one checkpoint of each family ---
def acceptance_check(ckpath):
    ck = torch.load(ckpath, map_location="cpu", weights_only=False)
    model = CPImageModel(d_hidden=256, rank=256, n_classes=10,
                         cp_init_mode="lambda")
    model.load_state_dict(ck["model_state_dict"])
    from_pairs = matrices_from_eigpairs(ck["eigenvalues"], ck["eigenvectors"])
    from_factors = cp_matrices_from_factors(model)
    return float((from_pairs - from_factors).abs().max())

acc_fb = acceptance_check(REPO / "checkpoints/extension_cp/mnist_cp_r256_lambda_seed42.pt")
acc_mb = acceptance_check(CKPT_DIR / "mnist_cp_r256_lambda_minibatch_wd01_seed42.pt")
out["acceptance_check_max_abs_diff"] = {"fullbatch_seed42": acc_fb,
                                        "minibatch_wd01_seed42": acc_mb}
print(f"acceptance check: fb {acc_fb:.2e}, mb {acc_mb:.2e} (must be <= 1e-5)")
assert acc_fb <= 1e-5 and acc_mb <= 1e-5, "eigenpair reconstruction invalid"

# --- dense matrices + fullbatch CP matrices for anchors ---
dense = {s: dense_input_matrices_from_ckpt(s) for s in SEEDS}
cp_fb = {}
for s in SEEDS:
    ck = torch.load(REPO / f"checkpoints/extension_cp/mnist_cp_r256_lambda_seed{s}.pt",
                    map_location="cpu", weights_only=False)
    cp_fb[s] = matrices_from_eigpairs(ck["eigenvalues"], ck["eigenvectors"])

# --- gates ---
g_dense = grid(dense, dense, exclude_same_seed=True)
out["gate_ii_dense_cross_seed"] = diag_stats(g_dense)  # must ~0.855
g_mb = grid(mats["cp_mb_wd01"], mats["cp_mb_wd01"], exclude_same_seed=True)
out["anchor_b_mbcp_cross_seed"] = diag_stats(g_mb)     # gate (i): complete sep
out["gate_iii_training_health"] = {
    "min_primary_acc": min(accs["cp_mb_wd01"].values()),
    "threshold": 0.938,
    "passes": min(accs["cp_mb_wd01"].values()) >= 0.938}

# --- headline comparisons ---
for key in ("cp_mb_wd01", "cp_mb_wd10"):
    g = grid(mats[key], dense)
    out[f"{key}_x_dense_wd"] = diag_stats(g)
    out[f"{key}_x_dense_wd"]["mean_grid"] = g.mean(0).tolist()
    g2 = grid(mats[key], cp_fb)
    out[f"{key}_x_cp_fullbatch"] = diag_stats(g2)

# confirmatory permutation test: primary arm x dense-wd, seed-averaged grid
mg = np.array(out["cp_mb_wd01_x_dense_wd"]["mean_grid"])
out["confirmatory_permutation"] = permutation_test(mg)

print(json.dumps({k: v for k, v in out.items()
                  if k not in ("test_acc",) and "mean_grid" not in str(k)},
                 default=str)[:600])
dest = REPO / "results/extension2/qfs_cp_bridge_minibatch.json"
with open(dest, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", dest)

# decision criteria (I1/I2/I3) for the primary comparison
D = out["cp_mb_wd01_x_dense_wd"]["diag_mean"]
anchor_b = out["anchor_b_mbcp_cross_seed"]["diag_mean"]
print(f"\nD (primary diag) = {D:.4f}; anchor_b = {anchor_b:.4f}")
print(f"I1 if D >= {anchor_b - 0.10:.3f}; I3 if D <= 0.25; I2 between")
