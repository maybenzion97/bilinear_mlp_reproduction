#!/usr/bin/env python3
"""Robustness checks for the Section 6.4 QFS bridge comparison.

Checks, labelled to match the output JSON keys (H4 runs first and gates
the rest):
  H4 reconstruction fidelity: per-class quadratic forms vs true model
     logits; bridge vs functional CP matrix conventions
  sanity gate: reproduce the stored bridge numbers
     (results/extension2/qfs_cp_bridge.json) with this script
  H1 gauge: class-mean-centered QFS variants
  H3 truncation: full-rank QFS alongside rank-20
  H2 functional equivalence: test-set logit correlations
  H5 accuracy confound: per-class accuracy vs per-class QFS

All matrix arithmetic float64. Uses existing checkpoints only (no new
training).

Inputs: data/MNIST/raw test files, checkpoints/vision/mnist (dense
none/wd), checkpoints/extension_cp (Lambda-CP R=256), and
results/extension2/qfs_cp_bridge.json.
Output: results/extension2/qfs_bridge_robustness.json
Env: conda from environment_cpu.yml.

Usage:
    python scripts/figures/qfs_bridge_robustness.py
"""
import json
import itertools
import numpy as np
import torch

torch.set_grad_enabled(False)

import pathlib
REPO = str(pathlib.Path(__file__).resolve().parents[2])
SP = REPO + "/results/extension2"
SEEDS = [42, 43, 44, 45, 46]
K = 20
OUT = {}

# ---------------------------------------------------------------- data
def load_mnist_test():
    import struct
    raw = REPO + "/data/MNIST/raw"
    with open(raw + "/t10k-images-idx3-ubyte", "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)
    with open(raw + "/t10k-labels-idx1-ubyte", "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    x = torch.from_numpy(x.copy()).double() / 255.0  # matches training: /255, no CoM
    return x, torch.from_numpy(y.copy()).long()

X, Y = load_mnist_test()
print(f"MNIST test: {X.shape}, labels {Y.min()}..{Y.max()}")

# ---------------------------------------------------------------- checkpoints
def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["config"]["apply_com"] is False
    return {k: v.double() for k, v in ck["model_state_dict"].items()}, ck["metrics"]

dense, cp = {}, {}
metrics = {}
for cfg in ["none", "wd"]:
    for s in SEEDS:
        sd, m = load_sd(f"{REPO}/checkpoints/vision/mnist/mnist_dense_{cfg}_seed{s}.pt")
        dense[(cfg, s)] = sd
        metrics[f"dense_{cfg}_{s}"] = m["val_acc"]
for s in SEEDS:
    sd, m = load_sd(f"{REPO}/checkpoints/extension_cp/mnist_cp_r256_lambda_seed{s}.pt")
    cp[s] = sd
    metrics[f"cp_lambda_{s}"] = m["val_acc"]

# ---------------------------------------------------------------- forwards (ground truth)
def dense_logits(sd, x):
    w_e, w = sd["embed.weight"], sd["blocks.0.weight"]
    w_l, w_r = w.chunk(2, dim=0)
    w_u = sd["head.weight"]
    h = x @ w_e.T
    return ((h @ w_l.T) * (h @ w_r.T)) @ w_u.T

def cp_logits(sd, x):
    # exact replica of BilinearCP.forward, lambda mode
    A, B, C = sd["bilinear.A"], sd["bilinear.B"], sd["bilinear.C"]
    lam = sd["bilinear.lambdas"]
    eps = 1e-8
    A_n = A / (A.norm(dim=0, keepdim=True) + eps)
    B_n = B / (B.norm(dim=0, keepdim=True) + eps)
    mask = (lam.abs() > lam.abs().max() * 0.05).double()
    glam = lam * mask
    h = x @ sd["embed.weight"].T
    hidden = (h @ A_n) * (h @ B_n) * glam
    return (hidden @ C.T) @ sd["head.weight"].T

# ---------------------------------------------------------------- interaction matrices
def dense_mats(sd):
    w_e = sd["embed.weight"]
    w_l, w_r = sd["blocks.0.weight"].chunk(2, dim=0)
    w_u = sd["head.weight"]
    lp, rp = w_l @ w_e, w_r @ w_e
    return torch.einsum("ch,hi,hj->cij", w_u, lp, rp)

def cp_mats_bridge(sd):
    # verbatim math of scripts/figures/qfs_cp_bridge.py::cp_input_matrices
    A, B, C = sd["bilinear.A"], sd["bilinear.B"], sd["bilinear.C"]
    lam = sd["bilinear.lambdas"]
    w_e, w_h = sd["embed.weight"], sd["head.weight"]
    a_proj, b_proj = A.T @ w_e, B.T @ w_e
    scale = (w_h @ C) * lam
    return torch.einsum("cr,ri,rj->cij", scale, a_proj, b_proj)

def cp_mats_functional(sd):
    # matches BilinearCP.forward: column-normalized A,B + threshold-gated lambdas
    A, B, C = sd["bilinear.A"], sd["bilinear.B"], sd["bilinear.C"]
    lam = sd["bilinear.lambdas"]
    eps = 1e-8
    A_n = A / (A.norm(dim=0, keepdim=True) + eps)
    B_n = B / (B.norm(dim=0, keepdim=True) + eps)
    glam = lam * (lam.abs() > lam.abs().max() * 0.05).double()
    w_e, w_h = sd["embed.weight"], sd["head.weight"]
    a_proj, b_proj = A_n.T @ w_e, B_n.T @ w_e
    scale = (w_h @ C) * glam
    return torch.einsum("cr,ri,rj->cij", scale, a_proj, b_proj)

# ---------------------------------------------------------------- H4 (gate)
# Decision criterion: bridge convention valid for a family iff
#   max_c,x |x^T A_c x - logit_c(x)| / rms(logits) < 1e-6 on 100 test images.
# If the CP bridge convention fails but the functional reconstruction passes,
# the bridge matrices differ from what the model computes; severity judged by
# QFS(bridge, functional) and downstream deltas.
print("\n=== H4: reconstruction vs true logits (100 test images) ===")
x100 = X[:100]
h4 = {}
def quad_forms(mats, x):
    return torch.einsum("bi,cij,bj->bc", x, mats, x)

for name, sd, fwd, matfn in [
    ("dense_wd_42", dense[("wd", 42)], dense_logits, dense_mats),
    ("dense_none_42", dense[("none", 42)], dense_logits, dense_mats),
    ("cp_bridge_42", cp[42], cp_logits, cp_mats_bridge),
    ("cp_functional_42", cp[42], cp_logits, cp_mats_functional),
]:
    true = fwd(sd, x100)
    qf = quad_forms(matfn(sd), x100)
    rel = (qf - true).abs().max().item() / true.pow(2).mean().sqrt().item()
    h4[name] = rel
    print(f"  {name:20s} max|qf-logit|/rms = {rel:.3e}")

# lambda mask status across all CP seeds
mask_counts = {}
for s in SEEDS:
    lam = cp[s]["bilinear.lambdas"]
    mask_counts[s] = int((lam.abs() <= lam.abs().max() * 0.05).sum())
print("  CP masked-lambda counts per seed:", mask_counts)
OUT["H4_logit_reproduction_relerr"] = h4
OUT["H4_cp_masked_lambda_counts"] = mask_counts

# class alignment + forward accuracy vs stored metrics
print("\n  forward accuracy vs checkpoint val_acc:")
acc_check = {}
logits_all = {}   # model tag -> [10000, 10] float64
for cfg in ["none", "wd"]:
    for s in SEEDS:
        lg = dense_logits(dense[(cfg, s)], X)
        logits_all[f"dense_{cfg}_{s}"] = lg
for s in SEEDS:
    logits_all[f"cp_lambda_{s}"] = cp_logits(cp[s], X)
for tag, lg in logits_all.items():
    acc = (lg.argmax(1) == Y).double().mean().item()
    acc_check[tag] = {"forward_acc": acc, "ckpt_val_acc": metrics[tag],
                      "abs_diff": abs(acc - metrics[tag])}
    if abs(acc - metrics[tag]) > 2e-3:
        print(f"  MISMATCH {tag}: fwd {acc:.4f} vs ckpt {metrics[tag]:.4f}")
worst = max(v["abs_diff"] for v in acc_check.values())
print(f"  worst |forward_acc - ckpt_val_acc| = {worst:.2e}")
OUT["H4_accuracy_check_worst_absdiff"] = worst
OUT["H4_accuracy_check"] = acc_check

# per-class accuracy (class alignment + H5)
percls_acc = {}
for tag, lg in logits_all.items():
    pred = lg.argmax(1)
    percls_acc[tag] = [float((pred[Y == c] == c).double().mean()) for c in range(10)]
OUT["per_class_accuracy"] = percls_acc

# ---------------------------------------------------------------- matrix stacks
def center(mats):
    return mats - mats.mean(dim=0, keepdim=True)

def sym(mats):
    return 0.5 * (mats + mats.transpose(1, 2))

stacks = {}  # (variant, seed) -> [10,784,784] symmetric
for cfg in ["none", "wd"]:
    for s in SEEDS:
        m = sym(dense_mats(dense[(cfg, s)]))
        stacks[(f"dense_{cfg}", s)] = m
        stacks[(f"dense_{cfg}_ctr", s)] = center(m)
for s in SEEDS:
    mb = sym(cp_mats_bridge(cp[s]))
    mf = sym(cp_mats_functional(cp[s]))
    stacks[("cp_bridge", s)] = mb
    stacks[("cp_bridge_ctr", s)] = center(mb)
    stacks[("cp_func", s)] = mf
    stacks[("cp_func_ctr", s)] = center(mf)

# rank-k reconstructions (cached)
def rank_k(mats, k=K):
    vals, vecs = torch.linalg.eigh(mats)
    idx = vals.abs().argsort(dim=-1, descending=True)[:, :k]
    v = torch.gather(vals, 1, idx)
    U = torch.gather(vecs.transpose(1, 2), 1,
                     idx.unsqueeze(-1).expand(-1, -1, vecs.shape[-1]))
    return torch.einsum("ck,cki,ckj->cij", v, U, U)

print("\ncomputing rank-20 truncations for", len(stacks), "stacks ...")
rk = {key: rank_k(m) for key, m in stacks.items()}

def qfs_grid(a, b):
    cross = torch.einsum("cij,dij->cd", a, b)
    na = a.flatten(1).norm(dim=1)
    nb = b.flatten(1).norm(dim=1)
    return (cross / na.unsqueeze(1) / nb.unsqueeze(0)).numpy()

def combo(va, vb, source):
    """diag/offdiag summary over all distinct-model seed pairs."""
    if va == vb:
        pairs = list(itertools.combinations(SEEDS, 2))
    else:
        pairs = [(a, b) for a in SEEDS for b in SEEDS]
    grids = [qfs_grid(source[(va, sa)], source[(vb, sb)]) for sa, sb in pairs]
    diags = np.concatenate([np.diag(g) for g in grids])
    offs = np.concatenate([g[~np.eye(10, dtype=bool)] for g in grids])
    mean_grid = np.mean(grids, axis=0)
    return {"n_pairs": len(pairs),
            "diag_mean": float(diags.mean()), "diag_std": float(diags.std()),
            "offdiag_mean": float(offs.mean()),
            "per_class_diag": np.diag(mean_grid).tolist()}

COMBOS = [
    # raw published-convention (sanity + H3 baseline)
    ("dense_wd", "dense_wd"), ("dense_none", "dense_none"),
    ("cp_bridge", "cp_bridge"), ("dense_none", "dense_wd"),
    ("dense_wd", "cp_bridge"), ("dense_none", "cp_bridge"),
    # H4-corrected CP matrices
    ("cp_func", "cp_func"), ("dense_wd", "cp_func"), ("dense_none", "cp_func"),
    ("cp_bridge", "cp_func"),
    # H1 centered
    ("dense_wd_ctr", "dense_wd_ctr"), ("dense_none_ctr", "dense_none_ctr"),
    ("cp_func_ctr", "cp_func_ctr"), ("cp_bridge_ctr", "cp_bridge_ctr"),
    ("dense_none_ctr", "dense_wd_ctr"),
    ("dense_wd_ctr", "cp_func_ctr"), ("dense_none_ctr", "cp_func_ctr"),
    ("dense_wd_ctr", "cp_bridge_ctr"),
]

print("\n=== QFS grids: rank-20 and full-rank ===")
res_k, res_full = {}, {}
for va, vb in COMBOS:
    key = f"{va}__x__{vb}"
    res_k[key] = combo(va, vb, rk)
    res_full[key] = combo(va, vb, stacks)
    print(f"  {key:44s} k20 diag {res_k[key]['diag_mean']:+.4f} "
          f"(off {res_k[key]['offdiag_mean']:+.3f})   "
          f"full diag {res_full[key]['diag_mean']:+.4f} "
          f"(off {res_full[key]['offdiag_mean']:+.3f})")
OUT["qfs_rank20"] = res_k
OUT["qfs_fullrank"] = res_full

# same-model bridge-vs-functional CP similarity (H4 severity)
same_model = []
for s in SEEDS:
    g = qfs_grid(rk[("cp_bridge", s)], rk[("cp_func", s)])
    same_model.append(float(np.diag(g).mean()))
OUT["H4_same_model_bridge_vs_func_diag_k20"] = same_model
print("\nH4 severity: same-model QFS(cp_bridge, cp_func) diag per seed:",
      [f"{v:.4f}" for v in same_model])

# sanity gate vs published JSON
pub = json.load(open(REPO + "/results/extension2/qfs_cp_bridge.json"))
gate = {
    "dense_wd_cross_seed": (res_k["dense_wd__x__dense_wd"]["diag_mean"],
                            pub["same_config_cross_seed"]["dense_wd_cross_seed"]["diag_mean"]),
    "dense_wd_x_cp_lambda": (res_k["dense_wd__x__cp_bridge"]["diag_mean"],
                             pub["cross_config"]["dense_wd_x_cp_lambda"]["summary"]["diag_mean"]),
    "dense_none_x_dense_wd": (res_k["dense_none__x__dense_wd"]["diag_mean"],
                              pub["cross_config"]["dense_none_x_dense_wd"]["summary"]["diag_mean"]),
    "cp_lambda_cross_seed": (res_k["cp_bridge__x__cp_bridge"]["diag_mean"],
                             pub["same_config_cross_seed"]["cp_lambda_cross_seed"]["diag_mean"]),
}
print("\n=== sanity gate (mine vs published, criterion |diff|<1e-4) ===")
for k_, (mine, theirs) in gate.items():
    print(f"  {k_:24s} mine {mine:.6f} published {theirs:.6f} diff {abs(mine-theirs):.2e}")
OUT["sanity_gate"] = {k_: {"mine": m, "published": p} for k_, (m, p) in gate.items()}

# ---------------------------------------------------------------- H2 functional
# Decision criteria: qualification REQUIRED if
#   ratio := mean same-class centered-logit corr (dense-wd x cp) /
#            mean same-class centered-logit corr (dense-wd x dense-wd cross-seed)
#   > 0.9. Conclusion functionally supported if < 0.7.
print("\n=== H2: test-set functional similarity (10k images) ===")
def centered_logits(lg):
    return lg - lg.mean(dim=1, keepdim=True)

def pair_stats(tag_a, tag_b):
    la, lb = logits_all[tag_a], logits_all[tag_b]
    ca, cb = centered_logits(la), centered_logits(lb)
    def pcorr(u, v):
        u = u - u.mean(0); v = v - v.mean(0)
        return (u * v).sum(0) / (u.norm(dim=0) * v.norm(dim=0))
    return {
        "raw_corr_per_class": pcorr(la, lb).tolist(),
        "ctr_corr_per_class": pcorr(ca, cb).tolist(),
        "raw_corr_mean": float(pcorr(la, lb).mean()),
        "ctr_corr_mean": float(pcorr(ca, cb).mean()),
        "pred_agreement": float((la.argmax(1) == lb.argmax(1)).double().mean()),
    }

def group(pairs):
    st = [pair_stats(a, b) for a, b in pairs]
    return {
        "n_pairs": len(st),
        "raw_corr_mean": float(np.mean([s["raw_corr_mean"] for s in st])),
        "ctr_corr_mean": float(np.mean([s["ctr_corr_mean"] for s in st])),
        "pred_agreement_mean": float(np.mean([s["pred_agreement"] for s in st])),
        "ctr_corr_per_class_mean": np.mean([s["ctr_corr_per_class"] for s in st], axis=0).tolist(),
    }

def tags(prefix):
    return [f"{prefix}_{s}" for s in SEEDS]

within = lambda pref: list(itertools.combinations(tags(pref), 2))
across = lambda pa, pb: [(a, b) for a in tags(pa) for b in tags(pb)]

h2 = {
    "within_dense_wd": group(within("dense_wd")),
    "within_dense_none": group(within("dense_none")),
    "within_cp": group(within("cp_lambda")),
    "dense_none_x_dense_wd": group(across("dense_none", "dense_wd")),
    "dense_wd_x_cp": group(across("dense_wd", "cp_lambda")),
    "dense_none_x_cp": group(across("dense_none", "cp_lambda")),
}
for k_, v in h2.items():
    print(f"  {k_:24s} raw corr {v['raw_corr_mean']:.4f}  "
          f"ctr corr {v['ctr_corr_mean']:.4f}  agree {v['pred_agreement_mean']:.4f}")
ratio = h2["dense_wd_x_cp"]["ctr_corr_mean"] / h2["within_dense_wd"]["ctr_corr_mean"]
print(f"  H2 ratio (cross-family / within-dense-wd, centered corr) = {ratio:.4f}")
OUT["H2"] = h2
OUT["H2_ratio_ctr_corr"] = ratio

# error-overlap: expected agreement if CP errors were superset of dense errors
err_wd = np.mean([1 - acc_check[t]["forward_acc"] for t in tags("dense_wd")])
err_cp = np.mean([1 - acc_check[t]["forward_acc"] for t in tags("cp_lambda")])
OUT["H2_error_rates"] = {"dense_wd": err_wd, "cp": err_cp}

# ---------------------------------------------------------------- H5 accuracy confound
# Criterion: strong confound signal if |pearson r| > 0.6 between per-class
# seed-avg QFS diag (wd x cp, published convention) and per-class acc gap.
print("\n=== H5: per-class accuracy confound ===")
qfs_diag_pc = np.array(res_k["dense_wd__x__cp_bridge"]["per_class_diag"])
acc_wd_pc = np.mean([percls_acc[t] for t in tags("dense_wd")], axis=0)
acc_cp_pc = np.mean([percls_acc[t] for t in tags("cp_lambda")], axis=0)
gap_pc = acc_wd_pc - acc_cp_pc
r_qfs_gap = float(np.corrcoef(qfs_diag_pc, gap_pc)[0, 1])
ctr_corr_pc = np.array(h2["dense_wd_x_cp"]["ctr_corr_per_class_mean"])
r_func_gap = float(np.corrcoef(ctr_corr_pc, gap_pc)[0, 1])
r_qfs_func = float(np.corrcoef(qfs_diag_pc, ctr_corr_pc)[0, 1])
print(f"  per-class acc gap (wd-cp): {np.round(gap_pc,4).tolist()}")
print(f"  per-class QFS diag        : {np.round(qfs_diag_pc,3).tolist()}")
print(f"  r(QFS diag, acc gap)  = {r_qfs_gap:+.3f}")
print(f"  r(func corr, acc gap) = {r_func_gap:+.3f}")
print(f"  r(QFS diag, func corr)= {r_qfs_func:+.3f}")
OUT["H5"] = {"per_class_acc_gap": gap_pc.tolist(),
             "r_qfs_diag_vs_acc_gap": r_qfs_gap,
             "r_func_corr_vs_acc_gap": r_func_gap,
             "r_qfs_diag_vs_func_corr": r_qfs_func}

with open(SP + "/qfs_bridge_robustness.json", "w") as fh:
    json.dump(OUT, fh, indent=2)
print("\nwrote", SP + "/qfs_bridge_robustness.json")
