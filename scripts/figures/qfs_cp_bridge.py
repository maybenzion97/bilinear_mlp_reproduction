#!/usr/bin/env python3
"""QFS bridge between Extension 1 (QFS) and Extension 2 (CP training).

Question: are the low-rank per-class surfaces that eigendecomposition
DISCOVERS in dense models the same objects that CP ENFORCES in training?
QFS compares quadratic decision surfaces directly from weights, data-free.

Both checkpoint families are reduced to the SAME convention: per-class
interaction matrices in input space, A_c = w_e^T Q_c w_e (dense) or the
CP-factor reconstruction (matching CPImageModel.decompose,
src/models/cp_model.py:147-224), symmetrized, 784-d eigh, float64.
Dense checkpoint-stored eigenpairs are NOT used: they live in the 256-d
embedding space with w_e-projected non-orthonormal eigenvectors, which
breaks the orthonormality assumption of the eigenpair QFS estimator.

Grids: 10x10 QFS(A_c^X, A_d^Y) at k=20 (top |lambda|), via
compute_weighted_similarity(method='quadratic_form') from
src/vision/subspace.py (default k=10 there; k=20 passed explicitly).

Anchors: dense x dense and CP x CP cross-seed diagonals (10 unordered
distinct-seed pairs; i=j excluded), off-diagonal floor, dense-none vs
dense-wd and dense-noise diagonals (regularization bracket).

Statistics: ONE confirmatory permutation test, dense-wd x Lambda-CP
(R=256) on the seed-averaged 10x10 matrix,
statistic = mean(diagonal) - mean(off-diagonal), one-sided greater,
p = (b+1)/(n+1), n = 10000 seeded resamples. All other combinations are
descriptive only.

Runtime assertions: apply_com == False for every bridge checkpoint;
float64 CP spectra match the checkpoint-stored spectra (= notebook
03b_cp_extension.ipynb inputs) for one checkpoint; eigenpair QFS equals
the direct rank-20 matrix Frobenius cosine to <= 1e-5 on dense pairs.

Also quantifies the estimator discrepancy on the Sec 5 headline pair
(mnist_dense_full_com_seed42 vs emnist_letters_regularized_seed42):
checkpoint-stored embedding-space eigenpair estimator vs direct rank-20
input-space matrix Frobenius cosine.

Inputs: checkpoints/vision/mnist (dense configs, seeds 42-46),
checkpoints/extension_cp (CP modes, seeds 42-46), and
checkpoints/extension_cross_dataset (Sec 5 headline pair).

Outputs:
    results/extension2/qfs_cp_bridge.json
    bilinear_mlp_reproduction_report/figures/extension_cp/qfs_cp_bridge.pdf

Env: conda from environment_cpu.yml.

Usage:
    python scripts/figures/qfs_cp_bridge.py [--k 20] [--n-perm 10000]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.subspace import compute_weighted_similarity

MNIST_CKPT = PROJECT_ROOT / "checkpoints/vision/mnist"
CP_CKPT = PROJECT_ROOT / "checkpoints/extension_cp"
XDATA_CKPT = PROJECT_ROOT / "checkpoints/extension_cross_dataset"
RESULTS_JSON = PROJECT_ROOT / "results/extension2/qfs_cp_bridge.json"
FIGURE_PDF = (
    PROJECT_ROOT.parent
    / "bilinear_mlp_reproduction_report/figures/extension_cp/qfs_cp_bridge.pdf"
)

SEEDS = [42, 43, 44, 45, 46]
DENSE_CONFIGS = ["none", "wd", "full", "noise", "noise015"]
CP_MODES = ["lambda", "gated", "fixed"]
CP_RANK = 256
PERM_SEED = 0
CONFIRMATORY = ("wd", "lambda")  # primary comparison; everything else descriptive


def load_ckpt(path: Path, expect_com: bool) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    com = ck["config"]["apply_com"]
    assert com == expect_com, f"{path.name}: apply_com={com}, expected {expect_com}"
    return ck


def eig_sorted(mats: torch.Tensor):
    """Symmetrize + eigh a [C, d, d] float64 stack; sort by |lambda| desc.

    Returns vals [C, d] and vecs [C, d, d] with vecs[c, i] the i-th
    eigenvector (rows), matching src.vision.subspace conventions.
    """
    mats = 0.5 * (mats + mats.transpose(1, 2))
    vals, vecs = torch.linalg.eigh(mats)
    idx = vals.abs().argsort(dim=-1, descending=True)
    vals = torch.gather(vals, 1, idx)
    vecs = torch.gather(
        vecs.transpose(1, 2), 1, idx.unsqueeze(-1).expand(-1, -1, vecs.shape[-1])
    )
    return vals, vecs


def dense_input_matrices(sd: dict) -> torch.Tensor:
    """A_c = w_e^T Q_c w_e in pixel space, float64. Shape [C, 784, 784]."""
    w_e = sd["embed.weight"].double()  # [256, 784]
    w_l, w_r = sd["blocks.0.weight"].double().chunk(2, dim=0)  # [256, 256] each
    w_u = sd["head.weight"].double()  # [C, 256]
    lp = w_l @ w_e  # [256, 784]
    rp = w_r @ w_e
    return torch.einsum("ch,hi,hj->cij", w_u, lp, rp)


def cp_input_matrices(sd: dict) -> torch.Tensor:
    """CP-factor reconstruction in pixel space, float64. Shape [C, 784, 784].

    Mirrors CPImageModel.decompose (src/models/cp_model.py:147-224).
    """
    A = sd["bilinear.A"].double()  # [256, R]
    B = sd["bilinear.B"].double()
    C = sd["bilinear.C"].double()
    if "bilinear.lambdas" in sd:
        lambdas = sd["bilinear.lambdas"].double()
    else:  # gated
        gates = torch.clamp(torch.sigmoid(sd["bilinear.gate_logits"].double()) * 1.1 - 0.05, 0, 1)
        lambdas = gates * sd["bilinear.scaling_factor"].double()
    w_e = sd["embed.weight"].double()  # [256, 784]
    w_h = sd["head.weight"].double()  # [C, 256]
    a_proj = A.T @ w_e  # [R, 784]
    b_proj = B.T @ w_e
    c_weighted = w_h @ C  # [C, R]
    scale = c_weighted * lambdas  # [C, R]
    return torch.einsum("cr,ri,rj->cij", scale, a_proj, b_proj)


def truncate(vals: torch.Tensor, vecs: torch.Tensor, k: int):
    # Keep top-k eigenpairs plus the full spectrum for energy fractions.
    return vals[:, :k].clone(), vecs[:, :k].clone()


def qfs_grid(ea, eb, k: int) -> np.ndarray:
    vals_a, vecs_a = ea
    vals_b, vecs_b = eb
    out = np.zeros((vals_a.shape[0], vals_b.shape[0]))
    for c in range(vals_a.shape[0]):
        for d in range(vals_b.shape[0]):
            out[c, d] = compute_weighted_similarity(
                vecs_a[c], vecs_b[d], vals_a[c], vals_b[d],
                k=k, method="quadratic_form",
            )
    return out


def direct_qfs_grid(mats_a: torch.Tensor, mats_b: torch.Tensor, k: int) -> np.ndarray:
    """tr(A_k B_k) / (||A_k||_F ||B_k||_F) on rank-k reconstructions, float64."""
    def rank_k(mats):
        vals, vecs = eig_sorted(mats)
        return torch.einsum("ck,cki,ckj->cij", vals[:, :k], vecs[:, :k], vecs[:, :k])

    ra, rb = rank_k(mats_a), rank_k(mats_b)
    cross = torch.einsum("cij,dij->cd", ra, rb)  # tr(A B) for symmetric A, B
    na = ra.flatten(1).norm(dim=1)
    nb = rb.flatten(1).norm(dim=1)
    return (cross / (na.unsqueeze(1) * nb.unsqueeze(0))).numpy()


def diag_offdiag(grid: np.ndarray):
    mask = np.eye(grid.shape[0], dtype=bool)
    return grid[mask], grid[~mask]


def summarize(grids: list, exclude_diag_pairs: bool = False) -> dict:
    """Aggregate diagonal/off-diagonal stats over a list of 10x10 grids."""
    diags = np.concatenate([diag_offdiag(g)[0] for g in grids])
    offs = np.concatenate([diag_offdiag(g)[1] for g in grids])
    n_sep = sum(diag_offdiag(g)[0].min() > diag_offdiag(g)[1].max() for g in grids)
    return {
        "n_grids": len(grids),
        "diag_mean": float(diags.mean()),
        "diag_std": float(diags.std()),
        "diag_min": float(diags.min()),
        "offdiag_mean": float(offs.mean()),
        "offdiag_std": float(offs.std()),
        "offdiag_max": float(offs.max()),
        "n_complete_separation": int(n_sep),
    }


def permutation_test(mean_grid: np.ndarray, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    diag, off = diag_offdiag(mean_grid)
    observed = diag.mean() - off.mean()
    b = 0
    n_cls = mean_grid.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n_cls)
        d, o = diag_offdiag(mean_grid[:, perm])
        if d.mean() - o.mean() >= observed:
            b += 1
    return {
        "statistic": float(observed),
        "n_permutations": n_perm,
        "n_geq": int(b),
        "p_value": float((b + 1) / (n_perm + 1)),
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--spectra-rtol", type=float, default=1e-3,
                    help="rel tol vs float32 checkpoint-stored CP spectra")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    k = args.k
    assertions = {}

    # --- load all bridge checkpoints, recompute input-space eigenpairs ---
    eigs = {}      # (family, config, seed) -> (vals_k, vecs_k)
    energy = {}    # (family, config) -> list of per-class energy fractions
    metrics = {}
    for cfg, seed in itertools.product(DENSE_CONFIGS, SEEDS):
        ck = load_ckpt(MNIST_CKPT / f"mnist_dense_{cfg}_seed{seed}.pt", expect_com=False)
        vals, vecs = eig_sorted(dense_input_matrices(ck["model_state_dict"]))
        frac = (vals[:, :k] ** 2).sum(1) / (vals ** 2).sum(1)
        energy.setdefault(("dense", cfg), []).extend(frac.tolist())
        eigs[("dense", cfg, seed)] = truncate(vals, vecs, k)
        metrics[f"dense_{cfg}_seed{seed}"] = ck["metrics"]["val_acc"]
    for mode, seed in itertools.product(CP_MODES, SEEDS):
        ck = load_ckpt(CP_CKPT / f"mnist_cp_r{CP_RANK}_{mode}_seed{seed}.pt", expect_com=False)
        vals, vecs = eig_sorted(cp_input_matrices(ck["model_state_dict"]))
        frac = (vals[:, :k] ** 2).sum(1) / (vals ** 2).sum(1)
        energy.setdefault(("cp", mode), []).extend(frac.tolist())
        eigs[("cp", mode, seed)] = truncate(vals, vecs, k)
        metrics[f"cp_r{CP_RANK}_{mode}_seed{seed}"] = ck["metrics"]["val_acc"]
        if mode == "lambda" and seed == 42:
            # spectra check vs checkpoint-stored values (what notebook 03b reads)
            stored = ck["eigenvalues"].double()
            rel = ((vals - stored).abs().max() / stored.abs().max()).item()
            assert rel <= args.spectra_rtol, f"CP spectra mismatch: rel {rel:.2e}"
            assertions["cp_spectra_max_rel_diff"] = rel
    assertions["apply_com_false_all_bridge_checkpoints"] = True
    print(f"loaded {len(eigs)} checkpoints; CP spectra rel diff "
          f"{assertions['cp_spectra_max_rel_diff']:.2e}")

    # --- acceptance check: eigenpair estimator == direct rank-k matrices ---
    m42 = dense_input_matrices(
        load_ckpt(MNIST_CKPT / "mnist_dense_wd_seed42.pt", False)["model_state_dict"])
    m43 = dense_input_matrices(
        load_ckpt(MNIST_CKPT / "mnist_dense_wd_seed43.pt", False)["model_state_dict"])
    g_est = qfs_grid(eigs[("dense", "wd", 42)], eigs[("dense", "wd", 43)], k)
    g_dir = direct_qfs_grid(m42, m43, k)
    acc_diff = float(np.abs(g_est - g_dir).max())
    assert acc_diff <= 1e-5, f"acceptance check failed: {acc_diff:.2e} > 1e-5"
    assertions["acceptance_max_abs_diff"] = acc_diff
    print(f"acceptance check (eigenpair vs direct rank-{k}): {acc_diff:.2e}")

    # --- cross-config grids: all 5x5 seed pairs (distinct models) ---
    cross = {}
    dense_x_cp = [(c, m) for c in ["none", "wd", "full"] for m in CP_MODES]
    dense_x_dense = [("none", "wd"), ("wd", "noise"), ("none", "noise"), ("wd", "noise015")]
    for cfg, mode in dense_x_cp:
        grids = [qfs_grid(eigs[("dense", cfg, sa)], eigs[("cp", mode, sb)], k)
                 for sa in SEEDS for sb in SEEDS]
        cross[f"dense_{cfg}_x_cp_{mode}"] = grids
    for ca, cb in dense_x_dense:
        grids = [qfs_grid(eigs[("dense", ca, sa)], eigs[("dense", cb, sb)], k)
                 for sa in SEEDS for sb in SEEDS]
        cross[f"dense_{ca}_x_dense_{cb}"] = grids

    # --- same-config cross-seed anchors: 10 unordered distinct-seed pairs ---
    same = {}
    for fam, cfgs in [("dense", DENSE_CONFIGS), ("cp", CP_MODES)]:
        for cfg in cfgs:
            grids = [qfs_grid(eigs[(fam, cfg, sa)], eigs[(fam, cfg, sb)], k)
                     for sa, sb in itertools.combinations(SEEDS, 2)]
            same[f"{fam}_{cfg}_cross_seed"] = grids

    # --- sanity gate: dense-wd cross-seed complete separation ---
    gate = {}
    for cfg in ["none", "wd", "full"]:
        s = summarize(same[f"dense_{cfg}_cross_seed"])
        gate[f"dense_{cfg}"] = {
            "min_diag": s["diag_min"], "max_offdiag": s["offdiag_max"],
            "passes": s["diag_min"] > s["offdiag_max"],
        }
    if not gate["dense_wd"]["passes"]:
        print("SANITY GATE FAILED (dense-wd cross-seed): "
              f"min diag {gate['dense_wd']['min_diag']:.4f} <= "
              f"max offdiag {gate['dense_wd']['max_offdiag']:.4f}")
    else:
        print(f"sanity gate passed: dense-wd min diag {gate['dense_wd']['min_diag']:.4f} "
              f"> max offdiag {gate['dense_wd']['max_offdiag']:.4f}")

    # --- statistics ---
    stats_out = {}
    for name, grids in cross.items():
        mean_grid = np.mean(grids, axis=0)
        entry = {
            "summary": summarize(grids),
            "mean_grid_diag_mean": float(np.diag(mean_grid).mean()),
            "mean_grid_offdiag_mean": float(diag_offdiag(mean_grid)[1].mean()),
            "permutation": permutation_test(mean_grid, args.n_perm, PERM_SEED),
        }
        stats_out[name] = entry
    conf_name = f"dense_{CONFIRMATORY[0]}_x_cp_{CONFIRMATORY[1]}"
    stats_out[conf_name]["confirmatory"] = True

    # secondary t-test on the confirmatory seed-averaged grid (consistency only)
    from scipy import stats as sstats
    conf_mean = np.mean(cross[conf_name], axis=0)
    d, o = diag_offdiag(conf_mean)
    t_stat, p_t = sstats.ttest_ind(d, o)
    stats_out[conf_name]["secondary_ttest"] = {"t": float(t_stat), "p": float(p_t)}

    # --- estimator discrepancy on the Sec 5 headline pair ---
    ck_m = load_ckpt(XDATA_CKPT / "mnist_dense_full_com_seed42.pt", expect_com=True)
    ck_l = load_ckpt(XDATA_CKPT / "emnist_letters_regularized_seed42.pt", expect_com=True)
    est = qfs_grid(
        (ck_m["eigenvalues"].double(), ck_m["eigenvectors"].double()),
        (ck_l["eigenvalues"].double(), ck_l["eigenvectors"].double()), k)
    direct = direct_qfs_grid(
        dense_input_matrices(ck_m["model_state_dict"]),
        dense_input_matrices(ck_l["model_state_dict"]), k)
    pairs = {"0-O": (0, 14), "1-I": (1, 8), "2-Z": (2, 25), "5-S": (5, 18)}
    discrepancy = {
        "pair": "mnist_dense_full_com_seed42 vs emnist_letters_regularized_seed42",
        "max_abs_diff": float(np.abs(est - direct).max()),
        "mean_abs_diff": float(np.abs(est - direct).mean()),
        "geometric_pairs": {
            lab: {"estimator": float(est[i, j]), "direct": float(direct[i, j])}
            for lab, (i, j) in pairs.items()
        },
    }
    print(f"estimator discrepancy (Sec 5 pair): max {discrepancy['max_abs_diff']:.4f}, "
          f"mean {discrepancy['mean_abs_diff']:.4f}")

    # --- figure: confirmatory seed-averaged heatmap, anchors annotated ---
    anchors = {
        "dense_wd_cross_seed_diag": summarize(same["dense_wd_cross_seed"])["diag_mean"],
        "cp_lambda_cross_seed_diag": summarize(same["cp_lambda_cross_seed"])["diag_mean"],
        "offdiag_floor": stats_out[conf_name]["mean_grid_offdiag_mean"],
        "dense_none_x_wd_diag": stats_out["dense_none_x_dense_wd"]["mean_grid_diag_mean"],
    }
    # scale spans grid AND anchors so the diag-vs-anchor gap is visible
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    vmin = min(float(conf_mean.min()), 0.0)
    im = ax.imshow(conf_mean, cmap="viridis", vmin=vmin, vmax=1)
    for i in range(10):
        for j in range(10):
            v = conf_mean[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v < 0.5 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel(f"$\\Lambda$-CP (R={CP_RANK}) class")
    ax.set_ylabel("dense-wd class")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title("QFS\n($k$=20)", fontsize=7)
    # anchors as marked lines on the colorbar (no baked-in figure title)
    for label, key, va in [("dense$\\times$dense", "dense_wd_cross_seed_diag", "center"),
                           ("CP$\\times$CP", "cp_lambda_cross_seed_diag", "center"),
                           ("off-diag", "offdiag_floor", "top")]:
        v = anchors[key]
        cbar.ax.axhline(v, color="k", lw=0.8)
        cbar.ax.text(2.6, v, f"{label} {v:.2f}", fontsize=6, va=va,
                     transform=cbar.ax.get_yaxis_transform())
    fig.tight_layout()
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    print(f"figure: {FIGURE_PDF}")

    # --- json ---
    out = {
        "k": k,
        "cp_rank": CP_RANK,
        "seeds": SEEDS,
        "confirmatory_pair": conf_name,
        "assertions": assertions,
        "sanity_gate": gate,
        "anchors": anchors,
        "same_config_cross_seed": {n: summarize(g) for n, g in same.items()},
        "cross_config": stats_out,
        "confirmatory_mean_grid": conf_mean.tolist(),
        "energy_fraction_k": {
            f"{fam}_{cfg}": {"mean": float(np.mean(v)), "min": float(np.min(v))}
            for (fam, cfg), v in energy.items()
        },
        "val_acc": metrics,
        "estimator_discrepancy_sec5_pair": discrepancy,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"metrics: {RESULTS_JSON}")

    # console summary of the headline numbers
    s = stats_out[conf_name]
    print(f"\nconfirmatory {conf_name}: diag {s['mean_grid_diag_mean']:.4f}, "
          f"offdiag {s['mean_grid_offdiag_mean']:.4f}, "
          f"perm p = {s['permutation']['p_value']:.2e}, "
          f"separation {s['summary']['n_complete_separation']}/25")
    for n in sorted(stats_out):
        s = stats_out[n]
        print(f"{n:32s} diag {s['mean_grid_diag_mean']:.4f} "
              f"offdiag {s['mean_grid_offdiag_mean']:.4f} "
              f"sep {s['summary']['n_complete_separation']}/{s['summary']['n_grids']} "
              f"p {s['permutation']['p_value']:.2e}")


if __name__ == "__main__":
    main()
