#!/usr/bin/env python3
"""
Tucker vs CP identifiability demo on the trained MNIST interaction tensor.

Builds the third-order interaction tensor B[c,i,j] (10 x 784 x 784) from the
dense checkpoint (config "full", seed 42), following Model.decompose() in
bilinear-decomposition-main/image/model.py:

    B[c,i,j] = sum_h w_u[c,h] * (w_l @ w_e)[h,i] * (w_r @ w_e)[h,j]
    B <- 0.5 * (B + B^T)   (symmetrize over i,j)

Then:
1. Rank-R CP (ALS) and Tucker/HOOI with matched parameter count.
2. Tucker non-identifiability: rotate input-mode factor U -> U Q and
   counter-rotate the core G -> G x_1 Q^T; reconstruction is unchanged
   but the per-column "features" are arbitrary.
3. CP stability: ALS from 3 random inits recovers the same rank-1 factors
   up to permutation/sign (greedy matching, triple-cosine congruence).

Input: checkpoints/vision/mnist/mnist_dense_full_seed42.pt (no new training).

Outputs:
    results/extension2/tucker_vs_cp.json
    bilinear_mlp_reproduction_report/figures/extension_cp/tucker_vs_cp_identifiability.pdf

Env: conda from environment_cpu.yml (tensorly).

Usage:
    python scripts/figures/tucker_vs_cp.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.tenalg import mode_dot

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT = PROJECT_ROOT / "checkpoints/vision/mnist/mnist_dense_full_seed42.pt"
RESULTS_JSON = PROJECT_ROOT / "results/extension2/tucker_vs_cp.json"
FIGURE_PDF = (
    PROJECT_ROOT.parent
    / "bilinear_mlp_reproduction_report/figures/extension_cp/tucker_vs_cp_identifiability.pdf"
)

CP_RANK = 25
CP_SEEDS = [42, 43, 44]  # 3 independent ALS inits
TUCKER_INPUT_RANK = 22  # core (10, 22, 22) matches CP param count (see below)
ROTATION_SEED = 0
N_SHOW = 5  # components per row in the figure


def build_interaction_tensor(ckpt_path: Path) -> np.ndarray:
    """B[c,i,j] in pixel space, symmetrized over (i,j). Shape (10, 784, 784)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    w_e = sd["embed.weight"].double()  # [256, 784]
    w_lr = sd["blocks.0.weight"].double()  # [512, 256] = stacked (w_l, w_r)
    w_u = sd["head.weight"].double()  # [10, 256]
    w_l, w_r = w_lr.chunk(2, dim=0)  # each [256, 256]

    lp = w_l @ w_e  # [256, 784] left projection to pixel space
    rp = w_r @ w_e  # [256, 784]
    b = torch.einsum("ch,hi,hj->cij", w_u, lp, rp)
    b = 0.5 * (b + b.transpose(1, 2))
    return b.numpy()


def rel_error(t: np.ndarray, t_hat: np.ndarray) -> float:
    return float(np.linalg.norm(t - t_hat) / np.linalg.norm(t))


def cp_params(shape, rank: int) -> int:
    return rank * sum(shape) + rank  # factors + weights


def tucker_params(shape, ranks) -> int:
    core = int(np.prod(ranks))
    factors = sum(s * r for s, r in zip(shape, ranks))
    return core + factors


def greedy_match(score: np.ndarray):
    """Greedy max matching on a square score matrix. Returns (rows, cols)."""
    s = score.copy()
    rows, cols = [], []
    for _ in range(s.shape[0]):
        p, q = np.unravel_index(np.argmax(s), s.shape)
        rows.append(int(p))
        cols.append(int(q))
        s[p, :] = -np.inf
        s[:, q] = -np.inf
    return np.array(rows), np.array(cols)


def factor_congruence(fa, fb):
    """Per-pair triple-cosine congruence between two CP factor sets.

    fa, fb: lists of 3 factor matrices [dim, R], columns unit-normalized.
    B is symmetric in modes 1,2, so a component (a,b) in one run can appear
    as (b,a) in another; take the max over the mode-(1,2) swap.
    Returns [R, R] matrix of |cos_0| * max(|cos_1 cos_2|, |cos_1x2 cos_2x1|).
    """
    c0 = np.abs(fa[0].T @ fb[0])
    c11 = np.abs(fa[1].T @ fb[1])
    c22 = np.abs(fa[2].T @ fb[2])
    c12 = np.abs(fa[1].T @ fb[2])
    c21 = np.abs(fa[2].T @ fb[1])
    return c0 * np.maximum(c11 * c22, c12 * c21)


def normalize_cols(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=0, keepdims=True)


def run_cp(t: np.ndarray, rank: int, seed: int):
    (weights, factors), errors = parafac(
        tl.tensor(t),
        rank=rank,
        n_iter_max=4000,
        tol=1e-12,
        init="random",
        random_state=seed,
        normalize_factors=True,
        linesearch=True,
        return_errors=True,
    )
    print(f"  ALS iterations: {len(errors)}")
    t_hat = tl.cp_to_tensor((weights, factors))
    return np.asarray(weights), [np.asarray(f) for f in factors], rel_error(t, t_hat)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--cp-rank", type=int, default=CP_RANK)
    parser.add_argument("--tucker-rank", type=int, default=TUCKER_INPUT_RANK)
    parser.add_argument("--cache", type=Path, default=None,
                        help="npz cache for decompositions (skips ALS if present)")
    args = parser.parse_args()

    tl.set_backend("numpy")
    np.random.seed(0)

    t = build_interaction_tensor(CHECKPOINT)
    shape = t.shape
    print(f"interaction tensor: {shape}, fro norm {np.linalg.norm(t):.4f}")

    # --- CP: 3 ALS restarts, same rank ---
    cache = {}
    if args.cache is not None and args.cache.exists():
        cache = dict(np.load(args.cache))
        print(f"loaded cache: {args.cache}")

    cp_runs = []
    for seed in CP_SEEDS:
        key = f"cp{args.cp_rank}_s{seed}"
        if f"{key}_w" in cache:
            w = cache[f"{key}_w"]
            f = [cache[f"{key}_f{m}"] for m in range(3)]
            err = float(cache[f"{key}_err"])
        else:
            w, f, err = run_cp(t, args.cp_rank, seed)
            # sort components by weight for deterministic reporting
            order = np.argsort(-np.abs(w))
            w, f = w[order], [fac[:, order] for fac in f]
            cache[f"{key}_w"] = w
            for m in range(3):
                cache[f"{key}_f{m}"] = f[m]
            cache[f"{key}_err"] = np.array(err)
        cp_runs.append({"seed": seed, "weights": w, "factors": f, "rel_err": err})
        print(f"CP  R={args.cp_rank} seed={seed}: rel err {err:.4f}")
    if args.cache is not None:
        np.savez(args.cache, **cache)

    # --- CP stability across restarts: greedy matching on triple congruence ---
    pair_scores = {}
    matches = {}
    for a in range(len(cp_runs)):
        for b in range(a + 1, len(cp_runs)):
            fa = [normalize_cols(f) for f in cp_runs[a]["factors"]]
            fb = [normalize_cols(f) for f in cp_runs[b]["factors"]]
            score = factor_congruence(fa, fb)
            rows, cols = greedy_match(score)
            matched = score[rows, cols]
            # headline metric: greedy match directly on pixel-mode |cos|
            # (swap-aware: mode-1 factor may appear as mode-2 in another run)
            cos_mat = np.maximum(np.abs(fa[1].T @ fb[1]), np.abs(fa[1].T @ fb[2]))
            crows, ccols = greedy_match(cos_mat)
            cos1 = cos_mat[crows, ccols]
            # stratify by component importance: reorder matched pairs so index p
            # is run-a component p (components are weight-sorted, 0 = largest)
            wa = np.abs(cp_runs[a]["weights"])
            matched_sorted = matched[np.argsort(rows)]
            cos1_sorted = cos1[np.argsort(crows)]
            k = 10
            w_norm = wa / wa.sum()
            key = f"seed{CP_SEEDS[a]}_vs_seed{CP_SEEDS[b]}"
            pair_scores[key] = {
                "mean_congruence": float(matched.mean()),
                "min_congruence": float(matched.min()),
                "mean_abs_cos_mode1": float(cos1.mean()),
                f"mean_congruence_top{k}": float(matched_sorted[:k].mean()),
                f"mean_abs_cos_mode1_top{k}": float(cos1_sorted[:k].mean()),
                "weighted_congruence": float((w_norm * matched_sorted).sum()),
            }
            matches[(a, b)] = (rows, cols)
            print(f"CP match {key}: mean congruence {matched.mean():.4f}, "
                  f"top{k} {matched_sorted[:k].mean():.4f}, "
                  f"weighted {(w_norm * matched_sorted).sum():.4f}, "
                  f"mean |cos| mode-1 {cos1.mean():.4f}")

    # --- Tucker/HOOI with matched parameter count ---
    tucker_ranks = (shape[0], args.tucker_rank, args.tucker_rank)
    core, factors = tucker(
        tl.tensor(t), rank=tucker_ranks, init="svd", n_iter_max=200, tol=1e-10,
        random_state=42,
    )
    t_tucker = tl.tucker_to_tensor((core, factors))
    tucker_err = rel_error(t, t_tucker)
    core, factors = np.asarray(core), [np.asarray(f) for f in factors]
    print(f"Tucker {tucker_ranks}: rel err {tucker_err:.4f}")

    n_cp = cp_params(shape, args.cp_rank)
    n_tk = tucker_params(shape, tucker_ranks)
    n_dense = int(np.prod(shape))
    print(f"params: CP {n_cp}, Tucker {n_tk}, dense {n_dense}")

    # --- Tucker non-identifiability: rotate mode-1 factor, counter-rotate core ---
    rng = np.random.default_rng(ROTATION_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((args.tucker_rank, args.tucker_rank)))
    u1_rot = factors[1] @ q
    core_rot = mode_dot(core, q.T, mode=1)  # G x_1 Q^T
    t_rot = tl.tucker_to_tensor((core_rot, [factors[0], u1_rot, factors[2]]))
    max_abs_diff = float(np.max(np.abs(t_rot - t_tucker)))
    print(f"rotated Tucker reconstruction max abs diff: {max_abs_diff:.3e}")

    # chance-level contrast: greedy-matched |cos| between original and rotated
    # Tucker mode-1 columns (same tensor, arbitrary basis)
    tk_cos_mat = np.abs(normalize_cols(factors[1]).T @ normalize_cols(u1_rot))
    tk_rows, tk_cols = greedy_match(tk_cos_mat)
    tk_rot_cos = float(tk_cos_mat[tk_rows, tk_cols].mean())
    print(f"Tucker matched |cos| original vs rotated: {tk_rot_cos:.4f}")

    # --- figure ---
    # Tucker: leading mode-1 columns by core slice energy, before/after rotation.
    # core.shape = (10, r, r); mode-1 slice energy per column j: ||core[:, j, :]||
    energy = np.linalg.norm(core, axis=(0, 2))
    top_tk = np.argsort(-energy)[:N_SHOW]
    energy_rot = np.linalg.norm(core_rot, axis=(0, 2))
    top_tk_rot = np.argsort(-energy_rot)[:N_SHOW]

    # CP: second run reordered by match to first; show the N_SHOW best-matched
    # pairs (weight-sorted), since near-degenerate components match poorly.
    rows01, cols01 = matches[(0, 1)]
    inv = np.empty_like(cols01)
    inv[rows01] = cols01  # component p of run 0 -> inv[p] of run 1
    f1_run0 = normalize_cols(cp_runs[0]["factors"][1])
    cand1 = normalize_cols(cp_runs[1]["factors"][1])[:, inv]
    cand2 = normalize_cols(cp_runs[1]["factors"][2])[:, inv]  # mode-(1,2) swap
    pick2 = np.abs(np.sum(f1_run0 * cand2, axis=0)) > np.abs(np.sum(f1_run0 * cand1, axis=0))
    f1_run1 = np.where(pick2, cand2, cand1)
    # sign-align run 1 to run 0 for display
    pair_cos = np.abs(np.sum(f1_run0 * f1_run1, axis=0))
    sign = np.sign(np.sum(f1_run0 * f1_run1, axis=0))
    f1_run1 = f1_run1 * sign
    show_idx = np.sort(np.argsort(-pair_cos)[:N_SHOW])  # best pairs, weight order

    panels = [
        ("Tucker $U_1$ (HOOI)", factors[1][:, top_tk], None),
        ("Tucker $U_1 Q$ (rotated)", u1_rot[:, top_tk_rot], None),
        ("CP run 1", f1_run0[:, show_idx], None),
        ("CP run 2 (matched)", f1_run1[:, show_idx], pair_cos[show_idx]),
    ]

    fig, axes = plt.subplots(len(panels), N_SHOW, figsize=(1.35 * N_SHOW, 1.5 * len(panels)))
    for r, (label, mat, annot) in enumerate(panels):
        for c in range(N_SHOW):
            ax = axes[r, c]
            img = mat[:, c].reshape(28, 28)
            v = np.abs(img).max()
            ax.imshow(img, cmap="gray", vmin=-v, vmax=v)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"comp {c + 1}", fontsize=7)
            if annot is not None:
                ax.set_xlabel(f"$|\\cos| = {annot[c]:.2f}$", fontsize=6)
        axes[r, 0].set_ylabel(label, fontsize=7)
    fig.tight_layout(pad=0.4)
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    print(f"figure: {FIGURE_PDF}")

    # --- metrics json ---
    metrics = {
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "tensor_shape": list(shape),
        "tensor_fro_norm": float(np.linalg.norm(t)),
        "dense_params": n_dense,
        "cp": {
            "rank": args.cp_rank,
            "params": n_cp,
            "seeds": CP_SEEDS,
            "rel_err_per_seed": {str(r["seed"]): r["rel_err"] for r in cp_runs},
            "rel_err_mean": float(np.mean([r["rel_err"] for r in cp_runs])),
            "stability": pair_scores,
            "stability_mean_congruence": float(
                np.mean([v["mean_congruence"] for v in pair_scores.values()])
            ),
            "stability_mean_abs_cos_mode1": float(
                np.mean([v["mean_abs_cos_mode1"] for v in pair_scores.values()])
            ),
            "stability_mean_congruence_top10": float(
                np.mean([v["mean_congruence_top10"] for v in pair_scores.values()])
            ),
            "stability_weighted_congruence": float(
                np.mean([v["weighted_congruence"] for v in pair_scores.values()])
            ),
        },
        "tucker": {
            "ranks": list(tucker_ranks),
            "params": n_tk,
            "rel_err": tucker_err,
            "rotation_seed": ROTATION_SEED,
            "rotated_reconstruction_max_abs_diff": max_abs_diff,
            "rotated_vs_original_matched_mean_abs_cos": tk_rot_cos,
        },
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"metrics: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
