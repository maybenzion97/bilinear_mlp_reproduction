#!/usr/bin/env python3
"""Compare Quadratic Form Similarity (QFS) against linear CKA baselines.

Task: cross-dataset class-pair similarity between per-class interaction
matrices A_c of a CoM-normalized MNIST model and a regularized
EMNIST-Letters model (checkpoints/extension_cross_dataset, seed 42 by
default, k=20 spectral truncation as in the report heatmaps).

Metrics compared on the same 10x26 digit-letter grid:
1. qfs           - tr(A B) / (||A||_F ||B||_F), weight-space, data-free.
2. cka_rank_k    - linear CKA (Kornblith et al. 2019) between rank-k spectral
                   representations R_c = [sqrt(|lambda_i|) (v_i . x)]_{i<=k},
                   an [n, k] matrix over a shared evaluation set
                   X = MNIST test + EMNIST-Letters test (CoM-normalized).
                   Equivalent to kernel alignment of K_c = X |A_c^(k)| X^T,
                   so eigenvalue signs are lost (real feature maps only).
3. cka_scalar    - linear CKA between scalar quadratic outputs
                   y_c(x) = x^T A_c^(k) x ([n, 1]); reduces to squared
                   Pearson correlation and keeps eigenvalue signs.
4. mean_cos      - mean cosine of principal angles between top-k eigenvector
                   subspaces (report cosine baseline).

Separation test matches the report: independent t-test between the 4
geometric pairs (0-O, 1-I, 2-Z, 5-S) and the 8 control pairs, plus
Mann-Whitney U as a rank-based check.

Outputs:
    results/extension_cross_dataset/qfs_vs_cka.json
    results/extension_cross_dataset/qfs_vs_cka_table.tex

Env: conda from environment_cpu.yml (torch, torchvision, scipy).

Usage:
    python scripts/figures/qfs_vs_cka.py [--k 20] [--seeds 42 43 44 45 46]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from src.paths import EXTENSION2_CHECKPOINTS, EXTENSION_CROSS_DATASET_RESULTS
from src.vision.subspace import compute_subspace_overlap, compute_weighted_similarity

# Geometric pairs (digit_idx, letter_idx) and control pairs, as in
# scripts/figures/generate_extension_cross_dataset_figures.py.
SIMILAR_PAIRS = [(0, 14, "0-O"), (1, 8, "1-I"), (2, 25, "2-Z"), (5, 18, "5-S")]
CONTROL_PAIRS = [
    (0, 23, "0-X"), (1, 22, "1-W"), (3, 7, "3-H"), (7, 14, "7-O"),
    (4, 0, "4-A"), (6, 5, "6-F"), (8, 11, "8-L"), (9, 17, "9-R"),
]


def load_eigs(path: Path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return ck["eigenvalues"], ck["eigenvectors"]


def _load_transforms_module():
    # src/data/__init__.py imports the original paper package (needs
    # transformers); load transforms.py standalone to avoid that dependency.
    import importlib.util

    path = PROJECT_ROOT / "src" / "data" / "transforms.py"
    spec = importlib.util.spec_from_file_location("com_transforms", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_eval_set(device: str = "cpu") -> torch.Tensor:
    # Shared evaluation set: MNIST test (10k) + EMNIST-Letters test (20.8k),
    # both CoM-normalized exactly as during training (cached transform).
    # Mirrors src.data.mnist.MNIST and src.data.emnist.EMNISTLetters.
    from torchvision import datasets

    tf = _load_transforms_module()
    root = str(PROJECT_ROOT / "data")

    mnist = datasets.MNIST(root=root, train=False, download=True)
    x_m = mnist.data.float().unsqueeze(1) / 255.0
    x_m = tf.apply_com_to_batch_cached(x_m, "mnist", "test", device=device)

    letters = datasets.EMNIST(root=root, split="letters", train=False, download=True)
    x_l = letters.data.float().transpose(1, 2).unsqueeze(1) / 255.0
    x_l = tf.apply_com_to_batch_cached(x_l, "emnist_letters", "test", device=device)

    x = torch.cat([x_m, x_l], dim=0)  # [n, 1, 28, 28]
    return x.reshape(x.shape[0], -1)  # [n, 784]


def topk_by_magnitude(vals: torch.Tensor, vecs: torch.Tensor, k: int):
    # vals: [C, m], vecs: [C, m, d] -> ([C, k], [C, k, d]) sorted by |lambda|.
    idx = vals.abs().argsort(dim=-1, descending=True)[:, :k]  # [C, k]
    vals_k = torch.gather(vals, 1, idx)
    vecs_k = torch.gather(vecs, 1, idx.unsqueeze(-1).expand(-1, -1, vecs.shape[-1]))
    return vals_k, vecs_k


def spectral_reps(vals: torch.Tensor, vecs: torch.Tensor, x: torch.Tensor, k: int):
    """Rank-k feature maps and scalar quadratic outputs on eval set x.

    Returns:
        feats: [C, n, k] centered, feats[c, i] = sqrt(|lambda_j|) (v_j . x_i)
        y:     [n, C] centered, y[i, c] = x_i^T A_c^(k) x_i
    """
    vals_k, vecs_k = topk_by_magnitude(vals, vecs, k)
    # Projections (v_j . x_i) for all classes at once: [C, n, k]
    proj = torch.einsum("nd,ckd->cnk", x, vecs_k)
    feats = proj * vals_k.abs().sqrt().unsqueeze(1)
    y = (proj.pow(2) * vals_k.unsqueeze(1)).sum(dim=-1).T  # [n, C]
    feats = feats - feats.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    return feats, y


def linear_cka_grid(feats_a: torch.Tensor, feats_b: torch.Tensor) -> np.ndarray:
    """Pairwise linear CKA between centered reps [Ca, n, k] and [Cb, n, k].

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)  (feature form).
    """
    ca, n, k = feats_a.shape
    cb = feats_b.shape[0]
    fa = feats_a.permute(1, 0, 2).reshape(n, ca * k)
    fb = feats_b.permute(1, 0, 2).reshape(n, cb * k)
    cross = (fa.T @ fb).reshape(ca, k, cb, k)
    cross_norm_sq = cross.pow(2).sum(dim=(1, 3))  # [Ca, Cb]
    self_a = torch.stack([(f.T @ f).norm() for f in feats_a])  # [Ca]
    self_b = torch.stack([(f.T @ f).norm() for f in feats_b])  # [Cb]
    cka = cross_norm_sq / (self_a.unsqueeze(1) * self_b.unsqueeze(0))
    return cka.numpy()


def scalar_cka_grid(y_a: torch.Tensor, y_b: torch.Tensor) -> np.ndarray:
    # Centered [n, Ca], [n, Cb] -> squared Pearson correlation grid [Ca, Cb].
    cross = y_a.T @ y_b
    na = y_a.norm(dim=0)
    nb = y_b.norm(dim=0)
    return (cross / (na.unsqueeze(1) * nb.unsqueeze(0))).pow(2).numpy()


def qfs_grid(m_vals, m_vecs, l_vals, l_vecs, k: int) -> np.ndarray:
    out = np.zeros((10, 26))
    for d in range(10):
        for l in range(26):
            out[d, l] = compute_weighted_similarity(
                m_vecs[d], l_vecs[l], m_vals[d], l_vals[l], k=k, method="quadratic_form"
            )
    return out


def mean_cos_grid(m_vals, m_vecs, l_vals, l_vecs, k: int) -> np.ndarray:
    out = np.zeros((10, 26))
    for d in range(10):
        d_idx = m_vals[d].abs().argsort(descending=True)
        d_vecs = m_vecs[d][d_idx]
        for l in range(26):
            l_idx = l_vals[l].abs().argsort(descending=True)
            out[d, l] = compute_subspace_overlap(
                d_vecs[:k], l_vecs[l][l_idx][:k], k=k, method="mean_cos"
            )
    return out


def separation_stats(grid: np.ndarray) -> dict:
    sim = np.array([grid[d, l] for d, l, _ in SIMILAR_PAIRS])
    dis = np.array([grid[d, l] for d, l, _ in CONTROL_PAIRS])
    t_stat, p_t = stats.ttest_ind(sim, dis)
    u_stat, p_u = stats.mannwhitneyu(sim, dis, alternative="greater")
    pooled_sd = np.sqrt(
        ((len(sim) - 1) * sim.var(ddof=1) + (len(dis) - 1) * dis.var(ddof=1))
        / (len(sim) + len(dis) - 2)
    )
    ranks = []
    for d, l, _ in SIMILAR_PAIRS:
        ranks.append(int((grid[d] >= grid[d, l]).sum()))  # 1 = best of 26
    return {
        "similar_mean": float(sim.mean()),
        "similar_std": float(sim.std()),
        "dissimilar_mean": float(dis.mean()),
        "dissimilar_std": float(dis.std()),
        "gap": float(sim.mean() - dis.mean()),
        "cohens_d": float((sim.mean() - dis.mean()) / pooled_sd),
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "u_stat": float(u_stat),
        "p_mannwhitney": float(p_u),
        "expected_pair_ranks": ranks,
        "mean_rank": float(np.mean(ranks)),
    }


def make_latex_table(per_pair: dict, metrics: list) -> str:
    header = {
        "qfs": "QFS",
        "cka_rank_k": "CKA$_{\\mathrm{rank}\\text{-}k}$",
        "cka_scalar": "CKA$_{\\mathrm{scalar}}$",
        "mean_cos": "Cosine",
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{QFS vs.\\ linear CKA baselines on cross-dataset class pairs "
        "(seed 42, $k=20$). CKA uses the union of MNIST and EMNIST-Letters "
        "test sets (CoM-normalized) as evaluation distribution.}",
        "\\label{tab:qfs_vs_cka}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Pair & " + " & ".join(header[m] for m in metrics) + " \\\\",
        "\\midrule",
    ]
    for _, _, label in SIMILAR_PAIRS:
        vals = " & ".join(f"{per_pair[label][m]:.3f}" for m in metrics)
        lines.append(f"{label} (similar) & {vals} \\\\")
    lines.append("\\midrule")
    for _, _, label in CONTROL_PAIRS:
        vals = " & ".join(f"{per_pair[label][m]:.3f}" for m in metrics)
        lines.append(f"{label} (control) & {vals} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    print("Loading shared evaluation set (MNIST test + EMNIST-Letters test, CoM)...")
    x = load_eval_set()
    print(f"  eval set: {tuple(x.shape)}")

    metrics = ["qfs", "cka_rank_k", "cka_scalar", "mean_cos"]
    results = {"k": args.k, "eval_set_size": int(x.shape[0]), "seeds": {}}

    for seed in args.seeds:
        m_path = EXTENSION2_CHECKPOINTS / f"mnist_dense_full_com_seed{seed}.pt"
        l_path = EXTENSION2_CHECKPOINTS / f"emnist_letters_regularized_seed{seed}.pt"
        if not (m_path.exists() and l_path.exists()):
            print(f"seed {seed}: missing checkpoints, skipping")
            continue
        print(f"seed {seed}: computing similarity grids...")
        m_vals, m_vecs = load_eigs(m_path)
        l_vals, l_vecs = load_eigs(l_path)

        m_feats, m_y = spectral_reps(m_vals, m_vecs, x, args.k)
        l_feats, l_y = spectral_reps(l_vals, l_vecs, x, args.k)

        grids = {
            "qfs": qfs_grid(m_vals, m_vecs, l_vals, l_vecs, args.k),
            "cka_rank_k": linear_cka_grid(m_feats, l_feats),
            "cka_scalar": scalar_cka_grid(m_y, l_y),
            "mean_cos": mean_cos_grid(m_vals, m_vecs, l_vals, l_vecs, args.k),
        }

        per_pair = {}
        for d, l, label in SIMILAR_PAIRS + CONTROL_PAIRS:
            per_pair[label] = {m: float(grids[m][d, l]) for m in metrics}

        results["seeds"][seed] = {
            "per_pair": per_pair,
            "separation": {m: separation_stats(grids[m]) for m in metrics},
        }

    # Cross-seed aggregation of group means and gaps.
    agg = {}
    for m in metrics:
        sim_means = [results["seeds"][s]["separation"][m]["similar_mean"] for s in results["seeds"]]
        dis_means = [results["seeds"][s]["separation"][m]["dissimilar_mean"] for s in results["seeds"]]
        mean_ranks = [results["seeds"][s]["separation"][m]["mean_rank"] for s in results["seeds"]]
        agg[m] = {
            "similar_mean_over_seeds": float(np.mean(sim_means)),
            "similar_std_over_seeds": float(np.std(sim_means)),
            "dissimilar_mean_over_seeds": float(np.mean(dis_means)),
            "dissimilar_std_over_seeds": float(np.std(dis_means)),
            "gap_over_seeds": float(np.mean(sim_means) - np.mean(dis_means)),
            "mean_rank_over_seeds": float(np.mean(mean_ranks)),
        }
    results["aggregate"] = agg

    EXTENSION_CROSS_DATASET_RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = EXTENSION_CROSS_DATASET_RESULTS / "qfs_vs_cka.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")

    primary = args.seeds[0]
    table = make_latex_table(results["seeds"][primary]["per_pair"], metrics)
    out_tex = EXTENSION_CROSS_DATASET_RESULTS / "qfs_vs_cka_table.tex"
    out_tex.write_text(table + "\n")
    print(f"Saved: {out_tex}")

    print(f"\n=== Per-pair values (seed {primary}, k={args.k}) ===")
    hdr = "pair       " + "".join(f"{m:>12}" for m in metrics)
    print(hdr)
    for _, _, label in SIMILAR_PAIRS + CONTROL_PAIRS:
        row = results["seeds"][primary]["per_pair"][label]
        print(f"{label:<11}" + "".join(f"{row[m]:>12.3f}" for m in metrics))

    print(f"\n=== Separation statistics (seed {primary}) ===")
    for m in metrics:
        s = results["seeds"][primary]["separation"][m]
        print(
            f"{m:<12} similar={s['similar_mean']:.3f}+/-{s['similar_std']:.3f} "
            f"dissimilar={s['dissimilar_mean']:.3f}+/-{s['dissimilar_std']:.3f} "
            f"gap={s['gap']:.3f} d={s['cohens_d']:.2f} "
            f"t={s['t_stat']:.2f} p_t={s['p_ttest']:.2e} "
            f"U={s['u_stat']:.0f} p_U={s['p_mannwhitney']:.2e} "
            f"ranks={s['expected_pair_ranks']} mean_rank={s['mean_rank']:.1f}"
        )

    print("\n=== LaTeX table ===")
    print(table)


if __name__ == "__main__":
    main()
