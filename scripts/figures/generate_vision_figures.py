#!/usr/bin/env python3
"""
Generate vision figures (Paper Section 4 + appendix items).

This script is the single entrypoint for all vision figures. It only uses
existing checkpoints + MNIST data (no new training).

Usage:
    python scripts/figures/generate_vision_figures.py
    python scripts/figures/generate_vision_figures.py --sections regularization
    ./scripts/train/run_vision.sh figures  # Preferred wrapper
"""

import sys
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import warnings

# Add project + original code paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from src.paths import (
    MNIST_CHECKPOINTS,
    FASHION_CHECKPOINTS,
    NOISE_SWEEP_CHECKPOINTS,
    SIZE_SWEEP_CHECKPOINTS,
    CHALLENGE_CHECKPOINTS,
    VISION_FIGURES,
    EXTENSION2_CHECKPOINTS,
)
from src.vision.spectral import (
    load_checkpoint_eigenvalues,
    load_all_checkpoints,
    aggregate_by_config,
    compute_rank_ratio,
    effective_rank,
)
from src.vision.context import VisionContext
from src.plot_utils.style import COLORS
from src.plot_utils.eigenspectrum import (
    plot_eigenspectrum_comparison,
    plot_eigenspectrum_per_class,
    plot_eigenvalue_decay,
)
from src.plot_utils.eigenvectors import plot_eigenvectors_grid
from src.plot_utils.ablation import (
    plot_ablation_bars,
    plot_accuracy_vs_effective_rank,
    plot_metric_comparison,
)
from src.plot_utils.style import set_publication_style
from src.artifact_loader import ensure_artifacts

# Reduce noisy, non-actionable warnings in local environments.
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension:*",
)


@dataclass(frozen=True)
class Dirs:
    vision_mnist_ckpts: Path
    vision_fashion_ckpts: Path
    noise_sweep_ckpts: Path
    size_sweep_ckpts: Path
    challenge_ckpts: Path
    figure_out: Path


def get_dirs() -> Dirs:
    return Dirs(
        vision_mnist_ckpts=MNIST_CHECKPOINTS,
        vision_fashion_ckpts=FASHION_CHECKPOINTS,
        noise_sweep_ckpts=NOISE_SWEEP_CHECKPOINTS,
        size_sweep_ckpts=SIZE_SWEEP_CHECKPOINTS,
        challenge_ckpts=CHALLENGE_CHECKPOINTS,
        figure_out=VISION_FIGURES,
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    """Save figure to the specified path."""
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved: {out_path}")


def generate_regularization_section(d: Dirs) -> None:
    """Paper Section 4: regularization/eigenspectrum/eigenvectors/ablation/tradeoff."""
    print("\n=== Vision / Regularization ===")

    # Load MNIST + Fashion checkpoints into a dataframe for ablations/tradeoff plots.
    mnist_df = load_all_checkpoints(d.vision_mnist_ckpts, dataset="mnist")
    fashion_df = load_all_checkpoints(d.vision_fashion_ckpts, dataset="fashion")
    mnist_agg = aggregate_by_config(mnist_df)
    fashion_agg = aggregate_by_config(fashion_df)

    # Eigenspectrum comparison (seed42 reference for each config)
    eigenvalues_dict: Dict[str, torch.Tensor] = {}
    for config in ["none", "noise", "wd", "full"]:
        path = d.vision_mnist_ckpts / f"mnist_dense_{config}_seed42.pt"
        vals, _ = load_checkpoint_eigenvalues(str(path))
        eigenvalues_dict[config] = vals

    fig = plot_eigenspectrum_comparison(
        eigenvalues_dict,
        title="MNIST: Eigenspectrum by Regularization Type (shaded = 90% CI across classes)",
        top_k=100,
    )
    _save_figure(fig, d.figure_out / "eigenspectrum_comparison.pdf")
    plt.close(fig)

    fig = plot_eigenvalue_decay(
        eigenvalues_dict,
        top_k=30,
        title="MNIST: Normalized Eigenvalue Decay",
    )
    _save_figure(fig, d.figure_out / "eigenvalue_decay.pdf")
    plt.close(fig)

    vals_full, _ = load_checkpoint_eigenvalues(str(d.vision_mnist_ckpts / "mnist_dense_full_seed42.pt"))
    fig = plot_eigenspectrum_per_class(
        vals_full,
        title="MNIST (Full Reg): Eigenspectrum Across Digit Classes",
        combined=True,
    )
    _save_figure(fig, d.figure_out / "eigenspectrum_per_class.pdf")
    plt.close(fig)

    # Eigenvector grids
    vals_none, vecs_none = load_checkpoint_eigenvalues(str(d.vision_mnist_ckpts / "mnist_dense_none_seed42.pt"))
    vals_reg, vecs_reg = load_checkpoint_eigenvalues(str(d.vision_mnist_ckpts / "mnist_dense_full_seed42.pt"))
    vals_noise, vecs_noise = load_checkpoint_eigenvalues(str(d.vision_mnist_ckpts / "mnist_dense_noise_seed42.pt"))

    fig = plot_eigenvectors_grid(
        vecs_none,
        vals_none,
        n_top=5,
        title="MNIST (No Reg): Top Eigenvectors",
        show_both_signs=True,
    )
    _save_figure(fig, d.figure_out / "eigenvectors_noreg.pdf")
    plt.close(fig)

    fig = plot_eigenvectors_grid(
        vecs_reg,
        vals_reg,
        n_top=5,
        title="MNIST (Full Regularization: sigma=0.5, λ=1.0): Top Eigenvectors",
        show_both_signs=True,
    )
    _save_figure(fig, d.figure_out / "eigenvectors_reg.pdf")
    plt.close(fig)

    # Noise-only eigenvectors (requested)
    fig = plot_eigenvectors_grid(
        vecs_noise,
        vals_noise,
        n_top=5,
        title="MNIST (Noise only: sigma=0.5, λ=0.0): Top Eigenvectors",
        show_both_signs=True,
    )
    _save_figure(fig, d.figure_out / "eigenvectors_noise.pdf")
    plt.close(fig)

    # Single-digit comparison (digit 0) across three regularization conditions
    # For presentation slide 4: cleaner comparison showing noise is key
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    digit = 0  # Focus on digit "0"
    n_show = 5  # Show top 5 eigenvectors per condition
    
    conditions = [
        ("No Regularization", vecs_none, vals_none),
        ("Noise Only (σ=0.5)", vecs_noise, vals_noise),
        ("Full Reg (σ=0.5, λ=1.0)", vecs_reg, vals_reg),
    ]
    
    for ax_idx, (label, vecs, vals) in enumerate(conditions):
        # Get top eigenvectors by absolute magnitude for this digit
        abs_vals = vals[digit].abs()
        sorted_indices = abs_vals.argsort(descending=True)[:n_show]
        
        # Create a small grid showing top-5 eigenvectors horizontally
        combined_img = []
        for i, idx in enumerate(sorted_indices):
            vec = vecs[digit, idx].numpy().reshape(28, 28)
            # Normalize each eigenvector
            vmax = np.abs(vec).max()
            if vmax > 1e-10:
                vec = vec / vmax
            combined_img.append(vec)
        
        # Stack horizontally with small gaps
        gap = np.ones((28, 2)) * 0  # Small white gap
        full_img = combined_img[0]
        for vec in combined_img[1:]:
            full_img = np.concatenate([full_img, gap, vec], axis=1)
        
        vmax = 1.0
        axes[ax_idx].imshow(full_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title(label, fontsize=11, fontweight='bold' if 'Noise Only' in label else 'normal')
    
    fig.suptitle(f"Digit {digit}: Top-5 Eigenvectors Across Regularization Conditions", fontsize=12, y=1.02)
    plt.tight_layout()
    _save_figure(fig, d.figure_out / "eigenvectors_single_digit_comparison.pdf")
    plt.close(fig)

    # Generate separate figures for V2 (with reg) and V3 (no reg) with eigenvalue labels
    # These show only digit 0's top-5 eigenvectors with λ values above each
    def _make_single_digit_eigenvectors_with_labels(vecs, vals, digit, n_show, title, out_name):
        """Create figure showing top eigenvectors for a single digit with eigenvalue labels."""
        fig, axes = plt.subplots(1, n_show, figsize=(n_show * 1.5, 2.2))
        
        # Get top eigenvectors by absolute magnitude
        abs_vals = vals[digit].abs()
        sorted_indices = abs_vals.argsort(descending=True)[:n_show]
        
        for i, idx in enumerate(sorted_indices):
            vec = vecs[digit, idx].numpy().reshape(28, 28)
            eigenvalue = vals[digit, idx].item()
            
            # Normalize eigenvector for display
            vmax = np.abs(vec).max()
            if vmax > 1e-10:
                vec = vec / vmax
            
            axes[i].imshow(vec, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[i].axis("off")
            # Show eigenvalue above
            axes[i].set_title(f"λ={eigenvalue:.1f}", fontsize=9)
        
        fig.suptitle(title, fontsize=11, y=1.0)
        plt.tight_layout()
        _save_figure(fig, d.figure_out / out_name)
        plt.close(fig)
    
    # V2: With regularization (noise + WD)
    _make_single_digit_eigenvectors_with_labels(
        vecs_reg, vals_reg, digit=0, n_show=5,
        title="Digit 0: Top-5 Eigenvectors (Full Reg)",
        out_name="eigenvectors_digit0_reg.pdf"
    )
    
    # V2 additional: Noise only
    _make_single_digit_eigenvectors_with_labels(
        vecs_noise, vals_noise, digit=0, n_show=5,
        title="Digit 0: Top-5 Eigenvectors (Noise Only)",
        out_name="eigenvectors_digit0_noise.pdf"
    )
    
    # V3: No regularization
    _make_single_digit_eigenvectors_with_labels(
        vecs_none, vals_none, digit=0, n_show=5,
        title="Digit 0: Top-5 Eigenvectors (No Reg)",
        out_name="eigenvectors_digit0_noreg.pdf"
    )

    # Fashion eigenvectors: add noise-only too
    fashion_noise_ckpt = d.vision_fashion_ckpts / "fashion_dense_noise_seed42.pt"
    if fashion_noise_ckpt.exists():
        vals_fashion_noise, vecs_fashion_noise = load_checkpoint_eigenvalues(str(fashion_noise_ckpt))
        fashion_classes = [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
        ]
        fig = plot_eigenvectors_grid(
            vecs_fashion_noise,
            vals_fashion_noise,
            n_top=5,
            title="Fashion-MNIST (Noise only: sigma=0.5, λ=0.0): Top Eigenvectors",
            class_names=fashion_classes,
            show_both_signs=True,
        )
        _save_figure(fig, d.figure_out / "fashion_eigenvectors_noise.pdf")
        plt.close(fig)
    else:
        print(f"WARNING: {fashion_noise_ckpt} missing; skipping fashion noise-only eigenvectors.")

    # Ablations
    fig = plot_ablation_bars(
        mnist_agg,
        metrics=["accuracy", "effective_rank"],
        title="MNIST: Ablation Study",
    )
    _save_figure(fig, d.figure_out / "mnist_ablation.pdf")
    plt.close(fig)

    fig = plot_ablation_bars(
        fashion_agg,
        metrics=["accuracy", "effective_rank"],
        title="Fashion-MNIST: Ablation Study",
    )
    _save_figure(fig, d.figure_out / "fashion_ablation.pdf")
    plt.close(fig)

    # Trade-off
    fig = plot_accuracy_vs_effective_rank(
        mnist_df,
        title="MNIST: Accuracy vs Interpretability",
    )
    _save_figure(fig, d.figure_out / "accuracy_vs_effrank_mnist.pdf")
    plt.close(fig)

    fig = plot_accuracy_vs_effective_rank(
        fashion_df,
        title="Fashion-MNIST: Accuracy vs Interpretability",
    )
    _save_figure(fig, d.figure_out / "accuracy_vs_effrank_fashion.pdf")
    plt.close(fig)

    # Cross-dataset
    fig = plot_metric_comparison(
        [mnist_agg, fashion_agg],
        ["MNIST", "Fashion-MNIST"],
        metric="effective_rank",
        title="Effective Rank: MNIST vs Fashion-MNIST",
    )
    _save_figure(fig, d.figure_out / "cross_dataset_effrank.pdf")
    plt.close(fig)

    fig = plot_metric_comparison(
        [mnist_agg, fashion_agg],
        ["MNIST", "Fashion-MNIST"],
        metric="accuracy",
        title="Accuracy: MNIST vs Fashion-MNIST",
    )
    _save_figure(fig, d.figure_out / "cross_dataset_accuracy.pdf")
    plt.close(fig)

    # Gate check printout
    mnist_ratio = compute_rank_ratio(mnist_df, "none", "full")
    mnist_wd_ratio = compute_rank_ratio(mnist_df, "none", "wd")
    print("\nGate check:")
    print(f"  effective rank ratio full/none: {mnist_ratio:.3f}")
    print(f"  effective rank ratio wd/none:   {mnist_wd_ratio:.3f}")

    # ---------- Paper Figure 4: noise sweep (if checkpoints exist) ----------
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    target_digit = 5
    found = []
    eigvecs = {}
    eigvals = {}
    accs = {}
    effranks = {}

    for nl in noise_levels:
        ckpt_path = d.noise_sweep_ckpts / f"mnist_noise_{nl}_seed42.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vals = ckpt["eigenvalues"]
        vecs = ckpt["eigenvectors"]
        found.append(nl)
        eigvecs[nl] = vecs
        eigvals[nl] = vals
        accs[nl] = float(ckpt["metrics"]["val_acc"])
        effranks[nl] = float(effective_rank(vals).mean().item())

    if len(found) >= 2:
        # (a) Top eigenvector for digit 5 across noise levels
        fig, axes = plt.subplots(1, len(found), figsize=(2.4 * len(found), 2.8))
        if len(found) == 1:
            axes = [axes]
        for i, nl in enumerate(sorted(found)):
            ax = axes[i]
            vals = eigvals[nl][target_digit]
            vecs = eigvecs[nl][target_digit]
            # top eigenvector by |eigenvalue|
            idx = int(vals.abs().argmax().item())
            img = vecs[idx].reshape(28, 28).numpy()
            vmax = float(np.abs(img).max() or 1.0)
            ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"sigma={nl}\nacc={accs[nl]*100:.1f}%", fontsize=9)
            ax.axis("off")
        # Avoid tight_layout warnings with grids of image axes; bbox_inches='tight' handles cropping.
        _save_figure(fig, d.figure_out / "figure_4_noise_eigenvectors.pdf")
        plt.close(fig)

        # (b) Effective rank vs noise
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = sorted(found)
        ys = [effranks[nl] for nl in xs]
        ax.plot(xs, ys, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Input noise std (sigma)")
        ax.set_ylabel("Effective rank")
        ax.set_title("Effect of input noise on effective rank")
        ax.grid(True, alpha=0.3)
        # Avoid tight_layout warnings with grids of image axes; bbox_inches='tight' handles cropping.
        _save_figure(fig, d.figure_out / "figure_4_noise_vs_rank.pdf")
        plt.close(fig)

        # (c) Accuracy vs noise (useful but not strictly required)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [accs[nl] * 100 for nl in xs], "s-", linewidth=2, markersize=6)
        ax.set_xlabel("Input noise std (sigma)")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title("Effect of input noise on accuracy")
        ax.grid(True, alpha=0.3)
        # Avoid tight_layout warnings with grids of image axes; bbox_inches='tight' handles cropping.
        _save_figure(fig, d.figure_out / "figure_4_noise_vs_accuracy.pdf")
        plt.close(fig)
    else:
        print("INFO: noise sweep checkpoints not found; skipping Figure 4 generation.")


def generate_truncation_similarity_section(d: Dirs) -> None:
    """Paper Figure 5 + appendix size/truncation/similarity extensions."""
    print("\n=== Vision / Truncation & similarity ===")

    from itertools import combinations, product
    from scipy import stats
    import pandas as pd

    from image.datasets import MNIST
    from src.vision.truncation import (
        load_size_sweep_checkpoints,
        compute_similarity_by_rank,
        compute_all_truncation_curves,
        aggregate_truncation_curves,
        compute_eigenvector_similarity,
    )

    sizes = [30, 50, 100, 300, 500, 1000]
    seeds = [42, 43, 44, 45, 46]
    ckpts_by_size = load_size_sweep_checkpoints(str(d.size_sweep_ckpts), sizes=sizes, seeds=seeds)

    # ---------- Figure 5A: similarity across ranks ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.viridis
    colors = {s: cmap(i / (len(sizes) - 1)) for i, s in enumerate(sizes)}

    max_rank = 20
    for size in sizes:
        ckpts = ckpts_by_size.get(size, [])
        if len(ckpts) < 2:
            continue

        sim_mean, sim_std = compute_similarity_by_rank(ckpts, max_rank=max_rank)
        ranks = np.arange(max_rank)

        n_pairs = len(list(combinations(range(len(ckpts)), 2)))
        n = max(1, n_pairs * 10)
        sem = sim_std / np.sqrt(n)
        t_val = stats.t.ppf(0.95, n - 1)  # 90% CI
        ci_low = sim_mean - t_val * sem
        ci_high = sim_mean + t_val * sem

        ax.plot(ranks, sim_mean, "-", color=colors[size], label=f"{size}", linewidth=2)
        ax.fill_between(ranks, ci_low, ci_high, color=colors[size], alpha=0.2)

    ax.set_xlabel("Eigenvector rank", fontsize=12)
    ax.set_ylabel("Absolute Cosine similarity", fontsize=12)
    ax.set_title("Eigenvector Similarity Across Seeds", fontsize=13)
    ax.legend(title="Model Size", fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_rank)
    ax.set_ylim(0.0, 1.05)  # IMPORTANT: do not truncate at 0.4
    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "figure_5a_similarity.pdf")
    plt.close(fig)

    # ---------- Figure 5B: truncation error (log scale) ----------
    test_data = MNIST(train=False, device="cpu")
    test_x = test_data.x.flatten(start_dim=1)
    test_y = test_data.y

    truncation_levels = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30]
    trunc_df = compute_all_truncation_curves(
        test_x, test_y,
        ckpts_by_size,
        truncation_levels=truncation_levels,
        return_error=True,
    )
    trunc_agg = aggregate_truncation_curves(trunc_df, confidence=0.90, metric_name="error")

    fig, ax = plt.subplots(figsize=(6, 5))
    for size in sizes:
        size_data = trunc_agg[trunc_agg["size"] == size].sort_values("k")
        if len(size_data) == 0:
            continue
        k_vals = size_data["k"].values
        err_mean = size_data["error_mean"].values * 100
        err_low = size_data["error_ci_low"].values * 100
        err_high = size_data["error_ci_high"].values * 100

        ax.plot(k_vals, err_mean, "o-", color=colors[size], label=f"{size}", linewidth=2, markersize=4)
        ax.fill_between(k_vals, err_low, err_high, color=colors[size], alpha=0.2)

    ax.set_xlabel("Eigenvector rank (per digit)", fontsize=12)
    ax.set_ylabel("Classification Error", fontsize=12)
    ax.set_title("Truncation Across Sizes", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(1, 100)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["1%", "2%", "5%", "10%", "20%", "50%", "100%"])
    ax.set_xlim(0, 30)
    ax.legend(title="Model Size", fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "figure_5b_truncation.pdf")
    plt.close(fig)

    # ---------- Appendix: accuracy drop under truncation ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    for size in sizes:
        size_data = trunc_agg[trunc_agg["size"] == size].sort_values("k")
        if len(size_data) == 0:
            continue
        k_vals = size_data["k"].values
        drop_mean = size_data["error_mean"].values * 100
        drop_low = size_data["error_ci_low"].values * 100
        drop_high = size_data["error_ci_high"].values * 100
        ax.plot(k_vals, drop_mean, "o-", color=colors[size], label=f"{size}", linewidth=1.8, markersize=3)
        ax.fill_between(k_vals, drop_low, drop_high, color=colors[size], alpha=0.15)
    # Saturation threshold: absolute accuracy drop (%)
    ax.axhline(0.1, linestyle=":", color="0.3", linewidth=1.5, label="0.1% drop threshold")
    ax.set_xlabel("Eigenvector rank (per digit)")
    ax.set_ylabel("Accuracy drop (%)")
    ax.set_title("Accuracy drop under truncation")
    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=0)
    ax.legend(title="Model Size", fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "appendix_mnist_acc_drop.pdf")
    plt.close(fig)

    # ---------- Appendix: inter-model size similarity vs reference (300) ----------
    reference_size = 300
    ref_ckpts = ckpts_by_size.get(reference_size, [])
    if len(ref_ckpts) == 0:
        print("WARNING: Missing reference size checkpoints (300); skipping inter_similarity plots.")
        return

    means, stds = [], []
    for size in sizes:
        ckpts = ckpts_by_size.get(size, [])
        if len(ckpts) == 0:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        sims = [
            compute_eigenvector_similarity(
                a["eigenvectors"], b["eigenvectors"],
                a["eigenvalues"], b["eigenvalues"],
                top_k=1,
            )
            for a, b in product(ckpts, ref_ckpts)
        ]
        means.append(float(np.mean(sims)))
        stds.append(float(np.std(sims)))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.errorbar(sizes, means, yerr=stds, fmt="o-", capsize=3, linewidth=1.8, markersize=4)
    ax.set_xlabel("Model size (d_hidden)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Inter-size eigenvector similarity (ref={reference_size}, top-1 per class)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.5, -0.22,
        "Cos sim of matched eigenvector ranks, averaged over digit classes.\n"
        f"Sizes={sizes}; seeds={seeds}; ref={reference_size}; top_k=1.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="0.35",
    )
    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "appendix_mnist_inter_similarity.pdf")
    plt.close(fig)

    # ---------- Appendix: inter-size similarity matrix (top eigenvector) ----------
    mat = np.zeros((len(sizes), len(sizes)), dtype=np.float32)
    for i, si in enumerate(sizes):
        for j, sj in enumerate(sizes):
            sims = [
                compute_eigenvector_similarity(
                    a["eigenvectors"], b["eigenvectors"],
                    a["eigenvalues"], b["eigenvalues"],
                    top_k=1,
                )
                for a, b in product(ckpts_by_size.get(si, []), ckpts_by_size.get(sj, []))
            ]
            mat[i, j] = float(np.mean(sims)) if sims else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(sizes)))
    ax.set_yticks(range(len(sizes)))
    ax.set_xticklabels(sizes, rotation=45, ha="right")
    ax.set_yticklabels(sizes)
    ax.set_title("Inter-size similarity (top eigenvector)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "appendix_mnist_inter_size_similarity.pdf")
    plt.close(fig)


def _plot_challenge_from_checkpoint(
    *,
    ckpt_path: Path,
    device: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Paper-style Figure 6 renderer for a single challenge checkpoint.
    Uses the shared plot_challenge_figure function from src/plot_utils/challenge.py
    """
    from src.plot_utils.challenge import plot_challenge_figure
    
    fig = plot_challenge_figure(ckpt_path, device=device, title=title)
    if fig is not None:
        _save_figure(fig, out_path)
        plt.close(fig)


def generate_challenge_section(d: Dirs, device: str = "cpu", seed: int = 42) -> None:
    """Paper Figure 6 and challenge variants (if checkpoints exist) rendered separately."""
    print("\n=== Vision / Challenge task ===")

    # Variants (trained by scripts/train_challenge_variants.py)
    # Note: Old mnist_challenge_seed{seed}.pt files are actually regular MNIST classifiers,
    # not challenge task models, so we use the "none" variant as the main Figure 6.
    variants = [
        ("none", "Challenge task (no reg): eigendecomposition (True − False)", True),
        ("noise", "Challenge task (noise only sigma=0.5): eigendecomposition (True − False)", False),
        ("wd", "Challenge task (weight decay only λ=1.0): eigendecomposition (True − False)", False),
        ("full", "Challenge task (full reg sigma=0.5, λ=1.0): eigendecomposition (True − False)", False),
    ]
    
    found_any = False
    for tag, title, is_primary in variants:
        ckpt = d.challenge_ckpts / f"mnist_challenge_{tag}_seed{seed}.pt"
        if not ckpt.exists():
            if is_primary:
                print(f"  Warning: Primary challenge checkpoint not found at {ckpt}")
            continue
        
        found_any = True
        # Primary variant also gets saved as figure_6_challenge.pdf
        if is_primary:
            _plot_challenge_from_checkpoint(
                ckpt_path=ckpt,
                device=device,
                title="Challenge task: eigendecomposition (True − False direction)",
                out_path=d.figure_out / "figure_6_challenge.pdf",
            )
        
        _plot_challenge_from_checkpoint(
            ckpt_path=ckpt,
            device=device,
            title=title,
            out_path=d.figure_out / f"figure_6_challenge_{tag}.pdf",
        )
    
    if not found_any:
        raise FileNotFoundError(
            f"No challenge checkpoints found in {d.challenge_ckpts}. "
            "Run: python scripts/train/train_challenge_variants.py"
        )


def plot_challenge_decay_panels(
    *,
    variants: Dict[str, Dict],
    out_path: Path,
    n_vals: int = 20,
) -> None:
    """
    Plot eigenvalue decay (pos + neg) for multiple regularization variants in a compact grid.
    """
    from matplotlib.gridspec import GridSpec

    names = list(variants.keys())
    fig = plt.figure(figsize=(3.1 * len(names), 4.8))
    gs = GridSpec(2, len(names), figure=fig, wspace=0.25, hspace=0.35)

    for j, name in enumerate(names):
        vals = variants[name]["eigenvalues"].detach().cpu()
        pos = vals[vals > 0]
        neg = vals[vals < 0]
        pos_sorted = pos.sort(descending=True).values[:n_vals]
        neg_sorted = neg.sort().values[:n_vals]  # most negative first

        ax = fig.add_subplot(gs[0, j])
        if len(pos_sorted):
            ax.plot(np.arange(1, len(pos_sorted) + 1), pos_sorted.numpy(), "-o", linewidth=1.8, markersize=3)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Index", fontsize=9)
        ax.set_ylabel("λ", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, j])
        if len(neg_sorted):
            ax.plot(np.arange(1, len(neg_sorted) + 1), neg_sorted.numpy(), "-o", linewidth=1.8, markersize=3)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Index", fontsize=9)
        ax.set_ylabel("λ", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Challenge task: eigenvalue decay by regularization", fontsize=12, y=1.02)
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_adversarial_section(d: Dirs, device: str = "cpu", target_class: int = 3) -> None:
    """Paper Figure 7 + appendix adversarial encoders (existing checkpoints only).
    Uses the shared generate_figure_7 function from src/plot_utils/adversarial.py
    """
    print("\n=== Vision / Adversarial masks ===")
    
    from src.plot_utils.adversarial import generate_figure_7
    
    seeds = [42, 43, 44, 45, 46]
    
    fig, summary = generate_figure_7(
        mnist_checkpoints_dir=d.vision_mnist_ckpts,
        seeds=seeds,
        device=device,
        target_class=target_class,
    )
    
    _save_figure(fig, d.figure_out / "figure_7_adversarial.pdf")
    plt.close(fig)
    
    print(f"  Noise-reg: Acc={summary['noise_acc_at_alpha']:.3f}±{summary['noise_acc_std']:.3f} @ α={summary['alpha_annotate']}")
    print(f"  No-reg: Acc={summary['noreg_acc_at_alpha']:.3f}±{summary['noreg_acc_std']:.3f} @ α={summary['alpha_annotate']}")


def generate_explanation_section(d: Dirs, device: str = "cpu") -> None:
    """Generate sample explanation figures for each digit class (0-9).
    
    These show how eigenvectors contribute to classifying a sample from each digit class.
    """
    print("\n=== Vision / Sample Explanations ===")
    
    from src.plot_utils.explanation import plot_sample_explanation
    from image.datasets import MNIST
    
    # Load checkpoint
    ckpt_path = d.vision_mnist_ckpts / "mnist_dense_full_seed42.pt"
    if not ckpt_path.exists():
        print(f"WARNING: checkpoint not found at {ckpt_path}; skipping explanation figures.")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    eigenvalues = ckpt["eigenvalues"]
    eigenvectors = ckpt["eigenvectors"]
    
    # Load MNIST test set to get sample images
    dataset = MNIST(train=False, device=device)
    
    # Get one sample per digit - dataset returns (img, label) where img is [1, H, W] and label is tensor
    samples_by_digit = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)
        if label_int not in samples_by_digit:
            samples_by_digit[label_int] = img
        if len(samples_by_digit) == 10:
            break
    
    # Generate explanation figure for each digit
    for digit in range(10):
        if digit not in samples_by_digit:
            print(f"WARNING: No sample found for digit {digit}; skipping.")
            continue
        
        sample = samples_by_digit[digit]
        if sample.dim() == 3:  # [C, H, W] -> [H, W]
            sample = sample.squeeze(0)
        
        fig = plot_sample_explanation(
            sample=sample,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            top_k_eigenvectors=5,
            top_k_classes=3,
            figsize=(14, 6),
        )
        fig.suptitle(f"Sample Explanation: Digit {digit}", fontsize=12, y=1.02)
        
        _save_figure(fig, d.figure_out / f"sample_explanation_digit_{digit}.pdf")
        plt.close(fig)
    
    print(f"Generated explanation figures for all 10 digits.")


def generate_appendix_adversarial_encoders(d: Dirs, device: str = "cpu") -> None:
    """Appendix: additional adversarial mask examples from existing models only."""
    from matplotlib.gridspec import GridSpec
    from src.vision.adversarial import compute_adversarial_mask

    # Conditions (use one seed for visualization)
    seed = 42
    conds = [
        ("No reg", d.vision_mnist_ckpts / f"mnist_dense_none_seed{seed}.pt"),
        ("Noise std=0.15", d.vision_mnist_ckpts / f"mnist_dense_noise015_seed{seed}.pt"),
        ("Noise std=0.3", d.noise_sweep_ckpts / f"mnist_noise_0.3_seed42.pt"),
    ]
    # Filter missing
    conds = [(name, p) for (name, p) in conds if p.exists()]
    if len(conds) == 0:
        print("WARNING: No checkpoints found for appendix adversarial encoders; skipping.")
        return

    digits = [3, 5, 8, 2]  # a few diverse examples
    top_k = 10

    # Layout: rows=digits, cols=2*conds (eigenvector + mask per condition)
    fig = plt.figure(figsize=(2.2 * 2 * len(conds) + 1, 2.1 * len(digits) + 1))
    # Reserve a left margin for row labels ("digit X") so nothing overlaps images.
    gs = GridSpec(len(digits), 2 * len(conds), figure=fig, wspace=0.15, hspace=0.2, left=0.10, right=0.99)

    def im_signed(ax, v: torch.Tensor) -> None:
        img = v.detach().cpu().reshape(28, 28).numpy()
        vmax = float(np.abs(img).max() or 1.0)
        ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axis("off")

    for c_idx, (cname, ckpt_path) in enumerate(conds):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vals = ckpt["eigenvalues"]
        vecs = ckpt["eigenvectors"]

        for r, digit in enumerate(digits):
            if c_idx == 0:
                # Row label in figure margin, not on the image.
                y_pos = 1.0 - (r + 0.5) / len(digits)
                fig.text(0.02, y_pos, f"digit {digit}", fontsize=10, ha="left", va="center")
            # Top positive eigenvector for this digit
            v = vals[digit]
            pos = torch.where(v > 0)[0]
            if len(pos) == 0:
                idx = int(v.abs().argmax().item())
            else:
                idx = int(pos[v[pos].argmax()].item())

            eig = vecs[digit, idx]
            mask = compute_adversarial_mask(vecs[digit], vals[digit], target_rank=0, top_k=top_k, use_positive_only=True)

            ax = fig.add_subplot(gs[r, 2 * c_idx])
            if r == 0:
                ax.set_title(f"{cname}\nEigenvector", fontsize=9)
            im_signed(ax, eig)

            ax = fig.add_subplot(gs[r, 2 * c_idx + 1])
            if r == 0:
                ax.set_title(f"{cname}\nMask", fontsize=9)
            im_signed(ax, mask)

    fig.suptitle("Appendix: adversarial masks (more examples)", fontsize=12, y=1.02)
    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "appendix_adversarial_encoders.pdf")
    plt.close(fig)


def generate_appendix_eigenspectrum_digits(d: Dirs, digit_list: Sequence[int] = (2, 4, 6)) -> None:
    """Appendix: eigenspectrum panels for digits 2/4/6 (existing checkpoint)."""
    print("\n=== Vision / Appendix: eigenspectrum digits ===")

    from matplotlib.gridspec import GridSpec
    from src.vision.spectral import load_checkpoint_eigenvalues

    ckpt_path = d.vision_mnist_ckpts / "mnist_dense_full_seed42.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for appendix digits: {ckpt_path}")

    eigenvalues, eigenvectors = load_checkpoint_eigenvalues(str(ckpt_path))

    n_vecs = 4
    n_vals = 20

    for digit in digit_list:
        vals = eigenvalues[digit].detach().cpu()
        vecs = eigenvectors[digit].detach().cpu()

        pos_idx = torch.where(vals > 0)[0]
        neg_idx = torch.where(vals < 0)[0]

        pos_sorted = pos_idx[vals[pos_idx].argsort(descending=True)] if len(pos_idx) else torch.tensor([], dtype=torch.long)
        neg_sorted = neg_idx[vals[neg_idx].argsort()] if len(neg_idx) else torch.tensor([], dtype=torch.long)

        pos_vals = vals[pos_sorted[:n_vals]].numpy() if len(pos_sorted) else np.array([])
        neg_vals = vals[neg_sorted[:n_vals]].numpy() if len(neg_sorted) else np.array([])

        fig = plt.figure(figsize=(1.8 * (1 + n_vecs), 3.6))
        gs = GridSpec(2, 1 + n_vecs, figure=fig, wspace=0.15, hspace=0.15)

        # Positive eigenvalues line + markers
        ax = fig.add_subplot(gs[0, 0])
        if len(pos_vals):
            x = np.arange(1, len(pos_vals) + 1)
            ax.plot(x, pos_vals, linewidth=2, color=COLORS.get("full", "C0"))
            ax.scatter(np.arange(1, min(n_vecs, len(pos_vals)) + 1), pos_vals[:n_vecs], s=25, color="black")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"Digit {digit}", fontsize=12)
        ax.set_xlabel(f"Top {n_vals}", fontsize=9)
        ax.set_ylabel("λ", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Negative eigenvalues line + markers
        ax = fig.add_subplot(gs[1, 0])
        if len(neg_vals):
            x = np.arange(1, len(neg_vals) + 1)
            ax.plot(x, neg_vals, linewidth=2, color=COLORS.get("none", "C3"))
            ax.scatter(np.arange(1, min(n_vecs, len(neg_vals)) + 1), neg_vals[:n_vecs], s=25, color="black")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel(f"Top {n_vals}", fontsize=9)
        ax.set_ylabel("λ", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Eigenvector panels
        for j in range(n_vecs):
            axv = fig.add_subplot(gs[0, 1 + j])
            if len(pos_sorted) > j:
                idx = int(pos_sorted[j].item())
                img = vecs[idx].reshape(28, 28).numpy()
                vmax = float(np.abs(img).max() or 1.0)
                axv.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axv.axis("off")

        for j in range(n_vecs):
            axv = fig.add_subplot(gs[1, 1 + j])
            if len(neg_sorted) > j:
                idx = int(neg_sorted[j].item())
                img = vecs[idx].reshape(28, 28).numpy()
                vmax = float(np.abs(img).max() or 1.0)
                axv.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axv.axis("off")

        # Avoid tight_layout warnings; saved with bbox_inches='tight'.
        out_name = f"appendix_mnist_eigenspectrum_digit{digit}.pdf"
        _save_figure(fig, d.figure_out / out_name)
        plt.close(fig)


def generate_appendix_sparsity(d: Dirs) -> None:
    """Appendix: sparsity proxy plots for eigenvalues/eigenvectors (existing checkpoints)."""
    print("\n=== Vision / Appendix: sparsity ===")

    def approx_l0(x: torch.Tensor) -> float:
        x = x.abs().flatten()
        l1 = float(x.sum().item())
        l2 = float(torch.sqrt((x ** 2).sum()).item())
        if l2 < 1e-12:
            return 0.0
        return (l1 / l2) ** 2

    def eigenvalue_sparsity(vals: torch.Tensor) -> float:
        # Mean over classes of approximate L0 of |eigenvalues|
        per = [approx_l0(vals[c].abs()) for c in range(vals.shape[0])]
        return float(np.mean(per))

    def eigenvector_sparsity(vals: torch.Tensor, vecs: torch.Tensor, top_k: int = 5) -> float:
        # Mean over classes and top-k eigenvectors (by |eigenvalue|) of approximate L0 of |eigenvector|
        per = []
        for c in range(vals.shape[0]):
            _, idx = torch.topk(vals[c].abs(), k=min(top_k, vals.shape[1]))
            for j in idx:
                per.append(approx_l0(vecs[c, int(j)].abs()))
        return float(np.mean(per))

    # --- Noise sweep (seed42 checkpoints) ---
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    noise_eigval, noise_eigvec = [], []
    noise_found = []
    for nl in noise_levels:
        ckpt_path = d.noise_sweep_ckpts / f"mnist_noise_{nl}_seed42.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vals = ckpt["eigenvalues"]
        vecs = ckpt["eigenvectors"]
        noise_found.append(nl)
        noise_eigval.append(eigenvalue_sparsity(vals))
        noise_eigvec.append(eigenvector_sparsity(vals, vecs, top_k=5))

    # --- Base configs across seeds (bars with error bars) ---
    base_cfgs = ["none", "noise", "wd", "full"]
    seeds = [42, 43, 44, 45, 46]

    base_eigval_mean, base_eigval_std = [], []
    base_eigvec_mean, base_eigvec_std = [], []

    for cfg in base_cfgs:
        vals_list, vec_list = [], []
        for seed in seeds:
            ckpt_path = d.vision_mnist_ckpts / f"mnist_dense_{cfg}_seed{seed}.pt"
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            vals_list.append(eigenvalue_sparsity(ckpt["eigenvalues"]))
            vec_list.append(eigenvector_sparsity(ckpt["eigenvalues"], ckpt["eigenvectors"], top_k=5))
        base_eigval_mean.append(float(np.mean(vals_list)) if vals_list else np.nan)
        base_eigval_std.append(float(np.std(vals_list)) if vals_list else np.nan)
        base_eigvec_mean.append(float(np.mean(vec_list)) if vec_list else np.nan)
        base_eigvec_std.append(float(np.std(vec_list)) if vec_list else np.nan)

    # Plot eigenvector sparsity
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    if len(noise_found):
        ax.plot(noise_found, noise_eigvec, "o-", linewidth=2)
    ax.set_title("Eigenvector sparsity vs input noise")
    ax.set_xlabel("Noise std")
    ax.set_ylabel("Approx L0 (pixel count)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(base_cfgs))
    ax.bar(x, base_eigvec_mean, yerr=base_eigvec_std, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(base_cfgs)
    ax.set_title("Eigenvector sparsity across configs")
    ax.set_ylabel("Approx L0 (pixel count)")
    ax.grid(True, alpha=0.3, axis="y")

    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "appendix_mnist_eigenvec_sparsity.pdf")
    plt.close(fig)

    # Plot eigenvalue sparsity
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    if len(noise_found):
        ax.plot(noise_found, noise_eigval, "o-", linewidth=2)
    ax.set_title("Eigenvalue sparsity vs input noise")
    ax.set_xlabel("Noise std")
    ax.set_ylabel("Approx L0 (effective rank proxy)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(base_cfgs))
    ax.bar(x, base_eigval_mean, yerr=base_eigval_std, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(base_cfgs)
    ax.set_title("Eigenvalue sparsity across configs")
    ax.set_ylabel("Approx L0 (effective rank proxy)")
    ax.grid(True, alpha=0.3, axis="y")

    # Avoid tight_layout warnings; saved with bbox_inches='tight'.
    _save_figure(fig, d.figure_out / "appendix_mnist_eigenval_sparsity.pdf")
    plt.close(fig)


def generate_extension_cross_dataset_section(d: Dirs) -> None:
    """Generate Extension 2 (Cross-Dataset Robustness) figures from JSON results and checkpoints."""
    import json
    from src.plot_utils.extension_cross_dataset import (
        DIGIT_LETTER_PAIRS,
        plot_digit_letter_eigenvector_comparison,
        plot_subspace_overlap_by_rank,
        plot_cosine_similarity_heatmap,
        plot_eigenvalue_distribution_overlay,
        plot_principal_angles,
        plot_eigenvector_embedding,
    )
    
    print("\n=== Extension 2 / Cross-Dataset Robustness ===")
    
    extension_cross_dataset_dir = PROJECT_ROOT / "results/extension_cross_dataset"
    extension_cross_dataset_ckpt_dir = extension_cross_dataset_dir / "checkpoints"
    
    # Load checkpoints for the new visualizations
    mnist_ckpt = d.vision_mnist_ckpts / "mnist_dense_full_seed42.pt"
    emnist_letters_ckpt = extension_cross_dataset_ckpt_dir / "emnist_letters_regularized_seed42.pt"
    emnist_digits_ckpt = extension_cross_dataset_ckpt_dir / "emnist_digits_regularized_seed42.pt"
    
    mnist_data = None
    emnist_letters_data = None
    emnist_digits_data = None
    
    if mnist_ckpt.exists():
        mnist_data = torch.load(mnist_ckpt, map_location="cpu", weights_only=False)
        print(f"Loaded MNIST checkpoint: {mnist_ckpt.name}")
    else:
        print(f"WARNING: MNIST checkpoint not found: {mnist_ckpt}")
    
    if emnist_letters_ckpt.exists():
        emnist_letters_data = torch.load(emnist_letters_ckpt, map_location="cpu", weights_only=False)
        print(f"Loaded EMNIST-Letters checkpoint: {emnist_letters_ckpt.name}")
    else:
        print(f"WARNING: EMNIST-Letters checkpoint not found: {emnist_letters_ckpt}")
    
    if emnist_digits_ckpt.exists():
        emnist_digits_data = torch.load(emnist_digits_ckpt, map_location="cpu", weights_only=False)
        print(f"Loaded EMNIST-Digits checkpoint: {emnist_digits_ckpt.name}")
    else:
        print(f"WARNING: EMNIST-Digits checkpoint not found: {emnist_digits_ckpt}")
    
    # ==========================================================================
    # 1. NEW: Subspace overlap by rank (line plot, replaces bar chart)
    # ==========================================================================
    if mnist_data is not None and emnist_letters_data is not None:
        print("\nGenerating: Subspace overlap by rank...")
        fig = plot_subspace_overlap_by_rank(
            mnist_vecs=mnist_data["eigenvectors"],
            emnist_vecs=emnist_letters_data["eigenvectors"],
            mnist_vals=mnist_data["eigenvalues"],
            emnist_vals=emnist_letters_data["eigenvalues"],
            pairs=DIGIT_LETTER_PAIRS,
        )
        _save_figure(fig, d.figure_out / "extension_cross_dataset_subspace_overlap.pdf")
        plt.close(fig)
        print("Generated: extension_cross_dataset_subspace_overlap.pdf")
    
    # ==========================================================================
    # 2. FIXED: Mechanism stability results (correct JSON keys)
    # ==========================================================================
    mechanism_path = extension_cross_dataset_dir / "mechanism_stability_results.json"
    if mechanism_path.exists():
        with open(mechanism_path) as f:
            mechanism_results = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left panel: Cross-dataset accuracy (FIXED: use 'functional' key)
        if 'functional' in mechanism_results:
            func = mechanism_results['functional']
            
            # Extract values
            labels = ['MNIST→EMNIST', 'EMNIST→MNIST']
            accuracies = [
                func.get('mnist_on_emnist', 0),
                func.get('emnist_on_mnist', 0),
            ]
            bidirectional = func.get('bidirectional_avg', 0)
            
            ax = axes[0]
            x = np.arange(len(labels))
            bars = ax.bar(x, accuracies, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax.axhline(bidirectional, color='green', linestyle='--', linewidth=2,
                      label=f'Bidirectional avg: {bidirectional:.1%}')
            
            # Add value labels on bars
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.1%}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Cross-Dataset Classification Accuracy', fontsize=12)
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
        
        # Right panel: Per-digit eigenvector overlap (FIXED: use 'representational' key)
        if 'representational' in mechanism_results:
            repr_data = mechanism_results['representational']
            per_class = repr_data.get('per_class_overlap', {})
            
            ax = axes[1]
            digits = sorted([int(k) for k in per_class.keys()])
            overlaps = [per_class[str(d)]['mean_cos'] for d in digits]
            
            bars = ax.bar(digits, overlaps, color='steelblue', alpha=0.8)
            
            # Add aggregate mean line
            agg = repr_data.get('aggregate', {})
            mean_overlap = agg.get('mean', np.mean(overlaps))
            ax.axhline(mean_overlap, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_overlap:.3f}')
            
            ax.set_xticks(digits)
            ax.set_xlabel('Digit Class', fontsize=12)
            ax.set_ylabel('Eigenvector Overlap (mean cosine)', fontsize=12)
            ax.set_title('MNIST↔EMNIST-Digits Eigenvector Similarity', fontsize=12)
            ax.legend(loc='lower right')
            ax.set_ylim(0, 0.7)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Extension 2: Mechanism Stability Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        _save_figure(fig, d.figure_out / "extension_cross_dataset_mechanism_stability.pdf")
        plt.close(fig)
        print("Generated: extension_cross_dataset_mechanism_stability.pdf")
    else:
        print(f"WARNING: {mechanism_path} not found, skipping mechanism stability figure")
    
    # ==========================================================================
    # 3. Eigenspectra comparison (MNIST vs EMNIST-Digits) with CI across seeds
    # ==========================================================================
    from src.plot_utils.extension_cross_dataset import plot_eigenspectra_comparison_with_ci
    
    # Load all seeds for aggregation
    seeds = [42, 43, 44, 45, 46]
    mnist_eigenvalues_list = []
    emnist_eigenvalues_list = []
    
    print("\nLoading eigenspectra from multiple seeds...")
    for seed in seeds:
        mnist_ckpt_seed = d.vision_mnist_ckpts / f"mnist_dense_full_seed{seed}.pt"
        emnist_ckpt_seed = extension_cross_dataset_ckpt_dir / f"emnist_digits_regularized_seed{seed}.pt"
        
        if mnist_ckpt_seed.exists():
            data = torch.load(mnist_ckpt_seed, map_location="cpu", weights_only=False)
            mnist_eigenvalues_list.append(data["eigenvalues"])
        
        if emnist_ckpt_seed.exists():
            data = torch.load(emnist_ckpt_seed, map_location="cpu", weights_only=False)
            emnist_eigenvalues_list.append(data["eigenvalues"])
    
    print(f"Loaded {len(mnist_eigenvalues_list)} MNIST seeds, {len(emnist_eigenvalues_list)} EMNIST seeds")
    
    if len(mnist_eigenvalues_list) > 0 and len(emnist_eigenvalues_list) > 0:
        fig = plot_eigenspectra_comparison_with_ci(
            mnist_eigenvalues_list=mnist_eigenvalues_list,
            emnist_eigenvalues_list=emnist_eigenvalues_list,
            top_k=50,
            ci_level=0.90,
        )
        _save_figure(fig, d.figure_out / "extension_cross_dataset_eigenspectra.pdf")
        plt.close(fig)
        print("Generated: extension_cross_dataset_eigenspectra.pdf")
    else:
        print("WARNING: Not enough checkpoints found for eigenspectra comparison")
    
    # ==========================================================================
    # 4. NEW: Digit-letter eigenvector comparisons (4 figures)
    # ==========================================================================
    if mnist_data is not None and emnist_letters_data is not None:
        print("\nGenerating: Digit-letter eigenvector comparisons...")
        for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
            fig = plot_digit_letter_eigenvector_comparison(
                mnist_vecs=mnist_data["eigenvectors"],
                mnist_vals=mnist_data["eigenvalues"],
                emnist_vecs=emnist_letters_data["eigenvectors"],
                emnist_vals=emnist_letters_data["eigenvalues"],
                digit_class=digit_idx,
                letter_class=letter_idx,
                pair_label=label,
                n_top=5,
            )
            filename = f"extension_cross_dataset_eigenvec_{label.replace('-', '_')}.pdf"
            _save_figure(fig, d.figure_out / filename)
            plt.close(fig)
            print(f"Generated: {filename}")
    
    # ==========================================================================
    # 5. NEW: Cosine similarity heatmaps (4 figures)
    # ==========================================================================
    if mnist_data is not None and emnist_letters_data is not None:
        print("\nGenerating: Cosine similarity heatmaps...")
        for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
            digit_label = label.split("-")[0]
            letter_label = label.split("-")[1]
            
            fig = plot_cosine_similarity_heatmap(
                digit_vecs=mnist_data["eigenvectors"][digit_idx],
                letter_vecs=emnist_letters_data["eigenvectors"][letter_idx],
                digit_vals=mnist_data["eigenvalues"][digit_idx],
                letter_vals=emnist_letters_data["eigenvalues"][letter_idx],
                k=10,
                digit_label=digit_label,
                letter_label=letter_label,
            )
            filename = f"extension_cross_dataset_cosine_heatmap_{label.replace('-', '_')}.pdf"
            _save_figure(fig, d.figure_out / filename)
            plt.close(fig)
            print(f"Generated: {filename}")
    
    # ==========================================================================
    # 6. NEW: Eigenvalue distribution overlays (4 figures)
    # ==========================================================================
    if mnist_data is not None and emnist_letters_data is not None:
        print("\nGenerating: Eigenvalue distribution overlays...")
        for digit_idx, letter_idx, label in DIGIT_LETTER_PAIRS:
            digit_label = label.split("-")[0]
            letter_label = label.split("-")[1]
            
            fig = plot_eigenvalue_distribution_overlay(
                digit_vals=mnist_data["eigenvalues"][digit_idx],
                letter_vals=emnist_letters_data["eigenvalues"][letter_idx],
                digit_label=digit_label,
                letter_label=letter_label,
            )
            filename = f"extension_cross_dataset_eigenval_dist_{label.replace('-', '_')}.pdf"
            _save_figure(fig, d.figure_out / filename)
            plt.close(fig)
            print(f"Generated: {filename}")
    
    # ==========================================================================
    # 7. NEW: Principal angles visualization (1 figure with all pairs)
    # ==========================================================================
    if mnist_data is not None and emnist_letters_data is not None:
        print("\nGenerating: Principal angles visualization...")
        fig = plot_principal_angles(
            mnist_vecs=mnist_data["eigenvectors"],
            emnist_vecs=emnist_letters_data["eigenvectors"],
            mnist_vals=mnist_data["eigenvalues"],
            emnist_vals=emnist_letters_data["eigenvalues"],
            pairs=DIGIT_LETTER_PAIRS,
            k=20,
        )
        _save_figure(fig, d.figure_out / "extension_cross_dataset_principal_angles.pdf")
        plt.close(fig)
        print("Generated: extension_cross_dataset_principal_angles.pdf")
    
    # ==========================================================================
    # 8. NEW: t-SNE and PCA embeddings (2 figures)
    # ==========================================================================
    if mnist_data is not None and emnist_letters_data is not None:
        print("\nGenerating: Eigenvector embeddings...")
        
        # PCA embedding
        fig = plot_eigenvector_embedding(
            mnist_vecs=mnist_data["eigenvectors"],
            emnist_vecs=emnist_letters_data["eigenvectors"],
            mnist_vals=mnist_data["eigenvalues"],
            emnist_vals=emnist_letters_data["eigenvalues"],
            pairs=DIGIT_LETTER_PAIRS,
            k=5,
            method="pca",
        )
        if fig is not None:
            _save_figure(fig, d.figure_out / "extension_cross_dataset_eigenvec_pca.pdf")
            plt.close(fig)
            print("Generated: extension_cross_dataset_eigenvec_pca.pdf")
        
        # t-SNE embedding
        fig = plot_eigenvector_embedding(
            mnist_vecs=mnist_data["eigenvectors"],
            emnist_vecs=emnist_letters_data["eigenvectors"],
            mnist_vals=mnist_data["eigenvalues"],
            emnist_vals=emnist_letters_data["eigenvalues"],
            pairs=DIGIT_LETTER_PAIRS,
            k=5,
            method="tsne",
        )
        if fig is not None:
            _save_figure(fig, d.figure_out / "extension_cross_dataset_eigenvec_tsne.pdf")
            plt.close(fig)
            print("Generated: extension_cross_dataset_eigenvec_tsne.pdf")
    
    print("\n=== Extension 2 figure generation complete ===")


def generate_paper_hub(d: Dirs) -> None:
    """Generate results/interactive/paper_hub.html (sections, not figure numbers)."""
    print("\n=== Vision / Hub: paper_hub.html ===")

    out_dir = PROJECT_ROOT / "results/interactive"
    _ensure_dir(out_dir)
    bundle_dir = out_dir / "paper_hub_bundle"
    figures_dir = bundle_dir / "figures"
    assets_dir = bundle_dir / "assets"
    _ensure_dir(figures_dir)
    _ensure_dir(assets_dir)

    out_path = bundle_dir / "paper_hub.html"
    zip_path = out_dir / "paper_hub_bundle.zip"

    def fig_src_path(name: str) -> Optional[Path]:
        p = d.figure_out / name
        return p if p.exists() else None

    def fig_rel_path(name: str) -> str:
        # Files are copied into bundle_dir/figures/<name>
        return f"figures/{name}"

    sections: Dict[str, List[Dict[str, str]]] = {
        "Vision / Regularization": [
            {"label": "Eigenspectrum comparison", "path": fig_rel_path("eigenspectrum_comparison.pdf")},
            {"label": "Eigenvalue decay", "path": fig_rel_path("eigenvalue_decay.pdf")},
            {"label": "Eigenvectors (no reg)", "path": fig_rel_path("eigenvectors_noreg.pdf")},
            {"label": "Update: Eigenvectors (noise only, sigma=0.5)", "path": fig_rel_path("eigenvectors_noise.pdf")},
            {"label": "Eigenvectors (full reg)", "path": fig_rel_path("eigenvectors_reg.pdf")},
            {"label": "Update: Fashion eigenvectors (noise only, sigma=0.5)", "path": fig_rel_path("fashion_eigenvectors_noise.pdf")},
            {"label": "Ablation (MNIST)", "path": fig_rel_path("mnist_ablation.pdf")},
            {"label": "Tradeoff (MNIST)", "path": fig_rel_path("accuracy_vs_effrank_mnist.pdf")},
        ],
        "Vision / Truncation & similarity": [
            {"label": "Figure 5a: similarity", "path": fig_rel_path("figure_5a_similarity.pdf")},
            {"label": "Figure 5b: truncation", "path": fig_rel_path("figure_5b_truncation.pdf")},
        ],
        "Vision / Challenge task": [
            {"label": "Figure 6: challenge", "path": fig_rel_path("figure_6_challenge.pdf")},
            {"label": "Update: Challenge decay by regularization (none/noise/wd/full)", "path": fig_rel_path("figure_6_challenge_eigenvalue_decay_by_reg.pdf")},
            {"label": "Figure 6 (no reg)", "path": fig_rel_path("figure_6_challenge_none.pdf")},
            {"label": "Figure 6 (noise only sigma=0.5)", "path": fig_rel_path("figure_6_challenge_noise.pdf")},
            {"label": "Figure 6 (weight decay only λ=1.0)", "path": fig_rel_path("figure_6_challenge_wd.pdf")},
            {"label": "Figure 6 (full reg sigma=0.5, λ=1.0)", "path": fig_rel_path("figure_6_challenge_full.pdf")},
        ],
        "Vision / Adversarial masks": [
            {"label": "Figure 7: adversarial masks", "path": fig_rel_path("figure_7_adversarial.pdf")},
        ],
        "Vision / Appendix": [
            {"label": "Appendix: eigenspectrum digit 2", "path": fig_rel_path("appendix_mnist_eigenspectrum_digit2.pdf")},
            {"label": "Appendix: eigenspectrum digit 4", "path": fig_rel_path("appendix_mnist_eigenspectrum_digit4.pdf")},
            {"label": "Appendix: eigenspectrum digit 6", "path": fig_rel_path("appendix_mnist_eigenspectrum_digit6.pdf")},
            {"label": "Appendix: truncation acc drop", "path": fig_rel_path("appendix_mnist_acc_drop.pdf")},
            {"label": "Appendix: inter-size similarity (ref=300, top-1)", "path": fig_rel_path("appendix_mnist_inter_similarity.pdf")},
            {"label": "Appendix: inter-size similarity matrix", "path": fig_rel_path("appendix_mnist_inter_size_similarity.pdf")},
            {"label": "Appendix: eigenvector sparsity", "path": fig_rel_path("appendix_mnist_eigenvec_sparsity.pdf")},
            {"label": "Appendix: eigenvalue sparsity", "path": fig_rel_path("appendix_mnist_eigenval_sparsity.pdf")},
            {"label": "Appendix: adversarial masks (more examples)", "path": fig_rel_path("appendix_adversarial_encoders.pdf")},
        ],
    }

    # Copy PDFs into bundle figures/.
    # Keep nav entries even if missing (render disabled buttons) so the hub reflects
    # the full intended structure and makes it obvious what's missing.
    import shutil

    for sec in list(sections.keys()):
        kept = []
        for it in sections[sec]:
            name = Path(it["path"]).name
            src = fig_src_path(name)
            if not src:
                it2 = dict(it)
                it2["missing"] = "1"
                kept.append(it2)
                continue
            dst = figures_dir / name
            shutil.copy(src, dst)
            it2 = dict(it)
            it2["missing"] = "0"
            kept.append(it2)
        sections[sec] = kept

    # Plotly widget: eigenspectrum-with-signs (digit selector)
    plotly_block = "<div><em>Plotly not available.</em></div>"
    plotly_js = ""
    try:
        from src.plot_utils.explanation import plot_eigenspectrum_with_signs
        from src.vision.spectral import load_checkpoint_eigenvalues
        import plotly

        ckpt = d.vision_mnist_ckpts / "mnist_dense_full_seed42.pt"
        if ckpt.exists():
            vals, vecs = load_checkpoint_eigenvalues(str(ckpt))
            digit_divs = []
            for digit in range(vals.shape[0]):
                fig = plot_eigenspectrum_with_signs(vals, vecs, digit=digit, n_eigenvectors=4, n_eigenvalues=20)
                html = fig.to_html(full_html=False, include_plotlyjs=False)
                digit_divs.append(f'<div class="plotlyDigit" data-digit="{digit}" style="display:none">{html}</div>')

            # Self-contained bundle: ship plotly.js locally.
            from plotly.offline import get_plotlyjs

            plotly_js_path = assets_dir / "plotly.min.js"
            plotly_js_path.write_text(get_plotlyjs(), encoding="utf-8")
            plotly_js = '<script src="assets/plotly.min.js"></script>'
            plotly_block = f"""
            <div class="card">
              <div style="display:flex; align-items:center; gap:12px;">
                <div style="font-weight:600;">Interactive: Eigenspectrum (with signs)</div>
                <label>Digit:
                  <select id="digitSelect">
                    {''.join([f'<option value=\"{i}\">{i}</option>' for i in range(10)])}
                  </select>
                </label>
              </div>
              <div id="plotlyContainer">
                {''.join(digit_divs)}
              </div>
            </div>
            """
    except Exception:
        pass

    # Build nav HTML
    nav_parts = []
    first_path = ""
    for sec, items in sections.items():
        if not items:
            continue
        nav_parts.append(f'<div class="navSection">{sec}</div>')
        for it in items:
            if not first_path:
                first_path = it["path"]
            is_missing = it.get("missing") == "1"
            disabled = "disabled" if is_missing else ""
            label = it["label"] + (" (missing)" if is_missing else "")
            onclick = "" if is_missing else f"onclick=\"loadPdf({it['path']!r}, {it['label']!r})\""
            nav_parts.append(f'<button class="navItem" {onclick} {disabled}>{label}</button>')

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Bilinear MLP Paper Hub (Vision)</title>
  {plotly_js}
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; }}
    .layout {{ display: grid; grid-template-columns: 320px 1fr; height: 100vh; }}
    .sidebar {{ border-right: 1px solid #e5e7eb; padding: 14px; overflow:auto; }}
    .content {{ padding: 14px; overflow:hidden; display:flex; flex-direction:column; gap:12px; }}
    .title {{ font-size: 16px; font-weight: 700; margin-bottom: 10px; }}
    .navSection {{ margin-top: 14px; font-size: 12px; font-weight: 700; color: #374151; }}
    .navItem {{ width: 100%; text-align:left; padding: 8px 10px; margin-top: 6px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; cursor: pointer; }}
    .navItem:hover {{ background: #f9fafb; }}
    .navItem:disabled {{ opacity: 0.55; cursor: not-allowed; background: #f9fafb; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; margin-top: 12px; background: #fff; }}
    .pdfCard {{ flex: 1; min-height: 0; margin-top: 0; display:flex; flex-direction:column; gap:8px; }}
    iframe {{ width: 100%; flex: 1; min-height: 0; border: 1px solid #e5e7eb; border-radius: 12px; }}
    .muted {{ color: #6b7280; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="sidebar">
      <div class="title">Paper Hub (Vision)</div>
      <div class="muted">Organized by paper sections/keywords</div>
      <div class="muted">Build: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
      {''.join(nav_parts)}
    </div>
    <div class="content">
      <div class="card pdfCard">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div id="pdfTitle" style="font-weight:700;">Figure</div>
          <div class="muted" id="pdfPath"></div>
        </div>
        <iframe id="pdfFrame" src=""></iframe>
      </div>
      {plotly_block}
    </div>
  </div>

  <script>
    function loadPdf(path, title) {{
      const frame = document.getElementById('pdfFrame');
      const t = document.getElementById('pdfTitle');
      const p = document.getElementById('pdfPath');
      t.textContent = title;
      p.textContent = path;
      frame.src = path;
    }}

    // init iframe
    const initial = {first_path!r};
    if (initial) {{
      loadPdf(initial, "Overview");
    }}

    // plotly digit selector
    function showDigit(d) {{
      const nodes = document.querySelectorAll('.plotlyDigit');
      nodes.forEach(n => {{
        n.style.display = (n.dataset.digit === d) ? 'block' : 'none';
      }});
    }}
    const sel = document.getElementById('digitSelect');
    if (sel) {{
      sel.addEventListener('change', (e) => showDigit(e.target.value));
      sel.value = '0';
      showDigit('0');
    }}
  </script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")

    # Write a stable entrypoint alongside the bundle dir to avoid confusion about which hub to open.
    # This file is small and just redirects to the bundle.
    stable_path = out_dir / "paper_hub.html"
    stable_html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta http-equiv="refresh" content="0; url=paper_hub_bundle/paper_hub.html"/>
  <title>Paper Hub (redirect)</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding: 18px; }
  </style>
</head>
<body>
  <div>Redirecting to <code>paper_hub_bundle/paper_hub.html</code>…</div>
  <div>If you are not redirected, open: <code>paper_hub_bundle/paper_hub.html</code></div>
</body>
</html>
"""
    stable_path.write_text(stable_html, encoding="utf-8")
    print(f"Saved: {stable_path}")

    # Zip up the bundle for sharing (self-contained).
    import zipfile

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(bundle_dir)))
    print(f"Saved: {zip_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified vision analysis")
    parser.add_argument(
        "--sections",
        type=str,
        nargs="+",
        default=["regularization", "truncation_similarity", "challenge", "adversarial", "explanation", "extension_cross_dataset", "appendix", "hub"],
        help="Sections to generate: regularization truncation_similarity challenge adversarial explanation extension_cross_dataset appendix hub",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu|mps|cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-class", type=int, default=3)
    args = parser.parse_args()

    # For analysis/plotting we default to CPU for stability/reproducibility.
    # If you want acceleration, pass --device mps or --device cuda explicitly.
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    set_publication_style()
    
    # Ensure artifacts are available (downloads from Google Drive if missing)
    ensure_artifacts()
    
    d = get_dirs()

    sections = set(args.sections)

    if "regularization" in sections:
        generate_regularization_section(d)
    if "truncation_similarity" in sections:
        generate_truncation_similarity_section(d)
    if "challenge" in sections:
        generate_challenge_section(d, device=device, seed=args.seed)
        # Optional: if the user has already trained challenge variants (in a separate training script),
        # we can plot the eigenvalue decay comparison.
        variants_spec = {
            "none (sigma=0.0, λ=0.0)": d.challenge_ckpts / f"mnist_challenge_none_seed{args.seed}.pt",
            "noise (sigma=0.5, λ=0.0)": d.challenge_ckpts / f"mnist_challenge_noise_seed{args.seed}.pt",
            "wd (sigma=0.0, λ=1.0)": d.challenge_ckpts / f"mnist_challenge_wd_seed{args.seed}.pt",
            "full (sigma=0.5, λ=1.0)": d.challenge_ckpts / f"mnist_challenge_full_seed{args.seed}.pt",
        }
        variants = {}
        for name, path in variants_spec.items():
            if path.exists():
                variants[name] = torch.load(path, map_location="cpu", weights_only=False)
        if len(variants) >= 2:
            plot_challenge_decay_panels(
                variants=variants,
                out_path=d.figure_out / "figure_6_challenge_eigenvalue_decay_by_reg.pdf",
            )
    if "adversarial" in sections:
        generate_adversarial_section(d, device=device, target_class=args.target_class)
    if "explanation" in sections:
        generate_explanation_section(d, device=device)
    if "extension_cross_dataset" in sections:
        generate_extension_cross_dataset_section(d)
    if "appendix" in sections:
        generate_appendix_eigenspectrum_digits(d)
        generate_appendix_sparsity(d)
        generate_appendix_adversarial_encoders(d, device=device)
    if "hub" in sections:
        print("\n=== Hub ===")
        print("PaperHub is now generated by: python scripts/figures/paper_hub.py")
        print("Skipping hub generation inside vision_analysis for separation of concerns.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

