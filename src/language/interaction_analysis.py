"""
Interaction matrix analysis using the original paper's Tracer class.

For each SAE output feature, Tracer computes Q matrices representing
how input feature pairs interact to produce the output feature.

Paper claim: 69% of features have >0.75 correlation with rank-2 approximation.

Usage:
    python src/language/interaction_analysis.py --config configs/language_interaction.yaml
"""

import sys
from pathlib import Path
import argparse
import time
import json
import warnings

# Suppress torchvision image extension warning (libjpeg not needed for our use case)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import numpy as np
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from sae.tracer import Tracer
# Note: We use our own SVD-based implementations below instead of the original
# from sae.functions import compute_effective_rank, compute_truncated_eigenvalues

from src.utils import (
    get_device,
    load_config,
    track_emissions,
    safe_eigh,
    init_wandb,
    finish_wandb,
)
from src.language.interaction_utils import (
    get_interaction_eigenpairs_from_tracer,
    compute_interaction_matrix,
)
from src.language.context import LanguageContext


# =============================================================================
# SVD-based implementations (more numerically stable for large matrices)
# For symmetric matrices: singular values = |eigenvalues|
# =============================================================================

def _safe_svdvals(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute singular values with CPU fallback for stability.
    
    SVD is more numerically stable than eigendecomposition and uses
    less workspace memory, making it better for large matrices.
    """
    device = tensor.device
    # Always compute on CPU for large matrices to avoid LAPACK workspace issues
    if tensor.shape[-1] > 4096 or device.type == "mps":
        return torch.linalg.svdvals(tensor.cpu()).to(device)
    return torch.linalg.svdvals(tensor)


def compute_effective_rank(data: torch.Tensor) -> torch.Tensor:
    """
    Compute effective rank using (L1/L2)^2 formula via SVD.
    
    For symmetric matrices, singular values equal |eigenvalues|,
    so this is mathematically equivalent to the eigenvalue-based formula.
    
    Args:
        data: Tensor of shape [..., n, n] (symmetric matrices)
        
    Returns:
        Effective rank for each matrix in the batch
    """
    # For symmetric matrices: singular values = |eigenvalues|
    vals = _safe_svdvals(data)
    
    l2 = vals.pow(2).sum(-1).sqrt()
    l1 = vals.sum(-1)  # Already positive since these are singular values
    
    return (l1 / l2).pow(2)


def compute_truncated_eigenvalues(data: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Compute sum of top-k singular values (= top-k |eigenvalues| for symmetric matrices).
    
    Args:
        data: Tensor of shape [..., n, n] (symmetric matrices)
        k: Number of top values to sum
        
    Returns:
        Sum of top-k singular values for each matrix
    """
    vals = _safe_svdvals(data)
    # SVD returns values in descending order, so just take first k
    return vals[..., :k].sum(-1)


def rank_k_variance_explained(Q: torch.Tensor, k: int = 2) -> float:
    """
    Compute fraction of eigenvalue mass captured by top-k eigenvalues.

    variance_explained = sum(|lambda_1|, ..., |lambda_k|) / sum(|lambda_i|)

    Paper claims 69% of features have >0.75 with k=2.
    
    Uses SVD for numerical stability (singular values = |eigenvalues| for symmetric matrices).
    """
    # Symmetrize Q
    Q_sym = 0.5 * (Q + Q.T)

    # Use SVD instead of eigendecomposition for stability
    try:
        singular_values = _safe_svdvals(Q_sym)
    except Exception:
        return float('nan')

    # Compute variance explained by top-k singular values
    # (SVD returns values in descending order)
    total_mass = singular_values.sum()

    if total_mass < 1e-10:
        return float('nan')

    top_k_mass = singular_values[:min(k, len(singular_values))].sum()
    return (top_k_mass / total_mass).item()


def analyze_single_feature_original(tracer: Tracer, feat_idx: int, rank_k: int = 2) -> dict:
    """
    Analyze a single feature using the original tracer.q() implementation.

    Always projects onto SAE latents (project=True) for interpretability.
    Processes one feature at a time to manage memory.
    """
    # Use original tracer.q() with projection - returns Q on the model's device
    Q = tracer.q(feat_idx, project=True)

    # Move to CPU immediately
    Q_cpu = Q.float().cpu()
    del Q

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Symmetrize for eigendecomposition
    Q_sym = 0.5 * (Q_cpu + Q_cpu.T)

    # Compute metrics
    var_explained = rank_k_variance_explained(Q_cpu, k=rank_k)
    eff_rank = compute_effective_rank(Q_sym.unsqueeze(0)).item()
    trunc_eig = compute_truncated_eigenvalues(Q_sym.unsqueeze(0), k=rank_k).item()

    del Q_cpu, Q_sym

    return {
        "variance_explained": var_explained,
        "effective_rank": eff_rank,
        "truncated_eigenvalue": trunc_eig,
    }


def analyze_single_feature_manual(tracer: Tracer, feat_idx: int, rank_k: int = 2) -> dict:
    """
    Analyze a single feature's interaction matrix using the interaction_utils module.

    Memory-efficient CPU-based implementation. Always projects onto SAE latents
    for spectral statistics (effective rank, variance explained).
    
    The core Q computation is factored out into interaction_utils.py for reuse
    by verify_correlation.py.
    """
    model, layer = tracer.model, tracer.layer
    out_latent = tracer.out_latents[feat_idx]  # Single feature vector [d_model]

    # Get weights for Q computation
    w_l = model.w_l[layer].float().cpu()  # [d_hidden, d_model]
    w_r = model.w_r[layer].float().cpu()  # [d_hidden, d_model]
    w_p = model.w_p[layer].float().cpu()  # [d_model, d_hidden]
    out_vec = out_latent.float().cpu()    # [d_model]

    # Compute unprojected Q using the utility function
    Q_unprojected = compute_interaction_matrix(w_l, w_r, w_p, out_vec, symmetrize=True)

    # Clean up weight tensors
    del w_l, w_r, w_p, out_vec

    # Project onto SAE latents for spectral statistics (preserves existing behavior)
    # This is needed for interpretability metrics as per the paper
    inp_latents = tracer.inp_latents.float().cpu()
    Q_projected = inp_latents.T @ Q_unprojected @ inp_latents
    Q_projected = 0.5 * (Q_projected + Q_projected.T)  # Re-symmetrize after projection
    del inp_latents, Q_unprojected

    # Compute metrics on projected Q
    var_explained = rank_k_variance_explained(Q_projected, k=rank_k)
    eff_rank = compute_effective_rank(Q_projected.unsqueeze(0)).item()
    trunc_eig = compute_truncated_eigenvalues(Q_projected.unsqueeze(0), k=rank_k).item()

    del Q_projected

    return {
        "variance_explained": var_explained,
        "effective_rank": eff_rank,
        "truncated_eigenvalue": trunc_eig,
    }


def analyze_single_feature(tracer: Tracer, feat_idx: int, rank_k: int = 2) -> dict:
    """
    Analyze a single feature's interaction matrix.

    Always uses manual implementation with projection onto SAE latents.
    The original tracer.q(project=True) einsum creates a ~281TB intermediate tensor
    when inp_latents is [1024, 8192]. Manual implementation uses matrix multiplication
    (inp_latents.T @ Q @ inp_latents) which is memory-efficient.
    """
    return analyze_single_feature_manual(tracer, feat_idx, rank_k)


def analyze_interactions_batch(tracer: Tracer, feature_indices: list, rank_k: int = 2, device: str = "cuda"):
    """
    Analyze interaction matrices for multiple output features.

    Always projects onto SAE latents and uses memory-efficient CPU computation.
    """
    results = {
        "variance_explained": [],
        "effective_ranks": [],
        "truncated_eigenvalues": [],
        "feature_indices": [],
    }

    is_cuda = device.startswith("cuda") or device == "cuda"

    for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
        try:
            # CPU-based analysis with projection (memory-efficient)
            metrics = analyze_single_feature(tracer, feat_idx, rank_k=rank_k)
            var_explained = metrics["variance_explained"]
            eff_rank = metrics["effective_rank"]
            trunc_eig = metrics["truncated_eigenvalue"]

            if not np.isnan(var_explained):
                results["variance_explained"].append(var_explained)
                results["effective_ranks"].append(eff_rank)
                results["truncated_eigenvalues"].append(trunc_eig)
                results["feature_indices"].append(feat_idx)

        except Exception as e:
            print(f"Error analyzing feature {feat_idx}: {e}")
            # Still try to clear memory on error
            if is_cuda:
                torch.cuda.empty_cache()
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Interaction matrix analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/interaction_analysis.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Auto-detect device (includes MPS support)
    # Tracer is now MPS-safe - einsum operations run on CPU internally
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)
    config_name = config.get("name", Path(args.config).stem)

    # Initialize wandb
    wandb_enabled = init_wandb(
        name=config_name,
        config=config,
        device=device,
        enabled=not args.no_wandb,
        tags=["language", "interaction"],
    )

    # Run analysis with emissions tracking
    with track_emissions("bilinear-mlp-reproduction") as tracker:
        # Use LanguageContext for unified model/Tracer loading
        ctx = LanguageContext(config, device)
        model = ctx.model
        model_name = ctx.model_name
        layer = ctx.layer
        
        # Create Tracer via context
        tracer = ctx.get_tracer()

        # Print tracer info for debugging
        print(f"\nTracer info:")
        print(f"  out_latents shape: {tracer.out_latents.shape}")
        print(f"  inp_latents shape: {tracer.inp_latents.shape}")
        print(f"  out SAE d_features: {tracer.out.d_features}")
        print(f"  out SAE d_model: {tracer.out.d_model}")

        # Analysis configuration
        analysis_config = config.get("analysis", {})
        n_features = analysis_config.get("n_features", 500)
        rank_k = analysis_config.get("rank_k", 2)

        # Get number of output features
        n_out_features = tracer.out.d_features
        print(f"\nTotal output features: {n_out_features}")

        # Select features to analyze
        feature_indices = list(range(min(n_features, n_out_features)))
        print(f"Analyzing {len(feature_indices)} features...")
        print(f"Always projecting onto SAE latents (memory-efficient)")

        # Debug: analyze first feature and print Q matrix info
        print(f"\n--- Diagnostic: First feature (idx=0) ---")
        # Use analyze_single_feature which always projects (memory-efficient)
        test_metrics = analyze_single_feature(tracer, 0, rank_k=rank_k)
        print(f"Rank-{rank_k} variance explained: {test_metrics['variance_explained']:.6f}")
        print(f"Effective rank: {test_metrics['effective_rank']:.2f}")
        print(f"Truncated eigenvalue sum: {test_metrics['truncated_eigenvalue']:.6f}")
        print(f"--- End diagnostic ---\n")

        # Run analysis
        results = analyze_interactions_batch(
            tracer, feature_indices, rank_k=rank_k, device=device
        )

    # Compute summary statistics
    variance_explained = np.array(results["variance_explained"])
    effective_ranks = np.array(results["effective_ranks"])

    if len(variance_explained) == 0:
        print("ERROR: No features successfully analyzed")
        if wandb_enabled:
            finish_wandb(tracker.result, extra_summary={"error": "no_features_analyzed"})
        return

    fraction_above_075 = (variance_explained > 0.75).mean()
    fraction_above_050 = (variance_explained > 0.50).mean()

    summary = {
        "n_analyzed": len(variance_explained),
        "n_total_features": n_out_features,
        "rank_k": rank_k,
        "model_name": model_name,
        "layer": layer,
        "fraction_above_075": float(fraction_above_075),
        "fraction_above_050": float(fraction_above_050),
        "mean_variance_explained": float(variance_explained.mean()),
        "std_variance_explained": float(variance_explained.std()),
        "median_variance_explained": float(np.median(variance_explained)),
        "mean_effective_rank": float(effective_ranks.mean()),
        "std_effective_rank": float(effective_ranks.std()),
        "paper_claim": "69% of features have >0.75 rank-2 variance explained",
        "our_result": f"{fraction_above_075*100:.1f}% of features have >0.75 rank-{rank_k} variance explained",
        "claim_supported": bool(fraction_above_075 > 0.60),  # Allow 9% margin
        "wall_time_seconds": tracker.result.wall_time_seconds,
        "co2_kg": tracker.result.emissions_kg,
    }

    print(f"\n{'='*60}")
    print(f"INTERACTION ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {layer}")
    print(f"Features analyzed: {summary['n_analyzed']}")
    print(f"Fraction with >0.75 variance explained: {summary['fraction_above_075']*100:.1f}%")
    print(f"Fraction with >0.50 variance explained: {summary['fraction_above_050']*100:.1f}%")
    print(f"Mean variance explained: {summary['mean_variance_explained']:.4f}")
    print(f"Mean effective rank: {summary['mean_effective_rank']:.2f}")
    print(f"\nPaper claim: {summary['paper_claim']}")
    print(f"Our result: {summary['our_result']}")
    print(f"Claim supported: {summary['claim_supported']}")
    print(f"{'='*60}")

    # Finalize wandb with results
    if wandb_enabled:
        finish_wandb(tracker.result, extra_summary={
            "n_features": n_features,
            "fraction_above_075": fraction_above_075,
            "fraction_above_050": fraction_above_050,
            "mean_variance_explained": summary["mean_variance_explained"],
            "mean_effective_rank": summary["mean_effective_rank"],
            "claim_supported": summary["claim_supported"],
        })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    full_results = {
        "summary": summary,
        "per_feature": {
            "feature_indices": [int(x) for x in results["feature_indices"]],
            "variance_explained": [float(x) for x in results["variance_explained"]],
            "effective_ranks": [float(x) for x in results["effective_ranks"]],
            "truncated_eigenvalues": [float(x) for x in results["truncated_eigenvalues"]],
        },
        "config": config,
        # Root-level emissions for validation script compatibility
        "emissions": {
            "co2_kg": tracker.result.emissions_kg,
            "wall_time_hours": tracker.result.wall_time_hours,
            "gpu_hours": tracker.result.gpu_hours,
        },
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
