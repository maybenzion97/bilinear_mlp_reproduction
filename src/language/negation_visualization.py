"""
Figure 8 Reproduction: Sentiment Negation Circuit Visualization.

This module implements the three panels of Figure 8 from the paper:
- 8A: Interaction submatrix (top 15 interactions)
- 8B: Feature projections onto eigenvectors + meaningful directions
- 8C: Activation vs rank-2 approximation scatter

Paper Description:
"The sentiment negation circuit that computes the not-good and not-bad output features.
A) The interaction submatrix containing the top 15 interactions.
B) The projection of top interacting features onto the top eigenvectors using cosine similarity.
   Clusters coincide with the projection of meaningful directions such as the difference in
   'bad' vs 'good' token unembeddings and the MLP's input activations for '[BOS] not'.
C) The not-good feature activation compared to its approximation by the top two eigenvectors."

Key Features (fw-medium, layer 7, expansion 8):
- Feature 3834: "not-good" (fires on "not lost", "no interference")
- Feature 751: "not-bad" (fires on "not free", "little relief")

Usage:
    python src/language/negation_visualization.py --config configs/language_negation_fw.yaml
"""

import sys
from pathlib import Path
import argparse
import gc
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from language.utils import Sight
from sae.sae import SAE
from sae.tracer import Tracer

from src.utils import (
    get_device,
    load_config,
    track_emissions,
    safe_eigh,
    setup_mps_fallbacks,
    is_mps_device,
)
from src.language.interaction_utils import (
    get_interaction_eigenpairs,
    predict_activation_from_eigenpairs,
)
from src.language.verify_correlation import (
    create_validation_dataloader,
    pearson_correlation,
    cosine_similarity_metric,
    get_metric_function,
)
from src.language.context import LanguageContext
from src.language.memory_efficient_eigen import top_k_eigenvectors_by_magnitude
from src.language.streaming_tracer import q_streaming


# Feature type classification for fw-medium layer 7 input features.
#
# IMPORTANT LIMITATION: fw-medium (trained on FineWeb-EDU, educational/scientific text)
# does not show the same clean semantic clustering as ts-medium (trained on TinyStories).
# The paper's Figure 8 uses ts-medium where input features clearly cluster into:
# - negation: features firing on "not", "never", "wasn't"
# - negative_sentiment: features firing on "bad", "hurt", "sad"  
# - positive_sentiment: features firing on "good", "nice", "happy"
#
# fw-medium's input features show different patterns (structural, domain-specific)
# rather than clear sentiment categories. This is documented here as a known limitation
# of reproducing Figure 8 with fw-medium instead of ts-medium.
#
# Classification key:
# - "negation": potential negation-related (e.g., "ever" in "Have you ever...")
# - "contrast": contrastive conjunction (e.g., "while")
# - "structural": structural patterns (e.g., "and", "of", "many")
# - "other": unclear or domain-specific
FEATURE_TYPES_FW_MEDIUM = {
    "508": "structural",    # fires on "and" in lists
    "582": "structural",    # fires on "series on" / "part of"
    "723": "structural",    # fires on "many" / "many others"
    "751": "other",         # no clear activations (output feature 'not-bad')
    "1202": "other",        # no clear activations
    "2034": "contrast",     # fires on "while" (contrastive)
    "3620": "other",        # no clear activations
    "3834": "other",        # output feature 'not-good' appearing as input
    "4064": "structural",   # fires on "of"
    "4556": "other",        # no clear activations
    "4727": "other",        # fires on "journal" (domain-specific)
    "4898": "other",        # fires on "year" (time-related)
    "5175": "other",        # fires on proper names
    "7198": "negation",     # fires on "ever" ("Have you ever..." - questioning/negation)
    "7369": "other",        # fires on "News" (domain-specific)
}


@dataclass
class Figure8Data:
    """Container for all Figure 8 data."""
    # Panel A: Interaction submatrix
    Q_submatrix: torch.Tensor
    submatrix_feature_indices: List[int]
    
    # Panel B: Eigenvector projections
    feature_projections: Dict[int, Tuple[float, float]]
    meaningful_directions: Dict[str, Tuple[float, float]]
    eigenvector_1: torch.Tensor
    eigenvector_2: torch.Tensor
    
    # Panel C: Scatter data
    z_true: torch.Tensor
    z_pred_rank2: torch.Tensor
    correlation: float
    
    # Metadata
    output_feature_idx: int
    layer: int
    model_name: str


def compute_figure_8a_submatrix(
    tracer: Tracer,
    feat_idx: int,
    top_k: int = 15,
    use_streaming: bool = False,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute the interaction submatrix for Figure 8A.
    
    Finds the top-k strongest interactions in the Q matrix and returns
    the submatrix containing those features.
    
    Memory-efficient approach: Compute Q in d_model space, then compute
    interaction strengths with SAE features to avoid OOM.
    
    Args:
        tracer: Tracer object with loaded SAEs
        feat_idx: Output feature index (e.g., 3834 for "not-good")
        top_k: Number of top interacting features to include
        use_streaming: Use streaming Q computation (reduces memory, enables CUDA)
        chunk_size: Chunk size for streaming (256=1GB, 128=0.5GB)
    
    Returns:
        (Q_submatrix, feature_indices) - The submatrix and the feature indices it contains
    """
    # Get Q matrix in d_model space (NOT projected to avoid OOM)
    # Streaming mode for CUDA: reduces memory from 17GB to ~1GB
    if use_streaming and str(tracer.device) == "cuda":
        Q = q_streaming(tracer, feat_idx, chunk_size=chunk_size).cpu()  # [d_model, d_model]
    else:
        Q = tracer.q(feat_idx, project=False).float().cpu()  # [d_model, d_model]
    
    # Get SAE input decoder directions
    inp_latents = tracer.inp_latents.float().cpu()  # [d_model, n_features]
    
    # Compute interaction strength for each SAE feature pair efficiently
    # For feature i and j: strength = v_i^T Q v_j where v_i, v_j are SAE decoder directions
    # We compute this for all pairs by doing: V^T Q V where V = [v_1, v_2, ..., v_n]
    # But this is still expensive, so we do it in chunks
    
    n_features = inp_latents.shape[1]
    chunk_size = 256  # Process features in chunks to avoid OOM
    
    print(f"  Computing interaction strengths for {n_features} features (chunked)...")
    
    # Compute Q @ V first (can be done efficiently)
    Q_V = torch.mm(Q, inp_latents)  # [d_model, n_features]
    
    # Now compute V^T @ (Q @ V) in chunks
    interaction_strengths = []
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk_latents = inp_latents[:, start:end]  # [d_model, chunk_size]
        chunk_interactions = torch.mm(chunk_latents.T, Q_V)  # [chunk_size, n_features]
        interaction_strengths.append(chunk_interactions)
    
    # Concatenate all chunks
    interaction_matrix = torch.cat(interaction_strengths, dim=0)  # [n_features, n_features]
    
    # Symmetrize
    interaction_matrix = 0.5 * (interaction_matrix + interaction_matrix.T)
    
    # Find top-k interactions by absolute value
    # Look at upper triangle to avoid counting symmetric pairs twice
    n = interaction_matrix.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1)
    triu_values = interaction_matrix[triu_indices[0], triu_indices[1]]
    
    # Get top interactions
    top_interaction_indices = triu_values.abs().topk(min(top_k * 3, len(triu_values))).indices
    
    # Collect unique features from top interactions
    unique_features = set()
    for idx in top_interaction_indices:
        i = triu_indices[0][idx].item()
        j = triu_indices[1][idx].item()
        unique_features.add(i)
        unique_features.add(j)
        if len(unique_features) >= top_k:
            break
    
    # Sort and limit to top_k
    feature_indices = sorted(list(unique_features))[:top_k]
    
    # Extract submatrix
    idx_tensor = torch.tensor(feature_indices)
    Q_submatrix = interaction_matrix[idx_tensor][:, idx_tensor]
    
    print(f"  Selected features: {feature_indices}")
    
    return Q_submatrix, feature_indices


def compute_figure_8b_projections(
    model: Transformer,
    tracer: Tracer,
    feat_idx: int,
    top_features: List[int],
    use_streaming: bool = False,
    chunk_size: int = 256,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, Tuple[float, float]], torch.Tensor, torch.Tensor]:
    """
    Compute projections for Figure 8B.
    
    Projects SAE input features onto top 2 eigenvectors, plus meaningful directions:
    - "bad" - "good" token unembedding
    - "[BOS] not" MLP input activation
    
    Memory-efficient: Uses iterative eigensolver that never materializes the full
    n_features x n_features projected Q matrix. Memory usage is O(d_model x n_features)
    instead of O(n_features^2).
    
    Mathematically equivalent to computing full Q_projected and sorting by eigenvalue magnitude.
    
    Args:
        model: Transformer model
        tracer: Tracer with loaded SAEs
        feat_idx: Output feature index
        top_features: List of SAE input feature indices to project
        use_streaming: Use streaming Q computation (reduces memory, enables CUDA)
        chunk_size: Chunk size for streaming (256=1GB, 128=0.5GB)
    
    Returns:
        (feature_projections, meaningful_directions, v1, v2)
    """
    # Get Q matrix in d_model space (small: 1024x1024 for fw-medium)
    # Streaming mode for CUDA: reduces memory from 17GB to ~1GB
    if use_streaming and str(tracer.device) == "cuda":
        Q = q_streaming(tracer, feat_idx, chunk_size=chunk_size).cpu()  # [d_model, d_model]
    else:
        Q = tracer.q(feat_idx, project=False).float().cpu()  # [d_model, d_model]
    
    # Get SAE input latents (decoder directions)
    inp_latents = tracer.inp_latents.float().cpu()  # [d_model, n_features]
    
    n_features = inp_latents.shape[1]
    d_model = inp_latents.shape[0]
    print(f"  Using memory-efficient eigensolver (avoiding {n_features}×{n_features} matrix)")
    print(f"  Memory: ~{(d_model * n_features * 4) / (1024**2):.0f}MB instead of ~{(n_features**2 * 4) / (1024**2):.0f}MB")
    
    # Use memory-efficient iterative method to get top-2 eigenvectors by magnitude
    # This is mathematically equivalent to:
    #   Q_projected = inp_latents.T @ Q @ inp_latents
    #   eigenvalues, eigenvectors = eigh(Q_projected)
    #   sort by abs(eigenvalues), take top 2
    eigenvalues_sorted, eigenvectors_sorted = top_k_eigenvectors_by_magnitude(
        Q=Q,
        inp_latents=inp_latents,
        k=2,
    )
    
    v1 = eigenvectors_sorted[:, 0]  # Top eigenvector in SAE latent space
    v2 = eigenvectors_sorted[:, 1]  # Second eigenvector
    
    print(f"  Top 2 eigenvalues: λ1={eigenvalues_sorted[0]:.4f}, λ2={eigenvalues_sorted[1]:.4f}")
    
    # v1, v2 are in SAE latent space [n_features]
    # Project features onto these eigenvectors
    feature_projections = {}
    for feat in top_features:
        proj_v1 = v1[feat].item()
        proj_v2 = v2[feat].item()
        feature_projections[feat] = (proj_v1, proj_v2)
    
    # Project eigenvectors back to d_model space for comparison with meaningful directions
    v1_dmodel = torch.mv(inp_latents, v1)  # [d_model]
    v2_dmodel = torch.mv(inp_latents, v2)  # [d_model]
    v1_dmodel = v1_dmodel / v1_dmodel.norm()
    v2_dmodel = v2_dmodel / v2_dmodel.norm()
    
    # Compute meaningful directions (project onto v1, v2 in d_model space)
    meaningful_directions = {}
    
    # 1. "bad" - "good" token unembedding direction
    try:
        tokenizer = model.tokenizer
        good_tokens = tokenizer("good", add_special_tokens=False).input_ids
        bad_tokens = tokenizer("bad", add_special_tokens=False).input_ids
        
        if good_tokens and bad_tokens:
            good_id = good_tokens[0]
            bad_id = bad_tokens[0]
            
            # Get unembedding vectors (in d_model space)
            # model.w_u is lm_head.weight with shape [vocab_size, d_model]
            good_unembed = model.w_u[good_id, :].float().cpu()
            bad_unembed = model.w_u[bad_id, :].float().cpu()
            sentiment_dir = bad_unembed - good_unembed
            sentiment_dir = sentiment_dir / sentiment_dir.norm()
            
            # Project onto v1_dmodel, v2_dmodel (which are in d_model space)
            proj_v1 = F.cosine_similarity(sentiment_dir.unsqueeze(0), v1_dmodel.unsqueeze(0)).item()
            proj_v2 = F.cosine_similarity(sentiment_dir.unsqueeze(0), v2_dmodel.unsqueeze(0)).item()
            meaningful_directions["bad-good"] = (proj_v1, proj_v2)
    except Exception as e:
        print(f"  Warning: Could not compute 'bad-good' direction: {e}")
    
    # 2. "[BOS] not" MLP input activation
    try:
        sight = Sight(model)
        device = next(model.parameters()).device
        
        # Tokenize "[BOS] not" 
        tokens = tokenizer("[BOS] not", return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            with sight.trace(tokens, validate=False, scan=False):
                not_activation = sight["mlp-in", tracer.layer].save()
        
        # Get activation at the "not" position (last token)
        not_dir = not_activation[0, -1].float().cpu()
        not_dir = not_dir / not_dir.norm()
        
        # Project onto v1_dmodel, v2_dmodel (which are in d_model space)
        proj_v1 = F.cosine_similarity(not_dir.unsqueeze(0), v1_dmodel.unsqueeze(0)).item()
        proj_v2 = F.cosine_similarity(not_dir.unsqueeze(0), v2_dmodel.unsqueeze(0)).item()
        meaningful_directions["[BOS] not"] = (proj_v1, proj_v2)
    except Exception as e:
        print(f"  Warning: Could not compute '[BOS] not' direction: {e}")
    
    return feature_projections, meaningful_directions, v1_dmodel, v2_dmodel


def compute_figure_8c_scatter(
    model: Transformer,
    sae_out: SAE,
    layer: int,
    feat_idx: int,
    dataloader,
    device: str,
    max_batches: int = 50,
    metric: str = "pearson",
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute true vs predicted activations for Figure 8C.
    
    Memory-efficient version: only accumulates active samples (z_true > 0)
    to avoid OOM from storing all batch data.
    
    Reuses existing functions from interaction_utils.py for DRY compliance.
    
    Args:
        model: Transformer model
        sae_out: Output SAE
        layer: Layer index
        feat_idx: Feature index (3834 for "not-good")
        dataloader: Validation dataloader
        device: Device for computation
        max_batches: Maximum batches to process
        metric: Similarity metric - 'pearson' or 'cosine' (default: 'pearson')
    
    Returns:
        (z_true, z_pred_rank2, correlation)
    """
    metric_fn = get_metric_function(metric)
    metric_label = "cosine similarity" if metric == "cosine" else "Pearson correlation"
    print(f"  Using memory-efficient Panel C (only accumulating active samples)")
    print(f"  Metric: {metric_label}")
    
    sight = Sight(model)
    sae_out_cpu = sae_out.cpu()  # Keep SAE on CPU to save memory
    
    # First, compute eigenpairs ONCE (before accumulating data)
    # This is the Q matrix for the UNPROJECTED model weights, only d_model x d_model
    print(f"  Pre-computing rank-2 eigenpairs for feature {feat_idx}...")
    out_direction = sae_out_cpu.w_enc.weight[feat_idx, :]
    eigenpairs = get_interaction_eigenpairs(
        model=model,
        layer=layer,
        feat_idx=feat_idx,
        out_direction=out_direction,
        device="cpu",
    )
    eigenvalues_2 = eigenpairs.eigenvalues[:2]
    eigenvectors_2 = eigenpairs.eigenvectors[:, :2]
    print(f"  Rank-2 eigenvalues: {eigenvalues_2.tolist()}")
    
    # Memory-efficient accumulation: ONLY keep active samples
    all_x_active = []
    all_z_active = []
    total_samples = 0
    active_count = 0
    
    # Check if we're on MPS
    is_mps = str(device).startswith("mps")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Figure 8C", total=min(max_batches, len(dataloader)))):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with sight.trace(batch, validate=False, scan=False):
                mlp_in = sight["mlp-in", layer].save()
                mlp_out = sight["mlp-out", layer].save()
            
            # Move to CPU immediately to free MPS memory
            mlp_in_cpu = mlp_in.flatten(0, 1).float().cpu()
            mlp_out_cpu = mlp_out.flatten(0, 1).float().cpu()
            
            # Delete MPS tensors immediately
            del mlp_in, mlp_out
            
            # Compute z_true on CPU (SAE encode is fast enough)
            z_true_batch = sae_out_cpu.encode(mlp_out_cpu)[:, feat_idx]
            
            # Find active samples in THIS batch (z_true > 0)
            active_mask = z_true_batch > 0
            n_active = active_mask.sum().item()
            
            if n_active > 0:
                # Only keep active samples to save memory
                all_x_active.append(mlp_in_cpu[active_mask].clone())
                all_z_active.append(z_true_batch[active_mask].clone())
                active_count += n_active
            
            total_samples += mlp_in_cpu.shape[0]
            
            # Explicitly delete tensors and clear memory
            del mlp_in_cpu, mlp_out_cpu, z_true_batch, active_mask, batch
            gc.collect()
            
            # Clear MPS cache if on Apple Silicon
            if is_mps:
                torch.mps.empty_cache()
    
    if len(all_x_active) == 0:
        print(f"  Warning: No active samples found for feature {feat_idx}")
        return torch.tensor([]), torch.tensor([]), 0.0
    
    # Concatenate only active samples
    x_active = torch.cat(all_x_active, dim=0)
    z_active = torch.cat(all_z_active, dim=0)
    
    print(f"  Found {active_count}/{total_samples} active samples ({100*active_count/total_samples:.1f}%)")
    print(f"  Memory saved: kept {active_count} instead of {total_samples} samples")
    
    # Rank-2 prediction using pre-computed eigenpairs
    z_pred = predict_activation_from_eigenpairs(x_active, eigenvalues_2, eigenvectors_2)
    
    # Compute metric (pearson or cosine)
    corr = metric_fn(z_active, z_pred)
    
    return z_active, z_pred, corr


def generate_figure_8_data(
    model: Transformer,
    tracer: Tracer,
    sae_out: SAE,
    layer: int,
    feat_idx: int,
    dataloader,
    device: str,
    top_k_interactions: int = 15,
    max_batches: int = 50,
    use_streaming: bool = False,
    chunk_size: int = 256,
    metric: str = "pearson",
) -> Figure8Data:
    """
    Generate all data needed for Figure 8.
    
    Args:
        model: Transformer model
        tracer: Tracer with loaded SAEs
        sae_out: Output SAE
        layer: Layer index
        feat_idx: Output feature index (3834 for "not-good")
        dataloader: Validation dataloader
        device: Device for computation
        top_k_interactions: Number of top interactions for Panel A
        max_batches: Maximum batches for Panel C
        use_streaming: Use streaming Q computation (reduces memory, enables CUDA)
        chunk_size: Chunk size for streaming (256=1GB, 128=0.5GB)
        metric: Similarity metric - 'pearson' or 'cosine' (default: 'pearson')
    
    Returns:
        Figure8Data containing all panel data
    """
    metric_label = "cosine similarity" if metric == "cosine" else "Pearson correlation"
    print(f"\nGenerating Figure 8 data for feature {feat_idx}...")
    print(f"  Metric: {metric_label}")
    if use_streaming:
        print(f"  Using streaming Q computation (chunk_size={chunk_size})")
    
    # Panel A: Interaction submatrix
    print("  Computing Panel A (interaction submatrix)...")
    Q_submatrix, submatrix_features = compute_figure_8a_submatrix(
        tracer, feat_idx, top_k=top_k_interactions,
        use_streaming=use_streaming, chunk_size=chunk_size,
    )
    
    # Panel B: Eigenvector projections
    print("  Computing Panel B (eigenvector projections)...")
    feature_projections, meaningful_directions, v1, v2 = compute_figure_8b_projections(
        model, tracer, feat_idx, submatrix_features,
        use_streaming=use_streaming, chunk_size=chunk_size,
    )
    
    # Panel C: Activation vs approximation
    print("  Computing Panel C (activation scatter)...")
    z_true, z_pred, corr = compute_figure_8c_scatter(
        model, sae_out, layer, feat_idx, dataloader, device, max_batches, metric
    )
    
    return Figure8Data(
        Q_submatrix=Q_submatrix,
        submatrix_feature_indices=submatrix_features,
        feature_projections=feature_projections,
        meaningful_directions=meaningful_directions,
        eigenvector_1=v1,
        eigenvector_2=v2,
        z_true=z_true,
        z_pred_rank2=z_pred,
        correlation=corr,
        output_feature_idx=feat_idx,
        layer=layer,
        model_name=model.config.repo,
    )


def save_figure_8_data(data: Figure8Data, output_path: Path, feature_types: Optional[Dict[str, str]] = None, emissions: Optional[dict] = None, metric: str = "pearson"):
    """Save Figure 8 data to JSON.
    
    Args:
        data: Figure8Data object with all panel data
        output_path: Path to save JSON file
        feature_types: Optional dict mapping feature indices to semantic types
                       (e.g., {"508": "structural", "7198": "negation"})
        emissions: Optional dict with emissions data (co2_kg, wall_time_hours, gpu_hours)
    """
    # Get feature types for the features in this data
    if feature_types is None:
        feature_types = FEATURE_TYPES_FW_MEDIUM
    
    # Filter to only include features that are in the data
    feature_indices = [str(f) for f in data.submatrix_feature_indices]
    relevant_feature_types = {
        f: feature_types.get(f, "other") for f in feature_indices
    }
    
    metric_label = "Cosine similarity" if metric == "cosine" else "Pearson correlation"
    results = {
        "output_feature_idx": data.output_feature_idx,
        "layer": data.layer,
        "model_name": data.model_name,
        "metric": metric,
        "metric_label": metric_label,
        "panel_a": {
            "Q_submatrix": data.Q_submatrix.tolist(),
            "feature_indices": data.submatrix_feature_indices,
        },
        "panel_b": {
            "feature_projections": {str(k): list(v) for k, v in data.feature_projections.items()},
            "meaningful_directions": {k: list(v) for k, v in data.meaningful_directions.items()},
        },
        "panel_c": {
            "z_true": data.z_true.tolist(),
            "z_pred_rank2": data.z_pred_rank2.tolist(),
            "correlation": data.correlation,
            "metric": metric,
            "metric_label": metric_label,
            "n_samples": len(data.z_true),
        },
        # Include feature type classification for visualization
        "feature_types": relevant_feature_types,
    }
    
    # Add emissions data if provided
    if emissions is not None:
        results["emissions"] = emissions
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFigure 8 data saved to: {output_path}")
    print(f"Feature types included: {relevant_feature_types}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 8 (Sentiment Negation Circuit)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/figure_8_data.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature", type=int, default=3834, 
                        help="Output feature index (default: 3834 'not-good')")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for validation dataloader (default: 16)")
    parser.add_argument("--top-k", type=int, default=15, 
                        help="Number of top interactions for Panel A")
    parser.add_argument("--n-samples", type=str, default="2000", 
                        help="Number of validation samples ('all' or integer)")
    parser.add_argument("--max-batches", type=int, default=50, 
                        help="Maximum batches for Panel C")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming Q computation (reduces memory, enables CUDA)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Chunk size for streaming (256=1GB, 128=0.5GB)")
    parser.add_argument("--metric", type=str, default="pearson", choices=["pearson", "cosine"],
                        help="Similarity metric: 'pearson' (default) or 'cosine'")
    parser.add_argument("--dataset", type=str, default=None, choices=["tinystories", "fineweb", "fineweb-16k"],
                        help="Validation dataset override ('tinystories', 'fineweb', or 'fineweb-16k'). "
                             "Defaults to model-appropriate dataset.")
    args = parser.parse_args()
    
    # Adjust output path for cosine metric - put in cosine/ subfolder
    if args.metric == "cosine":
        output_path = Path(args.output)
        cosine_dir = output_path.parent / "cosine"
        args.output = str(cosine_dir / output_path.name)
        print(f"Cosine metric: output will be saved to {args.output}")
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
    
    # Load config
    config = load_config(args.config)
    
    # Parse n_samples - support 'all' or integer
    if args.n_samples.lower() == "all" or args.n_samples == "-1":
        n_samples = -1  # -1 means all samples from validation set
    else:
        n_samples = int(args.n_samples)
    
    # Run with emissions tracking
    with track_emissions("bilinear-mlp-reproduction") as tracker:
        # Use LanguageContext for unified model/SAE/Tracer loading
        ctx = LanguageContext(config, device)
        model = ctx.model
        layer = ctx.layer
        
        # Create Tracer via context
        tracer = ctx.get_tracer()
        
        # Load output SAE via context
        sae_out = ctx.get_sae("mlp-out")
        
        # Create validation dataloader via context
        model_name = ctx.model_name.lower()
        if args.dataset is not None:
            dataset_name = args.dataset
        else:
            dataset_name = "fineweb" if "fw-" in model_name else "tinystories"
        dataloader = ctx.get_dataloader(
            n_samples=n_samples,
            batch_size=args.batch_size,
            dataset_name=dataset_name,
        )
        
        # Generate Figure 8 data
        figure_data = generate_figure_8_data(
            model=model,
            tracer=tracer,
            sae_out=sae_out,
            layer=layer,
            feat_idx=args.feature,
            dataloader=dataloader,
            device=device,
            top_k_interactions=args.top_k,
            max_batches=args.max_batches,
            use_streaming=args.streaming,
            chunk_size=args.chunk_size,
            metric=args.metric,
        )
    
    # Access emissions AFTER exiting the context manager (tracker has stopped)
    if tracker.result is not None:
        emissions = {
            "co2_kg": tracker.result.emissions_kg,
            "wall_time_hours": tracker.result.wall_time_hours,
            "gpu_hours": tracker.result.gpu_hours,
        }
    else:
        # Fallback if tracker failed
        emissions = {
            "co2_kg": 0.0,
            "wall_time_hours": 0.0,
            "gpu_hours": 0.0,
        }
        print("Warning: CodeCarbon tracker failed, emissions not recorded")
    
    save_figure_8_data(figure_data, Path(args.output), emissions=emissions, metric=args.metric)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FIGURE 8 GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Feature: {args.feature}")
    print(f"Panel A: {len(figure_data.submatrix_feature_indices)} features in submatrix")
    print(f"Panel B: {len(figure_data.feature_projections)} feature projections")
    print(f"         {len(figure_data.meaningful_directions)} meaningful directions")
    print(f"Panel C: {len(figure_data.z_true)} scatter points, correlation={figure_data.correlation:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
