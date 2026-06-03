"""
Correlation Verification for Figure 9 (Section 5).

This script verifies that weight-based eigenvectors actually predict SAE activations
on real data. It computes the correlation between:

1. True Activations: z_true = SAE.encode(mlp_out)
2. Predicted Activations: z_pred = sum_{j=1}^k lambda_j * (v_j^T x)^2

Where:
- x is the input to the bilinear layer (mlp_in, in residual stream space)
- v_j are eigenvectors of the interaction matrix Q
- lambda_j are corresponding eigenvalues

Paper claim: Rank-2 approximation captures >75% of variance (Figure 9).

Usage:
    python src/language/verify_correlation.py --config configs/language_interaction.yaml
    python src/language/verify_correlation.py --config configs/language_interaction.yaml --n-features 50 --ranks 1,2,4,8,16
"""

import sys
from pathlib import Path
import argparse
import json
import warnings

# Suppress torchvision image extension warning
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from language.utils import Sight
from sae.sae import SAE

from src.utils import (
    get_device,
    load_config,
    track_emissions,
    init_wandb,
    finish_wandb,
    setup_mps_fallbacks,
    is_mps_device,
)
from src.language.interaction_utils import (
    get_interaction_eigenpairs,
    get_interaction_eigenpairs_streaming,
    predict_activation_from_eigenpairs,
    InteractionEigenpairs,
)
from src.language.context import LanguageContext
from src.paths import LANGUAGE_EIGENPAIRS


def load_cached_eigenpairs(cache_dir: Path, feat_idx: int) -> InteractionEigenpairs:
    """
    Load precomputed eigenpairs from cache.
    
    Args:
        cache_dir: Directory containing cached eigenpair files
        feat_idx: Feature index to load
    
    Returns:
        InteractionEigenpairs object
    
    Raises:
        FileNotFoundError: If the cached file doesn't exist
    """
    cache_file = cache_dir / f"feat_{feat_idx:04d}.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cached eigenpairs not found: {cache_file}")
    
    data = torch.load(cache_file, map_location='cpu', weights_only=False)
    
    return InteractionEigenpairs(
        eigenvalues=data['eigenvalues'],
        eigenvectors=data['eigenvectors'],
        Q_matrix=torch.zeros(1),  # Q not cached, placeholder
        sort_indices=None,
    )


def resolve_eigenpairs_cache(load_eigenpairs: str, model_name: str, layer: int) -> Path:
    """
    Resolve the eigenpairs cache directory from user input.
    
    Args:
        load_eigenpairs: User-provided path or "auto"
        model_name: Model name (e.g., "tdooms/fw-medium")
        layer: Layer index
    
    Returns:
        Path to the cache directory
    
    Raises:
        FileNotFoundError: If auto-detection fails
    """
    if load_eigenpairs == "auto":
        # Auto-detect from model name and layer
        model_short = model_name.split("/")[-1]
        cache_dir = LANGUAGE_EIGENPAIRS / model_short / str(layer)
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Auto-detected cache not found: {cache_dir}\n"
                f"Run precomputation first:\n"
                f"  python src/language/precompute_eigenpairs.py --model {model_short} --layer {layer}"
            )
        return cache_dir
    else:
        cache_dir = Path(load_eigenpairs)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Eigenpairs cache not found: {cache_dir}")
        return cache_dir


def plot_correlation_vs_rank(summary: dict, output_path: str):
    """
    Generate a plot of Average Correlation vs Rank.
    
    Args:
        summary: Summary dict containing correlation_by_rank
        output_path: Path to save the PNG plot
    """
    ranks = []
    means = []
    stds = []
    
    for rank_str, stats in sorted(summary["correlation_by_rank"].items(), key=lambda x: int(x[0])):
        ranks.append(int(rank_str))
        means.append(stats["mean"])
        stds.append(stats["std"])
    
    ranks = np.array(ranks)
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot with error bars
    ax.errorbar(ranks, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
                linewidth=2, markersize=8, color='#2E86AB', ecolor='#A23B72')
    
    # Reference line at 0.75 (paper claim)
    ax.axhline(y=0.75, color='#F18F01', linestyle='--', linewidth=1.5, 
               label='Paper claim (75%)')
    
    ax.set_xlabel('Rank (k)', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Weight-Based Prediction vs True SAE Activation\n(Figure 9 Reproduction)', fontsize=14)
    ax.set_xticks(ranks)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Add annotation for rank-2
    if 2 in ranks:
        idx = list(ranks).index(2)
        ax.annotate(f'Rank-2: {means[idx]:.3f}', 
                    xy=(2, means[idx]), xytext=(2.5, means[idx] - 0.15),
                    fontsize=10, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Pearson correlation coefficient between two tensors.
    
    Args:
        x: First tensor [n]
        y: Second tensor [n]
    
    Returns:
        Correlation coefficient (float)
    """
    x = x.float()
    y = y.float()
    
    # Handle edge cases
    if len(x) < 2:
        return float('nan')
    
    x_mean = x.mean()
    y_mean = y.mean()
    
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    numerator = (x_centered * y_centered).sum()
    denominator = (x_centered.pow(2).sum().sqrt() * y_centered.pow(2).sum().sqrt())
    
    if denominator < 1e-10:
        return float('nan')
    
    return (numerator / denominator).item()


def cosine_similarity_metric(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute cosine similarity between two 1D tensors.
    
    Unlike Pearson correlation, cosine similarity does NOT center the vectors.
    This measures the angle between vectors in the original space.
    
    Args:
        x: First tensor [n]
        y: Second tensor [n]
    
    Returns:
        Cosine similarity (float in [-1, 1])
    """
    import torch.nn.functional as F
    
    x = x.float()
    y = y.float()
    
    # Handle edge cases
    if len(x) < 2:
        return float('nan')
    
    # Compute cosine similarity
    x_norm = x.norm()
    y_norm = y.norm()
    
    if x_norm < 1e-10 or y_norm < 1e-10:
        return float('nan')
    
    return (x @ y / (x_norm * y_norm)).item()


def get_metric_function(metric: str):
    """
    Get the metric function based on metric name.
    
    Args:
        metric: 'pearson' or 'cosine'
    
    Returns:
        Metric function that takes (x, y) tensors and returns float
    """
    if metric == "pearson":
        return pearson_correlation
    elif metric == "cosine":
        return cosine_similarity_metric
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'pearson' or 'cosine'.")


def create_validation_dataloader(
    tokenizer, 
    config: dict, 
    device: str, 
    n_samples: int = 2000, 
    batch_size: int = 32,
    dataset_name: str = "tinystories",
):
    """
    Create a DataLoader for validation data from TinyStories or FineWeb dataset.
    
    Args:
        tokenizer: Model tokenizer
        config: Experiment config
        device: Device to use
        n_samples: Number of samples to load (-1 for all)
        batch_size: Batch size for DataLoader
        dataset_name: "tinystories" or "fineweb"
    
    Returns:
        DataLoader yielding batches with input_ids and attention_mask
    """
    import itertools
    from datasets import Dataset
    
    n_ctx = config.get("sae", {}).get("n_ctx", 256)
    
    if dataset_name == "fineweb":
        print("Loading FineWeb-Edu validation data (streaming)...")
        
        # FineWeb is large, use streaming
        ds_stream = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            "sample-10BT",
            split="train", 
            streaming=True
        )
        
        # Collect samples from stream
        if n_samples > 0:
            samples = list(itertools.islice(ds_stream, n_samples))
        else:
            # Load a reasonable amount for "all" - FineWeb is huge
            samples = list(itertools.islice(ds_stream, 50000))
        
        # Convert to Dataset
        dataset = Dataset.from_list(samples)
        print(f"  Loaded {len(dataset)} samples from FineWeb-Edu")
        
    else:  # tinystories (default)
        print("Loading TinyStories validation data...")
        
        # Load dataset (use validation split if available, else sample from train)
        try:
            dataset = load_dataset("roneneldan/TinyStories", split="validation")
        except Exception:
            dataset = load_dataset("roneneldan/TinyStories", split="train")
        
        # Sample if dataset is larger than needed (-1 means use all)
        if n_samples > 0 and len(dataset) > n_samples:
            dataset = dataset.select(range(n_samples))
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=n_ctx,
            padding="max_length",
            return_tensors="pt",
        )
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    print(f"  Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader


def find_active_features(sae_activations: torch.Tensor, min_active: int = 10) -> list:
    """
    Find features that are active in the batch.
    
    Args:
        sae_activations: SAE activations [batch*seq, d_features]
        min_active: Minimum number of active positions required
    
    Returns:
        List of feature indices that are sufficiently active
    """
    # Count active positions per feature (activation > 0)
    active_counts = (sae_activations > 0).sum(dim=0)  # [d_features]
    
    # Find features with enough active positions
    active_features = (active_counts >= min_active).nonzero(as_tuple=True)[0]
    
    return active_features.tolist()


def estimate_total_tokens(dataloader, max_batches: int, batch_size: int, n_ctx: int):
    """
    Estimate total tokens processed for a run.
    
    Args:
        dataloader: DataLoader used for evaluation
        max_batches: Max batches to process
        batch_size: Batch size used by DataLoader
        n_ctx: Sequence length
    
    Returns:
        Tuple of (estimated_tokens or None, total_batches_used)
    """
    try:
        total_batches = min(max_batches, len(dataloader))
    except TypeError:
        total_batches = max_batches
    
    if batch_size is None or n_ctx is None:
        return None, total_batches
    
    return total_batches * batch_size * n_ctx, total_batches


def _select_active_features_exact(
    sae_out: SAE,
    sight: Sight,
    dataloader,
    layer: int,
    feature_indices: list,
    min_active_per_feature: int,
    device: str,
    max_batches: int,
    total_batches: int,
) -> list:
    active_counts = torch.zeros(sae_out.d_features, dtype=torch.int64)
    feature_indices_tensor = None
    if feature_indices:
        feature_indices_tensor = torch.tensor(feature_indices, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Exact pass 1/2: counting actives", total=total_batches)
        ):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with sight.trace(batch, validate=False, scan=False):
                mlp_out = sight["mlp-out", layer].save()
            
            mlp_out_flat = mlp_out.flatten(0, 1).float()
            z_true = sae_out.encode(mlp_out_flat)
            
            if feature_indices_tensor is not None:
                z_true_sel = z_true.index_select(1, feature_indices_tensor)
                active_counts[feature_indices_tensor.cpu()] += (z_true_sel > 0).sum(dim=0).cpu()
            else:
                active_counts += (z_true > 0).sum(dim=0).cpu()
    
    if feature_indices:
        return [f for f in feature_indices if active_counts[f] >= min_active_per_feature]
    
    active_mask = active_counts >= min_active_per_feature
    return active_mask.nonzero(as_tuple=True)[0].tolist()


def _build_eigenpairs_for_chunk(
    chunk: list,
    model,
    sae_out: SAE,
    layer: int,
    use_streaming: bool,
    chunk_size: int,
    device: str,
    eigenpairs_cache_dir: Path,
    k_max: int,
) -> dict:
    eigenpairs_by_feature = {}
    for feat_idx in tqdm(chunk, desc="Precomputing eigenpairs (chunk)"):
        if eigenpairs_cache_dir is not None:
            try:
                eigenpairs = load_cached_eigenpairs(eigenpairs_cache_dir, feat_idx)
            except FileNotFoundError:
                continue
        else:
            out_direction = sae_out.w_enc.weight[feat_idx, :]
            if use_streaming:
                eigenpairs = get_interaction_eigenpairs_streaming(
                    model=model,
                    layer=layer,
                    feat_idx=feat_idx,
                    out_direction=out_direction,
                    chunk_size=chunk_size,
                    compute_device=device,
                )
            else:
                eigenpairs = get_interaction_eigenpairs(
                    model=model,
                    layer=layer,
                    feat_idx=feat_idx,
                    out_direction=out_direction.cpu(),
                    device="cpu",
                )
        
        eigenpairs_by_feature[feat_idx] = (
            eigenpairs.eigenvalues[:k_max].cpu(),
            eigenpairs.eigenvectors[:, :k_max].cpu(),
        )
    return eigenpairs_by_feature


def _init_exact_accumulators(chunk: list, rank_count: int, save_scatter: bool):
    accumulators = {}
    scatter_buffers = {}
    for feat_idx in chunk:
        accumulators[feat_idx] = {
            "count": 0,
            "sum_z": 0.0,
            "sum_z2": 0.0,
            "sum_pred": torch.zeros(rank_count, dtype=torch.float64),
            "sum_pred2": torch.zeros(rank_count, dtype=torch.float64),
            "sum_z_pred": torch.zeros(rank_count, dtype=torch.float64),
        }
        if save_scatter:
            scatter_buffers[feat_idx] = {"z_true": [], "z_pred": []}
    return accumulators, scatter_buffers


def _accumulate_exact_chunk(
    dataloader,
    sight: Sight,
    sae_out: SAE,
    layer: int,
    chunk: list,
    chunk_tensor: torch.Tensor,
    eigenpairs_by_feature: dict,
    rank_indices: torch.Tensor,
    ranks: list,
    max_batches: int,
    total_batches: int,
    device: str,
    save_scatter: bool,
    max_scatter_samples: int,
    count_tokens: bool,
) -> tuple:
    accumulators, scatter_buffers = _init_exact_accumulators(chunk, len(ranks), save_scatter)
    rank2_idx = ranks.index(2) if save_scatter and 2 in ranks else None
    tokens_counted = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Exact pass 2/2: accumulating (chunk)", total=total_batches)
        ):
            if batch_idx >= max_batches:
                break
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with sight.trace(batch, validate=False, scan=False):
                mlp_in = sight["mlp-in", layer].save()
                mlp_out = sight["mlp-out", layer].save()
            
            mlp_in_flat = mlp_in.flatten(0, 1).float()
            mlp_out_flat = mlp_out.flatten(0, 1).float()
            z_true = sae_out.encode(mlp_out_flat)
            
            if count_tokens:
                tokens_counted += mlp_in_flat.shape[0]
            
            mlp_in_cpu = mlp_in_flat.cpu()
            z_true_selected = z_true.index_select(1, chunk_tensor).cpu()
            
            for idx, feat_idx in enumerate(chunk):
                if feat_idx not in eigenpairs_by_feature:
                    continue
                
                z_feat = z_true_selected[:, idx]
                active_mask = z_feat > 0
                if not active_mask.any():
                    continue
                
                x_active = mlp_in_cpu[active_mask]
                z_active = z_feat[active_mask].float()
                
                if x_active.shape[0] == 0:
                    continue
                
                eigenvalues_k, eigenvectors_k = eigenpairs_by_feature[feat_idx]
                
                projections = x_active @ eigenvectors_k
                weighted = projections.square() * eigenvalues_k
                cumsum = weighted.cumsum(dim=1)
                z_pred_by_rank = cumsum[:, rank_indices]
                
                z_active_f64 = z_active.double()
                z_pred_f64 = z_pred_by_rank.double()
                
                acc = accumulators[feat_idx]
                acc["count"] += z_active_f64.shape[0]
                acc["sum_z"] += z_active_f64.sum().item()
                acc["sum_z2"] += (z_active_f64 ** 2).sum().item()
                acc["sum_pred"] += z_pred_f64.sum(dim=0)
                acc["sum_pred2"] += (z_pred_f64 ** 2).sum(dim=0)
                acc["sum_z_pred"] += (z_pred_f64 * z_active_f64.unsqueeze(1)).sum(dim=0)
                
                if rank2_idx is not None:
                    z_pred_rank2 = z_pred_by_rank[:, rank2_idx]
                    buffer = scatter_buffers[feat_idx]
                    remaining = max_scatter_samples - len(buffer["z_true"])
                    if remaining > 0:
                        take = min(remaining, z_active.shape[0])
                        buffer["z_true"].extend(z_active[:take].cpu().tolist())
                        buffer["z_pred"].extend(z_pred_rank2[:take].cpu().tolist())
    
    return accumulators, scatter_buffers, tokens_counted


def _compute_metric_from_sums(
    metric: str,
    n_active: int,
    sum_z: float,
    sum_z2: float,
    sum_pred: float,
    sum_pred2: float,
    sum_z_pred: float,
) -> float:
    if n_active <= 1:
        return float("nan")
    
    if metric == "cosine":
        denom = (sum_z2 ** 0.5) * (sum_pred2 ** 0.5)
        return float("nan") if denom < 1e-10 else sum_z_pred / denom
    
    mean_z = sum_z / n_active
    mean_pred = sum_pred / n_active
    cov = sum_z_pred - n_active * mean_z * mean_pred
    var_z = sum_z2 - n_active * (mean_z ** 2)
    var_pred = sum_pred2 - n_active * (mean_pred ** 2)
    # Numerical safety: clamp negative variances from round-off to zero
    if var_z < 0:
        var_z = 0.0
    if var_pred < 0:
        var_pred = 0.0
    denom = (var_z ** 0.5) * (var_pred ** 0.5)
    return float("nan") if denom < 1e-10 else cov / denom


def _finalize_exact_chunk(
    chunk: list,
    accumulators: dict,
    scatter_buffers: dict,
    results: dict,
    feature_results: list,
    scatter_files_saved: list,
    ranks: list,
    metric: str,
    scatter_dir: Path,
    model_name: str,
    max_scatter_samples: int,
    min_active_per_feature: int,
):
    for feat_idx in chunk:
        acc = accumulators[feat_idx]
        n_active = acc["count"]
        if n_active < min_active_per_feature:
            continue
        
        feature_corrs = {}
        sum_z = acc["sum_z"]
        sum_z2 = acc["sum_z2"]
        
        for rank_idx, k in enumerate(ranks):
            sum_pred = acc["sum_pred"][rank_idx].item()
            sum_pred2 = acc["sum_pred2"][rank_idx].item()
            sum_z_pred = acc["sum_z_pred"][rank_idx].item()
            
            corr = _compute_metric_from_sums(
                metric,
                n_active,
                sum_z,
                sum_z2,
                sum_pred,
                sum_pred2,
                sum_z_pred,
            )
            
            if not np.isnan(corr):
                results[k].append(corr)
                feature_corrs[k] = corr
        
        result_dict = {
            "feat_idx": feat_idx,
            "n_active": int(n_active),
            "correlations": feature_corrs,
        }
        
        if scatter_dir is not None and 2 in ranks:
            buffer = scatter_buffers.get(feat_idx, None)
            if buffer is not None and buffer["z_true"]:
                scatter_data = {
                    "feat_idx": feat_idx,
                    "n_active": int(n_active),
                    "z_true": buffer["z_true"][:max_scatter_samples],
                    "z_pred_rank2": buffer["z_pred"][:max_scatter_samples],
                    "correlation_rank2": feature_corrs.get(2, None),
                }
                scatter_file = scatter_dir / f"scatter_{model_name}_feat_{feat_idx}.json"
                with open(scatter_file, "w") as f:
                    json.dump(scatter_data, f)
                scatter_files_saved.append(str(scatter_file))
                result_dict["scatter_file"] = str(scatter_file)
        
        feature_results.append(result_dict)


def verify_correlation(
    model,
    sae_out: SAE,
    layer: int,
    dataloader,
    feature_indices: list,
    ranks: list,
    device: str,
    min_active_per_feature: int = 50,
    target_samples_per_feature: int = 500,
    max_batches: int = 50,
    max_features: int = -1,
    save_scatter: bool = False,
    max_scatter_samples: int = 1000,
    scatter_dir: Path = None,
    model_name: str = "unknown",
    use_streaming: bool = True,
    chunk_size: int = 256,
    metric: str = "pearson",
    eigenpairs_cache_dir: Path = None,
    batch_size: int = 32,
    n_ctx: int = 256,
    streaming_threshold_tokens: int = 500_000,
    streaming_max_samples: int = 2000,
    streaming_memory_gb: float = 2.0,
    force_full_accum: bool = False,
    discovery_batches: int = 20,
    exact_streaming: bool = False,
    exact_chunk_size: int = 64,
):
    """
    Compute correlation between weight-based predictions and actual SAE activations.
    
    CRITICAL FIX: Accumulates samples across multiple batches to ensure statistical
    significance. The correlation is computed on the full accumulated data, not per-batch.
    
    Memory-efficient scatter: When save_scatter=True, scatter data is saved to individual
    files immediately and freed from memory. This allows processing many features without OOM.
    
    GPU-accelerated Q computation: When use_streaming=True, uses memory-efficient streaming
    computation that runs on GPU (cuda/mps) for 5-10x speedup over CPU-only computation.
    
    The key equation being verified:
        z_c(x) ≈ sum_{j=1}^k lambda_j * (v_j^T x)^2
    
    Args:
        model: Transformer model
        sae_out: Output SAE (for encoding mlp_out)
        layer: Layer index
        dataloader: DataLoader yielding validation batches
        feature_indices: List of feature indices to analyze (or None for auto-detect)
        ranks: List of ranks to evaluate [1, 2, 4, 8, 16]
        device: Device for computation
        min_active_per_feature: Minimum active samples required per feature
        target_samples_per_feature: Target number of active samples before computing correlation
        max_batches: Maximum number of batches to process
        max_features: Maximum number of features to analyze (-1 for all)
        save_scatter: Whether to save scatter data for Figure 9C
        max_scatter_samples: Maximum scatter samples per feature
        scatter_dir: Directory to save scatter files (created if save_scatter=True)
        model_name: Model name for scatter file naming
        use_streaming: Whether to use GPU-accelerated streaming Q computation (default True)
        chunk_size: Chunk size for streaming computation (default 256)
        metric: Similarity metric to use - 'pearson' or 'cosine' (default: 'pearson')
        eigenpairs_cache_dir: If provided, load precomputed eigenpairs from this directory
            instead of computing them on-the-fly. This enables rapid iteration with
            different metrics/thresholds without recomputation.
        batch_size: Batch size used by the dataloader (for token estimation)
        n_ctx: Sequence length used in tokenization (for token estimation)
        streaming_threshold_tokens: Auto-switch to streaming accumulation when
            estimated tokens exceed this threshold.
        streaming_max_samples: Cap per-feature samples in streaming mode.
        streaming_memory_gb: Approximate memory budget for per-feature samples.
        force_full_accum: If True, disable auto streaming accumulation.
        discovery_batches: Batches to scan for active features in streaming mode.
        exact_streaming: If True, use an exact streaming correlation pass without
            capping samples/features (memory-safe but slower).
        exact_chunk_size: Number of features per chunk in exact streaming mode.
            Smaller chunks reduce memory but increase runtime.
    
    Returns:
        Dict with correlation results per rank
    """
    # Get the metric function
    metric_fn = get_metric_function(metric)
    
    results = {k: [] for k in ranks}
    feature_results = []
    scatter_files_saved = []
    
    # Create scatter directory if saving scatter data
    if save_scatter and scatter_dir is not None:
        scatter_dir = Path(scatter_dir)
        scatter_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Scatter data will be saved to: {scatter_dir}")
    
    sight = Sight(model)
    sae_out_device = sae_out.to(device)
    
    print(f"\nAccumulating activations for layer {layer} across batches...")
    target_label = "all" if target_samples_per_feature <= 0 else target_samples_per_feature
    print(f"  Target: {target_label} active samples per feature")
    print(f"  Max batches: {max_batches}")
    
    estimated_tokens, total_batches = estimate_total_tokens(
        dataloader=dataloader,
        max_batches=max_batches,
        batch_size=batch_size,
        n_ctx=n_ctx,
    )
    if estimated_tokens is not None:
        print(f"  Estimated tokens: {estimated_tokens:,}")
    
    use_streaming_accum = (
        not force_full_accum
        and streaming_threshold_tokens > 0
        and estimated_tokens is not None
        and estimated_tokens > streaming_threshold_tokens
    )
    
    if exact_streaming:
        print("\nExact streaming accumulation enabled:")
        if estimated_tokens is not None:
            print(f"  Estimated tokens: {estimated_tokens:,}")
        else:
            print("  Estimated tokens: unknown")
        print("  Mode: exact (no caps, no concatenation)")
        
        # Pass 1: count active features across all batches
        feature_indices = _select_active_features_exact(
            sae_out=sae_out_device,
            sight=sight,
            dataloader=dataloader,
            layer=layer,
            feature_indices=feature_indices,
            min_active_per_feature=min_active_per_feature,
            device=device,
            max_batches=max_batches,
            total_batches=total_batches,
        )
        if len(feature_indices) == 0:
            print("  Warning: None of requested features are active.")
        
        # Limit to max_features (if specified, -1 means all)
        if max_features > 0 and len(feature_indices) > max_features:
            feature_indices = feature_indices[:max_features]
        
        n_features = len(feature_indices)
        print(f"  Active features selected: {n_features}")
        
        if n_features == 0:
            run_info = {
                "accumulation_mode": "streaming_exact",
                "estimated_tokens": estimated_tokens,
                "total_tokens": 0,
                "streaming_threshold_tokens": streaming_threshold_tokens,
            }
            return results, feature_results, scatter_files_saved, run_info
        
        # --- Pass 2: exact correlation accumulation (chunked to limit memory) ---
        k_max = max(ranks)
        rank_indices = torch.tensor([k - 1 for k in ranks], dtype=torch.long)
        
        if exact_chunk_size <= 0:
            exact_chunk_size = len(feature_indices)
        n_chunks = (len(feature_indices) + exact_chunk_size - 1) // exact_chunk_size
        print(f"  Exact chunk size: {exact_chunk_size} ({n_chunks} chunks)")
        
        total_tokens = 0
        
        for chunk_idx in range(0, len(feature_indices), exact_chunk_size):
            chunk = feature_indices[chunk_idx:chunk_idx + exact_chunk_size]
            print(f"\nExact chunk {chunk_idx // exact_chunk_size + 1}/{n_chunks}: {len(chunk)} features")
            
            eigenpairs_by_feature = _build_eigenpairs_for_chunk(
                chunk=chunk,
                model=model,
                sae_out=sae_out_device,
                layer=layer,
                use_streaming=use_streaming,
                chunk_size=chunk_size,
                device=device,
                eigenpairs_cache_dir=eigenpairs_cache_dir,
                k_max=k_max,
            )
            chunk_tensor = torch.tensor(chunk, device=device, dtype=torch.long)
            
            accumulators, scatter_buffers, tokens_counted = _accumulate_exact_chunk(
                dataloader=dataloader,
                sight=sight,
                sae_out=sae_out_device,
                layer=layer,
                chunk=chunk,
                chunk_tensor=chunk_tensor,
                eigenpairs_by_feature=eigenpairs_by_feature,
                rank_indices=rank_indices,
                ranks=ranks,
                max_batches=max_batches,
                total_batches=total_batches,
                device=device,
                save_scatter=save_scatter,
                max_scatter_samples=max_scatter_samples,
                count_tokens=(chunk_idx == 0),
            )
            total_tokens += tokens_counted
            
            _finalize_exact_chunk(
                chunk=chunk,
                accumulators=accumulators,
                scatter_buffers=scatter_buffers,
                results=results,
                feature_results=feature_results,
                scatter_files_saved=scatter_files_saved,
                ranks=ranks,
                metric=metric,
                scatter_dir=scatter_dir if save_scatter else None,
                model_name=model_name,
                max_scatter_samples=max_scatter_samples,
                min_active_per_feature=min_active_per_feature,
            )
            
            # Free chunk memory
            del eigenpairs_by_feature, accumulators, scatter_buffers, chunk_tensor
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        run_info = {
            "accumulation_mode": "streaming_exact",
            "estimated_tokens": estimated_tokens,
            "total_tokens": total_tokens,
            "streaming_threshold_tokens": streaming_threshold_tokens,
            "exact_chunk_size": exact_chunk_size,
        }
        
        return results, feature_results, scatter_files_saved, run_info
    
    if use_streaming_accum:
        print("\nAuto streaming accumulation enabled:")
        print(f"  Estimated tokens ({estimated_tokens:,}) exceed threshold ({streaming_threshold_tokens:,})")
        
        # Cap target samples in streaming mode
        stream_target = target_samples_per_feature
        if stream_target <= 0 or stream_target > streaming_max_samples:
            print(
                f"  Capping target samples per feature from {target_label} to {streaming_max_samples} (streaming mode)"
            )
            stream_target = streaming_max_samples
        
        # Compute feature cap based on memory budget
        d_model = getattr(model.config, "d_model", None) or sae_out_device.d_model
        memory_bytes = int(streaming_memory_gb * (1024 ** 3))
        bytes_per_feature = max(1, stream_target * d_model * 4)
        max_features_allowed = max(1, memory_bytes // bytes_per_feature)
        
        effective_max_features = max_features
        if effective_max_features <= 0:
            effective_max_features = max_features_allowed
        elif effective_max_features > max_features_allowed:
            print(
                f"  Capping max features from {effective_max_features} to {max_features_allowed} (memory budget)"
            )
            effective_max_features = max_features_allowed
        
        discovery_batches = max(1, min(discovery_batches, max_batches))
        print(f"  Discovery batches: {discovery_batches}")
        print(f"  Streaming target samples per feature: {stream_target}")
        print(f"  Streaming feature cap: {effective_max_features}")
        
        # --- Discovery pass: find active features without storing full activations ---
        active_counts = torch.zeros(sae_out_device.d_features, dtype=torch.int64)
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Discovery pass", total=min(discovery_batches, total_batches))
            ):
                if batch_idx >= discovery_batches:
                    break
                
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with sight.trace(batch, validate=False, scan=False):
                    mlp_out = sight["mlp-out", layer].save()
                
                mlp_out_flat = mlp_out.flatten(0, 1).float()
                z_true = sae_out_device.encode(mlp_out_flat)
                active_counts += (z_true > 0).sum(dim=0).cpu()
        
        # Select features to analyze
        if feature_indices:
            feature_indices = [f for f in feature_indices if active_counts[f] >= min_active_per_feature]
            if effective_max_features > 0 and len(feature_indices) > effective_max_features:
                print(
                    f"  Warning: Capping requested features from {len(feature_indices)} to {effective_max_features}"
                )
                feature_indices = feature_indices[:effective_max_features]
        else:
            active_mask = active_counts >= min_active_per_feature
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            
            if effective_max_features > 0 and len(active_indices) > effective_max_features:
                active_counts_masked = active_counts.clone()
                active_counts_masked[~active_mask] = -1
                top_vals, top_idx = torch.topk(active_counts_masked, k=effective_max_features)
                feature_indices = [
                    idx.item() for idx, val in zip(top_idx, top_vals) if val.item() >= min_active_per_feature
                ]
            else:
                feature_indices = active_indices.tolist()
        
        n_features = len(feature_indices)
        print(f"  Active features selected: {n_features}")
        
        if n_features == 0:
            run_info = {
                "accumulation_mode": "streaming",
                "estimated_tokens": estimated_tokens,
                "total_tokens": 0,
                "streaming_threshold_tokens": streaming_threshold_tokens,
                "streaming_target_samples": stream_target,
                "streaming_memory_gb": streaming_memory_gb,
                "streaming_feature_cap": effective_max_features,
                "discovery_batches": discovery_batches,
            }
            return results, feature_results, scatter_files_saved, run_info
        
        # --- Streaming accumulation: collect per-feature samples up to target ---
        sample_store = {feat_idx: {"x": [], "z": []} for feat_idx in feature_indices}
        sample_counts = {feat_idx: 0 for feat_idx in feature_indices}
        done_features = set()
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Processing batches (streaming)", total=total_batches)
            ):
                if batch_idx >= max_batches:
                    break
                
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with sight.trace(batch, validate=False, scan=False):
                    mlp_in = sight["mlp-in", layer].save()
                    mlp_out = sight["mlp-out", layer].save()
                
                mlp_in_flat = mlp_in.flatten(0, 1).float()
                mlp_out_flat = mlp_out.flatten(0, 1).float()
                
                z_true = sae_out_device.encode(mlp_out_flat)
                z_true_selected = z_true[:, feature_indices]
                
                total_tokens += mlp_in_flat.shape[0]
                
                for idx, feat_idx in enumerate(feature_indices):
                    if stream_target > 0 and sample_counts[feat_idx] >= stream_target:
                        continue
                    
                    z_feat = z_true_selected[:, idx]
                    active_mask = z_feat > 0
                    if not active_mask.any():
                        continue
                    
                    x_active = mlp_in_flat[active_mask].cpu()
                    z_active = z_feat[active_mask].cpu()
                    
                    if stream_target > 0:
                        needed = stream_target - sample_counts[feat_idx]
                        if needed <= 0:
                            continue
                        if x_active.shape[0] > needed:
                            x_active = x_active[:needed]
                            z_active = z_active[:needed]
                    
                    if x_active.shape[0] == 0:
                        continue
                    
                    sample_store[feat_idx]["x"].append(x_active)
                    sample_store[feat_idx]["z"].append(z_active)
                    sample_counts[feat_idx] += x_active.shape[0]
                    
                    if stream_target > 0 and sample_counts[feat_idx] >= stream_target:
                        done_features.add(feat_idx)
                
                if stream_target > 0 and len(done_features) == len(feature_indices):
                    break
        
        # Concatenate per-feature samples
        feature_samples = {}
        for feat_idx in feature_indices:
            if sample_store[feat_idx]["x"]:
                x_concat = torch.cat(sample_store[feat_idx]["x"], dim=0)
                z_concat = torch.cat(sample_store[feat_idx]["z"], dim=0)
                feature_samples[feat_idx] = (x_concat, z_concat)
        
        # Determine eigenpairs source
        if eigenpairs_cache_dir is not None:
            eigenpairs_source = f"cached ({eigenpairs_cache_dir})"
        elif use_streaming:
            eigenpairs_source = f"streaming ({device})"
        else:
            eigenpairs_source = "standard (CPU)"
        print(f"\nAnalyzing {len(feature_samples)} features across ranks {ranks} using {eigenpairs_source}...")
        
        for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
            if feat_idx not in feature_samples:
                continue
            
            x_active, z_active = feature_samples[feat_idx]
            n_active = z_active.shape[0]
            
            if n_active < min_active_per_feature:
                continue
            
            # Get eigenpairs - either from cache or compute on-the-fly
            if eigenpairs_cache_dir is not None:
                try:
                    eigenpairs = load_cached_eigenpairs(eigenpairs_cache_dir, feat_idx)
                except FileNotFoundError:
                    continue
            else:
                out_direction = sae_out_device.w_enc.weight[feat_idx, :]
                if use_streaming:
                    eigenpairs = get_interaction_eigenpairs_streaming(
                        model=model,
                        layer=layer,
                        feat_idx=feat_idx,
                        out_direction=out_direction,
                        chunk_size=chunk_size,
                        compute_device=device,
                    )
                else:
                    eigenpairs = get_interaction_eigenpairs(
                        model=model,
                        layer=layer,
                        feat_idx=feat_idx,
                        out_direction=out_direction.cpu(),
                        device="cpu",
                    )
            
            feature_corrs = {}
            z_pred_rank2 = None
            
            for k in ranks:
                k_actual = min(k, len(eigenpairs.eigenvalues))
                eigenvalues_k = eigenpairs.eigenvalues[:k_actual]
                eigenvectors_k = eigenpairs.eigenvectors[:, :k_actual]
                
                z_pred = predict_activation_from_eigenpairs(
                    x_active, eigenvalues_k, eigenvectors_k
                )
                
                if k == 2 and save_scatter:
                    z_pred_rank2 = z_pred
                
                corr = metric_fn(z_active, z_pred)
                
                if not np.isnan(corr):
                    results[k].append(corr)
                    feature_corrs[k] = corr
            
            result_dict = {
                "feat_idx": feat_idx,
                "n_active": int(n_active),
                "correlations": feature_corrs,
            }
            
            if save_scatter and z_pred_rank2 is not None and scatter_dir is not None:
                scatter_limit = max_scatter_samples if max_scatter_samples > 0 else len(z_active)
                scatter_limit = min(scatter_limit, len(z_active))
                
                scatter_data = {
                    "feat_idx": feat_idx,
                    "n_active": int(n_active),
                    "z_true": z_active[:scatter_limit].cpu().tolist(),
                    "z_pred_rank2": z_pred_rank2[:scatter_limit].cpu().tolist(),
                    "correlation_rank2": feature_corrs.get(2, None),
                }
                
                scatter_file = scatter_dir / f"scatter_{model_name}_feat_{feat_idx}.json"
                with open(scatter_file, "w") as f:
                    json.dump(scatter_data, f)
                scatter_files_saved.append(str(scatter_file))
                result_dict["scatter_file"] = str(scatter_file)
            
            feature_results.append(result_dict)
        
        run_info = {
            "accumulation_mode": "streaming",
            "estimated_tokens": estimated_tokens,
            "total_tokens": total_tokens,
            "streaming_threshold_tokens": streaming_threshold_tokens,
            "streaming_target_samples": stream_target,
            "streaming_memory_gb": streaming_memory_gb,
            "streaming_feature_cap": effective_max_features,
            "discovery_batches": discovery_batches,
        }
        
        return results, feature_results, scatter_files_saved, run_info
    
    # Accumulate mlp_in and SAE activations across batches
    all_mlp_in = []
    all_z_true = []
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=total_batches)):
            if batch_idx >= max_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass to capture mlp_in and mlp_out
            with sight.trace(batch, validate=False, scan=False):
                mlp_in = sight["mlp-in", layer].save()
                mlp_out = sight["mlp-out", layer].save()
            
            # Flatten batch and sequence: [batch*seq, d_model]
            mlp_in_flat = mlp_in.flatten(0, 1).float()
            mlp_out_flat = mlp_out.flatten(0, 1).float()
            
            # Compute SAE activations
            z_true = sae_out_device.encode(mlp_out_flat)  # [batch*seq, d_features]
            
            # Move to CPU and accumulate
            all_mlp_in.append(mlp_in_flat.cpu())
            all_z_true.append(z_true.cpu())
            total_tokens += mlp_in_flat.shape[0]
        
        # Concatenate all accumulated data
        print(f"\nConcatenating {len(all_mlp_in)} batches ({total_tokens} tokens)...")
        mlp_in_all = torch.cat(all_mlp_in, dim=0)  # [total_tokens, d_model]
        z_true_all = torch.cat(all_z_true, dim=0)  # [total_tokens, d_features]
        
        del all_mlp_in, all_z_true
        
        print(f"  mlp_in_all shape: {mlp_in_all.shape}")
        print(f"  z_true_all shape: {z_true_all.shape}")
        print(f"  z_true_all nonzero: {(z_true_all > 0).sum().item()}")
        
        # Find features with sufficient activations
        print("Finding active features...")
        all_active_features = find_active_features(z_true_all, min_active=min_active_per_feature)
        print(f"  Found {len(all_active_features)} features with >= {min_active_per_feature} active positions")
        
        # Select features to analyze
        if feature_indices:
            active_set = set(all_active_features)
            feature_indices = [f for f in feature_indices if f in active_set]
            if len(feature_indices) == 0:
                print(f"  Warning: None of requested features are active, using auto-detected features")
                feature_indices = all_active_features
        else:
            feature_indices = all_active_features
        
        # Limit to max_features (if specified, -1 means all)
        if max_features > 0 and len(feature_indices) > max_features:
            feature_indices = feature_indices[:max_features]
        
        n_features = len(feature_indices)
        
        # Determine eigenpairs source
        if eigenpairs_cache_dir is not None:
            eigenpairs_source = f"cached ({eigenpairs_cache_dir})"
        elif use_streaming:
            eigenpairs_source = f"streaming ({device})"
        else:
            eigenpairs_source = "standard (CPU)"
        print(f"\nAnalyzing {n_features} features across ranks {ranks} using {eigenpairs_source}...")
        
        for feat_idx in tqdm(feature_indices, desc="Analyzing features"):
            # Get eigenpairs - either from cache or compute on-the-fly
            if eigenpairs_cache_dir is not None:
                # Load precomputed eigenpairs from cache (fast path)
                try:
                    eigenpairs = load_cached_eigenpairs(eigenpairs_cache_dir, feat_idx)
                except FileNotFoundError:
                    # Feature not in cache, skip it
                    continue
            else:
                # Compute eigenpairs on-the-fly (slow path)
                # Get encoder direction for this feature (as per paper's Tracer default)
                # The SAE activation is z_c = ReLU(mlp_out · w_enc[c]), so we need encoder direction
                out_direction = sae_out_device.w_enc.weight[feat_idx, :]
                
                # Get eigenpairs from weights (unprojected Q in residual stream space)
                if use_streaming:
                    # GPU-accelerated streaming computation
                    eigenpairs = get_interaction_eigenpairs_streaming(
                        model=model,
                        layer=layer,
                        feat_idx=feat_idx,
                        out_direction=out_direction,
                        chunk_size=chunk_size,
                        compute_device=device,
                    )
                else:
                    # Standard CPU computation
                    eigenpairs = get_interaction_eigenpairs(
                        model=model,
                        layer=layer,
                        feat_idx=feat_idx,
                        out_direction=out_direction.cpu(),
                        device="cpu",
                    )
            
            # True activation for this feature across ALL accumulated tokens
            z_true_feat = z_true_all[:, feat_idx]  # [total_tokens]
            
            # Filter for active positions (z > 0)
            active_mask = z_true_feat > 0
            n_active = active_mask.sum().item()
            
            # Skip features with insufficient active samples
            if n_active < min_active_per_feature:
                continue
            
            x_active = mlp_in_all[active_mask]   # [n_active, d_model]
            z_active = z_true_feat[active_mask]  # [n_active]
            
            feature_corrs = {}
            z_pred_rank2 = None  # Store rank-2 predictions for scatter plot
            
            # Compute correlation for each rank
            for k in ranks:
                k_actual = min(k, len(eigenpairs.eigenvalues))
                eigenvalues_k = eigenpairs.eigenvalues[:k_actual]
                eigenvectors_k = eigenpairs.eigenvectors[:, :k_actual]
                
                # Predicted activation: sum_j lambda_j * (v_j . x)^2
                z_pred = predict_activation_from_eigenpairs(
                    x_active, eigenvalues_k, eigenvectors_k
                )
                
                # Store rank-2 predictions for scatter plot
                if k == 2 and save_scatter:
                    z_pred_rank2 = z_pred
                
                # Compute metric (pearson or cosine) on FULL accumulated data
                corr = metric_fn(z_active, z_pred)
                
                if not np.isnan(corr):
                    results[k].append(corr)
                    feature_corrs[k] = corr
            
            # Build feature result dict
            result_dict = {
                "feat_idx": feat_idx,
                "n_active": n_active,
                "correlations": feature_corrs,
            }
            
            # Save scatter data to file immediately (memory-efficient streaming)
            if save_scatter and z_pred_rank2 is not None and scatter_dir is not None:
                scatter_limit = max_scatter_samples if max_scatter_samples > 0 else len(z_active)
                scatter_limit = min(scatter_limit, len(z_active))
                
                scatter_data = {
                    "feat_idx": feat_idx,
                    "n_active": n_active,
                    "z_true": z_active[:scatter_limit].cpu().tolist(),
                    "z_pred_rank2": z_pred_rank2[:scatter_limit].cpu().tolist(),
                    "correlation_rank2": feature_corrs.get(2, None),
                }
                
                # Save to individual file immediately
                scatter_file = scatter_dir / f"scatter_{model_name}_feat_{feat_idx}.json"
                with open(scatter_file, "w") as f:
                    json.dump(scatter_data, f)
                scatter_files_saved.append(str(scatter_file))
                
                # Note in result that scatter data is in separate file
                result_dict["scatter_file"] = str(scatter_file)
                
                # Free memory
                del scatter_data
            
            feature_results.append(result_dict)
    
    run_info = {
        "accumulation_mode": "full",
        "estimated_tokens": estimated_tokens,
        "total_tokens": total_tokens,
        "streaming_threshold_tokens": streaming_threshold_tokens,
    }
    
    return results, feature_results, scatter_files_saved, run_info


def main():
    parser = argparse.ArgumentParser(description="Verify weight-correlation predictions")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/correlation_analysis.json")
    parser.add_argument("--device", type=str, default=None)
    
    # Model/SAE overrides (optional, overrides config values)
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--layer", type=int, default=None, help="SAE layer (overrides config)")
    parser.add_argument("--expansion", type=int, default=None, help="SAE expansion (overrides config)")
    parser.add_argument("--k", type=int, default=None, help="SAE top-k (overrides config)")
    
    # Analysis parameters
    parser.add_argument("--n-features", type=str, default="all", help="Number of features to analyze ('all' or integer)")
    parser.add_argument("--ranks", type=str, default="1,2,4,8,16", help="Ranks to evaluate (comma-separated or range like 1-60)")
    parser.add_argument("--n-samples", type=str, default="2000", help="Number of validation samples ('all' or integer)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--max-batches", type=int, default=50, help="Maximum batches to process")
    parser.add_argument("--min-active", type=int, default=1, help="Minimum active samples per feature")
    parser.add_argument("--target-samples", type=str, default="500", help="Target active samples per feature ('all' or integer, default: 500)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--plot", type=str, default=None, help="Path to save correlation plot (PNG)")
    
    # Performance options
    parser.add_argument("--no-streaming", action="store_true", help="Disable GPU-accelerated streaming Q computation")
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size for streaming Q computation")
    
    # Accumulation safety (auto streaming)
    parser.add_argument(
        "--streaming-threshold",
        type=int,
        default=500_000,
        help="Auto-switch to streaming accumulation if estimated tokens exceed this",
    )
    parser.add_argument(
        "--streaming-max-samples",
        type=int,
        default=2000,
        help="Cap active samples per feature in streaming mode",
    )
    parser.add_argument(
        "--streaming-memory-gb",
        type=float,
        default=2.0,
        help="Approximate memory budget (GB) for streaming feature samples",
    )
    parser.add_argument(
        "--force-full-accum",
        action="store_true",
        help="Disable auto streaming accumulation and always concatenate all batches",
    )
    parser.add_argument(
        "--discovery-batches",
        type=int,
        default=20,
        help="Batches to scan for active features in streaming mode",
    )
    parser.add_argument(
        "--streaming-exact",
        action="store_true",
        help="Use exact streaming accumulation (no caps, slower but equivalent)",
    )
    parser.add_argument(
        "--exact-chunk-size",
        type=int,
        default=64,
        help="Feature chunk size for exact streaming (smaller = less memory, slower)",
    )
    
    # Figure 9C scatter data options
    parser.add_argument("--save-scatter", action="store_true", help="Save scatter data (z_true, z_pred) for Figure 9C")
    parser.add_argument("--max-scatter-samples", type=int, default=1000, help="Max scatter samples per feature (-1 for all)")
    
    # Metric selection
    parser.add_argument("--metric", type=str, default="pearson", choices=["pearson", "cosine"],
                        help="Similarity metric: 'pearson' (default) or 'cosine'")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "fineweb"],
                        help="Validation dataset: 'tinystories' (default) or 'fineweb'. "
                             "Use 'fineweb' for fw-small and fw-medium models.")
    
    # Eigenpairs caching (Phase 2 - load precomputed)
    parser.add_argument("--load-eigenpairs", type=str, default=None,
                        help="Path to precomputed eigenpairs directory, or 'auto' to auto-detect. "
                             "Enables rapid iteration without recomputation.")
    
    args = parser.parse_args()
    
    # Parse n_features - support 'all' or integer
    if args.n_features.lower() == "all" or args.n_features == "-1":
        n_features = -1  # -1 means all features
    else:
        n_features = int(args.n_features)
    
    # Parse target_samples - support 'all' or integer
    if args.target_samples.lower() == "all" or args.target_samples == "-1":
        target_samples = -1  # -1 means use all available (streaming may cap)
    else:
        target_samples = int(args.target_samples)
    
    # Parse n_samples - support 'all' or integer
    if args.n_samples.lower() == "all" or args.n_samples == "-1":
        n_samples = -1  # -1 means all samples from validation set
    else:
        n_samples = int(args.n_samples)
    
    # Adjust output path for cosine metric - put in cosine/ subfolder
    if args.metric == "cosine":
        output_path = Path(args.output)
        # Insert 'cosine' subfolder before the filename
        cosine_dir = output_path.parent / "cosine"
        args.output = str(cosine_dir / output_path.name)
        print(f"Cosine metric: output will be saved to {args.output}")
    
    # Parse ranks - support both comma-separated and range syntax
    if "-" in args.ranks and "," not in args.ranks:
        # Range syntax: "1-60"
        start, end = map(int, args.ranks.split("-"))
        ranks = list(range(start, end + 1))
    else:
        # Comma-separated syntax: "1,2,4,8,16"
        ranks = [int(r.strip()) for r in args.ranks.split(",")]
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - some operations may use CPU fallback")
    
    # Load config
    config = load_config(args.config)
    config_name = config.get("name", Path(args.config).stem)
    
    # Apply CLI overrides to config
    if args.model is not None:
        config.setdefault("model", {})["pretrained"] = args.model
    if args.layer is not None:
        config.setdefault("sae", {})["layer"] = args.layer
    if args.expansion is not None:
        # Set both flat and nested expansion values
        config.setdefault("sae", {})["expansion"] = args.expansion
        # Also update nested configs if they exist
        if "input" in config.get("sae", {}):
            config["sae"]["input"]["expansion"] = args.expansion
        if "output" in config.get("sae", {}):
            config["sae"]["output"]["expansion"] = args.expansion
    if args.k is not None:
        # Set both flat and nested k values
        config.setdefault("sae", {})["k"] = args.k
        # Also update nested configs if they exist
        if "input" in config.get("sae", {}):
            config["sae"]["input"]["k"] = args.k
        if "output" in config.get("sae", {}):
            config["sae"]["output"]["k"] = args.k
    
    # Initialize wandb
    wandb_enabled = init_wandb(
        name=f"correlation_{config_name}",
        config={**config, "ranks": ranks, "n_features": n_features},
        device=device,
        enabled=not args.no_wandb,
        tags=["language", "correlation", "figure9"],
    )
    
    # Run with emissions tracking
    with track_emissions("bilinear-mlp-reproduction") as tracker:
        # Use LanguageContext for unified model/SAE loading
        ctx = LanguageContext(config, device)
        model = ctx.model
        layer = ctx.layer
        expansion = ctx.expansion
        k = ctx.k
        model_name = ctx.model_name
        
        # Load output SAE via context
        sae_out = ctx.get_sae("mlp-out")
        
        # Create validation DataLoader via context
        dataloader = ctx.get_dataloader(
            n_samples=n_samples,
            batch_size=args.batch_size,
            dataset_name=args.dataset,
        )
        
        # Select features to analyze
        # Pass None to find active features dynamically, then limit to n_features
        feature_indices = None  # Will be populated by find_active_features in verify_correlation
        
        # Create scatter directory if saving scatter data
        output_path = Path(args.output)
        scatter_dir = None
        if args.save_scatter:
            # For cosine metric, scatter data goes to cosine/scatter_data
            scatter_dir = output_path.parent / "scatter_data"
            scatter_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract model short name for scatter file naming
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Resolve eigenpairs cache directory if provided
        eigenpairs_cache_dir = None
        if args.load_eigenpairs:
            eigenpairs_cache_dir = resolve_eigenpairs_cache(
                args.load_eigenpairs, model_name, layer
            )
            print(f"Using cached eigenpairs from: {eigenpairs_cache_dir}")
        
        # Run correlation analysis with batch accumulation
        n_ctx = config.get("sae", {}).get("n_ctx", 256)
        
        results, feature_results, scatter_files, run_info = verify_correlation(
            model=model,
            sae_out=sae_out,
            layer=layer,
            dataloader=dataloader,
            feature_indices=feature_indices,
            ranks=ranks,
            device=device,
            min_active_per_feature=args.min_active,
            target_samples_per_feature=target_samples,
            max_batches=args.max_batches,
            max_features=n_features,
            save_scatter=args.save_scatter,
            max_scatter_samples=args.max_scatter_samples,
            scatter_dir=scatter_dir,
            model_name=model_short,
            use_streaming=not args.no_streaming,
            chunk_size=args.chunk_size,
            metric=args.metric,
            eigenpairs_cache_dir=eigenpairs_cache_dir,
            batch_size=args.batch_size,
            n_ctx=n_ctx,
            streaming_threshold_tokens=args.streaming_threshold,
            streaming_max_samples=args.streaming_max_samples,
            streaming_memory_gb=args.streaming_memory_gb,
            force_full_accum=args.force_full_accum,
            discovery_batches=args.discovery_batches,
            exact_streaming=args.streaming_exact,
            exact_chunk_size=args.exact_chunk_size,
        )
    
    # Compute summary statistics
    metric_label = "Cosine similarity" if args.metric == "cosine" else "Pearson correlation"
    summary = {
        "model_name": model_name,
        "layer": layer,
        "expansion": expansion,
        "k": k,
        "metric": args.metric,
        "metric_label": metric_label,
        "dataset": args.dataset,
        "n_features_requested": "all" if n_features == -1 else n_features,
        "n_features_analyzed": len(feature_results),
        "ranks": ranks,
        "correlation_by_rank": {},
        "scatter_data_saved": args.save_scatter,
        "scatter_data_dir": str(scatter_dir) if scatter_dir else None,
        "scatter_files_count": len(scatter_files) if args.save_scatter else 0,
        "max_scatter_samples": args.max_scatter_samples if args.save_scatter else None,
        "use_streaming": not args.no_streaming,
        "chunk_size": args.chunk_size if not args.no_streaming else None,
        "eigenpairs_cached": args.load_eigenpairs is not None,
        "eigenpairs_cache_dir": str(eigenpairs_cache_dir) if eigenpairs_cache_dir else None,
        "accumulation_mode": run_info.get("accumulation_mode"),
        "estimated_tokens": run_info.get("estimated_tokens"),
        "total_tokens": run_info.get("total_tokens"),
        "streaming_threshold_tokens": run_info.get("streaming_threshold_tokens"),
        "streaming_target_samples": run_info.get("streaming_target_samples"),
        "streaming_memory_gb": run_info.get("streaming_memory_gb"),
        "streaming_feature_cap": run_info.get("streaming_feature_cap"),
        "discovery_batches": run_info.get("discovery_batches"),
        "exact_chunk_size": run_info.get("exact_chunk_size"),
        "paper_claim": "69% of features have >0.75 rank-2 correlation",
        "wall_time_seconds": tracker.result.wall_time_seconds,
        "co2_kg": tracker.result.emissions_kg,
    }
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULTS ({metric_label.upper()})")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {layer}")
    print(f"Metric: {metric_label}")
    print(f"Features analyzed: {summary['n_features_analyzed']}")
    print()
    
    for k in ranks:
        corrs = results[k]
        if len(corrs) > 0:
            mean_corr = np.mean(corrs)
            std_corr = np.std(corrs)
            median_corr = np.median(corrs)
            summary["correlation_by_rank"][str(k)] = {
                "mean": float(mean_corr),
                "std": float(std_corr),
                "median": float(median_corr),
                "n_features": len(corrs),
            }
            print(f"Rank {k:2d}: mean={mean_corr:.4f}, std={std_corr:.4f}, median={median_corr:.4f} (n={len(corrs)})")
        else:
            print(f"Rank {k:2d}: No valid correlations computed")
    
    print(f"\nPaper claim: {summary['paper_claim']}")
    rank2_stats = summary["correlation_by_rank"].get("2") or summary["correlation_by_rank"].get(2)
    if rank2_stats:
        rank2_corr = rank2_stats["mean"]
        # Calculate percentage above 0.75 from per-feature results (handle int/str keys)
        rank2_values = []
        for r in feature_results:
            corr_map = r.get("correlations", {})
            corr = corr_map.get("2")
            if corr is None:
                corr = corr_map.get(2)
            if corr is not None:
                rank2_values.append(corr)
        pct_above_75 = sum(1 for v in rank2_values if v > 0.75) / len(rank2_values) * 100 if rank2_values else 0
        print(f"Our rank-2: mean={rank2_corr:.4f}, {pct_above_75:.1f}% above 0.75 threshold")
    print(f"{'='*60}")
    
    # Finalize wandb
    if wandb_enabled:
        extra_summary = {
            "n_features_analyzed": summary["n_features_analyzed"],
        }
        for k, stats in summary["correlation_by_rank"].items():
            extra_summary[f"correlation_rank_{k}_mean"] = stats["mean"]
            extra_summary[f"correlation_rank_{k}_std"] = stats["std"]
        finish_wandb(tracker.result, extra_summary=extra_summary)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        "summary": summary,
        "per_feature": [
            {
                "feat_idx": fr["feat_idx"],
                "n_active": fr["n_active"],
                "correlations": {str(k): float(v) for k, v in fr["correlations"].items()},
            }
            for fr in feature_results
        ],
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
    if args.save_scatter and scatter_dir:
        print(f"Scatter data saved to: {scatter_dir} ({len(scatter_files)} files)")
    
    # Generate plot if requested
    if args.plot and len(summary["correlation_by_rank"]) > 0:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_correlation_vs_rank(summary, str(plot_path))


if __name__ == "__main__":
    main()
