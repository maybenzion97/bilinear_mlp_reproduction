#!/usr/bin/env python3
"""
Precompute Eigenpairs for Language Experiments (Phase 1).

This script precomputes and caches eigendecompositions of interaction matrices
for all features in an SAE. This enables rapid iteration on correlation analysis
with different metrics (Pearson/cosine) and thresholds without recomputation.

The eigendecomposition is ~90% of compute time in verify_correlation.py.
Once cached, analysis runs in seconds instead of hours.

Storage estimates per model:
- ts-medium (512 d_model, 2048 features): ~2 GB
- fw-small (768 d_model, 3072 features): ~7 GB  
- fw-medium (1024 d_model, 8192 features): ~32 GB

Usage:
    # Precompute for fw-medium layer 7
    python src/language/precompute_eigenpairs.py --model fw-medium --layer 7
    
    # Precompute for ts-medium layer 5 (Figure 8)
    python src/language/precompute_eigenpairs.py --model ts-medium --layer 5
    
    # Resume interrupted computation
    python src/language/precompute_eigenpairs.py --model fw-medium --layer 7 --resume
    
    # Use float16 for 50% storage reduction
    python src/language/precompute_eigenpairs.py --model fw-medium --layer 7 --float16
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings

# Suppress torchvision image extension warning
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from src.utils import (
    get_device,
    setup_mps_fallbacks,
    is_mps_device,
    track_emissions,
)
from src.language.context import LanguageContext
from src.language.interaction_utils import get_interaction_eigenpairs_streaming
from src.paths import LANGUAGE_EIGENPAIRS


# Model configurations: model_name -> layer -> config
# For SAE versions (v0-v4), use sae_tag field
# Dataset field specifies validation dataset: "tinystories" or "fineweb"
MODEL_CONFIGS = {
    "ts-medium": {
        4: {"expansion": 4, "dataset": "tinystories", "purpose": "Figure 9 (paper's ts-tiny)"},
        5: {"expansion": 4, "dataset": "tinystories", "purpose": "Figure 8 (only layer with mlp-in SAE)"},
    },
    "fw-small": {
        8: {"expansion": 4, "dataset": "fineweb", "purpose": "Figure 9"},
    },
    "fw-medium": {
        7: {"expansion": 8, "dataset": "fineweb", "purpose": "Figure 8 tutorial"},
        10: {"expansion": 8, "dataset": "fineweb", "purpose": "Figure 9 (2/3 depth)"},
        12: {"expansion": 16, "dataset": "fineweb", "purpose": "Figure 10 (SAE training versions v0-v4)"},
    },
}

# Figure 10: SAE training versions (fw-medium layer 12, expansion 16)
FIGURE_10_SAE_VERSIONS = ["v0", "v1", "v2", "v3", "v4"]


def get_output_dir(model: str, layer: int, base_dir: Path = None, sae_tag: str = None) -> Path:
    """Get the output directory for cached eigenpairs."""
    if base_dir is None:
        base_dir = LANGUAGE_EIGENPAIRS
    
    # Normalize model name (e.g., "tdooms/fw-medium" -> "fw-medium")
    model_short = model.split("/")[-1]
    
    if sae_tag:
        # Figure 10: Include SAE version in path
        return base_dir / model_short / str(layer) / sae_tag
    else:
        return base_dir / model_short / str(layer)


def save_eigenpair(
    output_file: Path,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    feat_idx: int,
    model_name: str,
    layer: int,
    use_float16: bool = False,
):
    """Save eigenpair data to disk."""
    if use_float16:
        eigenvalues = eigenvalues.half()
        eigenvectors = eigenvectors.half()
    
    data = {
        'eigenvalues': eigenvalues.cpu(),
        'eigenvectors': eigenvectors.cpu(),
        'feat_idx': feat_idx,
        'model_name': model_name,
        'layer': layer,
        'd_model': eigenvalues.shape[0],
        'dtype': 'float16' if use_float16 else 'float32',
    }
    
    torch.save(data, output_file)


def load_eigenpair(filepath: Path) -> dict:
    """Load eigenpair data from disk."""
    return torch.load(filepath, map_location='cpu', weights_only=False)


def count_existing_features(output_dir: Path) -> int:
    """Count how many features have already been computed."""
    if not output_dir.exists():
        return 0
    return len(list(output_dir.glob("feat_*.pt")))


def load_sae_with_tag(repo: str, layer: int, expansion: int, k: int, tag: str, device: str):
    """
    Load SAE with a specific training tag (v0, v1, v2, v3, v4) for Figure 10.
    
    These SAE versions represent different training durations:
    - v0: 1x training (under-trained)
    - v1: 2x training
    - v2: 4x training
    - v3: 8x training
    - v4: 16x training (well-trained)
    """
    import json
    from sae.sae import SAE, SAEConfig
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_model
    
    point = ('mlp-out', layer)
    config = SAEConfig(point=point, expansion=expansion, k=k, d_model=0)
    base_name = f"{config.name}-{tag}"
    
    config_path = hf_hub_download(repo_id=repo, filename=f"{base_name}/config.json")
    model_path = hf_hub_download(repo_id=repo, filename=f"{base_name}/model.safetensors")
    
    sae = SAE.from_config(**json.load(open(config_path)))
    load_model(sae, model_path)
    return sae.to(device)


def precompute_eigenpairs(
    model_name: str,
    layer: int,
    expansion: int,
    device: str,
    output_dir: Path,
    resume: bool = True,
    use_float16: bool = False,
    chunk_size: int = 256,
    max_features: int = -1,
    sae_tag: str = None,
) -> dict:
    """
    Precompute eigendecompositions for all SAE features.
    
    Args:
        model_name: Model name (e.g., "fw-medium" or "tdooms/fw-medium")
        layer: Layer index for SAE
        expansion: SAE expansion factor
        device: Device for computation
        output_dir: Directory to save eigenpairs
        resume: If True, skip already-computed features
        use_float16: If True, save in float16 for 50% storage reduction
        chunk_size: Chunk size for streaming Q computation
        max_features: Maximum features to compute (-1 for all)
        sae_tag: Optional SAE version tag (v0, v1, v2, v3, v4) for Figure 10
    
    Returns:
        Summary dict with stats and emissions
    """
    # Ensure full model name
    if "/" not in model_name:
        model_name = f"tdooms/{model_name}"
    
    model_short = model_name.split("/")[-1]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EIGENPAIR PRECOMPUTATION")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Layer: {layer}")
    print(f"Expansion: {expansion}")
    if sae_tag:
        print(f"SAE Version: {sae_tag}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Resume: {resume}")
    print(f"Float16: {use_float16}")
    print(f"{'='*60}\n")
    
    # Load model and SAE
    if sae_tag:
        # Figure 10: Load specific SAE version (v0-v4)
        print(f"Loading SAE with tag: {sae_tag}")
        from language.transformer import Transformer
        
        # Load model
        if device == "mps":
            model = Transformer.from_pretrained(model_name, device="cpu").to(device)
        else:
            model = Transformer.from_pretrained(model_name, device=device)
        
        # Load SAE with specific version tag
        sae_repo = f"{model_name}-scope"
        sae_out = load_sae_with_tag(
            repo=sae_repo,
            layer=layer,
            expansion=expansion,
            k=30,
            tag=sae_tag,
            device=device,
        )
    else:
        # Default: Load using LanguageContext
        config = {
            "model": {"pretrained": model_name},
            "sae": {
                "layer": layer,
                "output": {
                    "name": "mlp-out",
                    "expansion": expansion,
                    "k": 30,
                },
            },
        }
        
        ctx = LanguageContext(config, device)
        model = ctx.model
        sae_out = ctx.get_sae("mlp-out")
    
    n_features = sae_out.d_features
    d_model = sae_out.d_model
    
    print(f"\nSAE Info:")
    print(f"  d_model: {d_model}")
    print(f"  n_features: {n_features}")
    
    # Estimate storage
    bytes_per_feature = d_model * 4 + d_model * d_model * 4  # eigenvalues + eigenvectors
    if use_float16:
        bytes_per_feature //= 2
    total_bytes = bytes_per_feature * n_features
    print(f"  Estimated storage: {total_bytes / 1e9:.2f} GB")
    
    # Determine features to compute
    if max_features > 0:
        n_features = min(max_features, n_features)
    
    # Check existing progress
    existing_count = count_existing_features(output_dir)
    if resume and existing_count > 0:
        print(f"\n  Resuming: {existing_count}/{n_features} already computed")
    
    # Compute eigenpairs
    computed_count = 0
    skipped_count = 0
    
    print(f"\nPrecomputing eigenpairs for {n_features} features...")
    
    for feat_idx in tqdm(range(n_features), desc="Features"):
        output_file = output_dir / f"feat_{feat_idx:04d}.pt"
        
        # Skip if already computed (resume mode)
        if resume and output_file.exists():
            skipped_count += 1
            continue
        
        # Get encoder direction for this feature
        out_direction = sae_out.w_enc.weight[feat_idx, :]
        
        # Compute eigenpairs using streaming (GPU-accelerated)
        eigenpairs = get_interaction_eigenpairs_streaming(
            model=model,
            layer=layer,
            feat_idx=feat_idx,
            out_direction=out_direction,
            chunk_size=chunk_size,
            compute_device=device,
        )
        
        # Save to disk
        save_eigenpair(
            output_file=output_file,
            eigenvalues=eigenpairs.eigenvalues,
            eigenvectors=eigenpairs.eigenvectors,
            feat_idx=feat_idx,
            model_name=model_name,
            layer=layer,
            use_float16=use_float16,
        )
        
        computed_count += 1
    
    print(f"\nCompleted:")
    print(f"  Computed: {computed_count}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Total: {computed_count + skipped_count}")
    
    return {
        "model_name": model_name,
        "layer": layer,
        "expansion": expansion,
        "sae_tag": sae_tag,
        "n_features": n_features,
        "d_model": d_model,
        "computed_count": computed_count,
        "skipped_count": skipped_count,
        "dtype": "float16" if use_float16 else "float32",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Precompute eigenpairs for language experiments"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name: ts-medium, fw-small, fw-medium"
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer index (uses default if not specified)"
    )
    parser.add_argument(
        "--expansion", type=int, default=None,
        help="SAE expansion factor (uses default if not specified)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/language/eigenpairs/{model}/{layer})"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu, mps, cuda (default: auto-detect)"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from existing progress (default: True)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume, recompute all features"
    )
    parser.add_argument(
        "--float16", action="store_true",
        help="Save in float16 for 50% storage reduction"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256,
        help="Chunk size for streaming Q computation (default: 256)"
    )
    parser.add_argument(
        "--max-features", type=int, default=-1,
        help="Maximum features to compute (-1 for all)"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--sae-tag", type=str, default=None,
        help="SAE version tag for Figure 10 (v0, v1, v2, v3, v4, or 'all')"
    )
    parser.add_argument(
        "--figure10", action="store_true",
        help="Precompute all SAE versions (v0-v4) for Figure 10 (fw-medium layer 12)"
    )
    
    args = parser.parse_args()
    
    # Handle --figure10 shortcut
    if args.figure10:
        args.model = "fw-medium"
        args.layer = 12
        args.expansion = 16
        args.sae_tag = "all"
        print("Figure 10 mode: fw-medium layer 12 with all SAE versions (v0-v4)")
    
    # Normalize model name
    model = args.model
    if "/" not in model:
        model_short = model
    else:
        model_short = model.split("/")[-1]
    
    # Get model config
    if model_short not in MODEL_CONFIGS:
        print(f"Error: Unknown model '{model_short}'")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    model_config = MODEL_CONFIGS[model_short]
    
    # Determine layer
    layer = args.layer
    if layer is None:
        # Use first available layer as default
        layer = list(model_config.keys())[0]
        print(f"Using default layer: {layer}")
    
    if layer not in model_config:
        print(f"Error: Layer {layer} not configured for model '{model_short}'")
        print(f"Available layers: {list(model_config.keys())}")
        sys.exit(1)
    
    # Get expansion
    expansion = args.expansion
    if expansion is None:
        expansion = model_config[layer]["expansion"]
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - some operations may use CPU fallback")
    
    # Resume mode
    resume = args.resume and not args.no_resume
    
    # Determine SAE versions to process
    if args.sae_tag == "all":
        sae_tags = FIGURE_10_SAE_VERSIONS
        print(f"Processing all SAE versions: {sae_tags}")
    elif args.sae_tag:
        sae_tags = [args.sae_tag]
    else:
        sae_tags = [None]  # No tag = default SAE
    
    # Process each SAE version
    for sae_tag in sae_tags:
        print(f"\n{'#'*60}")
        if sae_tag:
            print(f"# Processing SAE version: {sae_tag}")
        else:
            print(f"# Processing default SAE")
        print(f"{'#'*60}")
        
        # Get output directory (includes sae_tag if provided)
        if args.output_dir:
            output_dir = Path(args.output_dir)
            if sae_tag:
                output_dir = output_dir / sae_tag
        else:
            output_dir = get_output_dir(model_short, layer, sae_tag=sae_tag)
        
        # Run with emissions tracking
        with track_emissions("bilinear-mlp-reproduction") as tracker:
            stats = precompute_eigenpairs(
                model_name=model,
                layer=layer,
                expansion=expansion,
                device=device,
                output_dir=output_dir,
                resume=resume,
                use_float16=args.float16,
                chunk_size=args.chunk_size,
                max_features=args.max_features,
                sae_tag=sae_tag,
            )
        
        # Create manifest with emissions
        manifest = {
            **stats,
            "output_dir": str(output_dir),
            "emissions": {
                "co2_kg": tracker.result.emissions_kg,
                "wall_time_hours": tracker.result.wall_time_hours,
                "wall_time_seconds": tracker.result.wall_time_seconds,
                "gpu_hours": tracker.result.gpu_hours,
            },
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "chunk_size": args.chunk_size,
        }
        
        # Save manifest
        manifest_file = output_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n{'='*60}")
        if sae_tag:
            print(f"PRECOMPUTATION COMPLETE ({sae_tag})")
        else:
            print("PRECOMPUTATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Manifest: {manifest_file}")
        print(f"Emissions: {tracker.result.emissions_kg:.6f} kg CO2")
        print(f"Wall time: {tracker.result.wall_time_hours:.2f} hours")
        print(f"{'='*60}")
    
    # Summary for multiple versions
    if len(sae_tags) > 1:
        print(f"\n{'#'*60}")
        print(f"# ALL VERSIONS COMPLETE")
        print(f"{'#'*60}")
        print(f"Processed {len(sae_tags)} SAE versions: {sae_tags}")
        print(f"Base directory: {get_output_dir(model_short, layer)}")
    
    # Print usage hint (only once at the end)
    print("\nTo use cached eigenpairs:")
    base_cache_dir = get_output_dir(model_short, layer, sae_tag=sae_tags[0] if sae_tags[0] else None)
    print(f"  python src/language/verify_correlation.py \\")
    print(f"      --load-eigenpairs {base_cache_dir} \\")
    print(f"      --metric cosine --min-active 50")


if __name__ == "__main__":
    main()
