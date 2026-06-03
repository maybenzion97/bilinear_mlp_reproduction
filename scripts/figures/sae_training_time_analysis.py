#!/usr/bin/env python3
"""
SAE Training Time Analysis for Figure 10.

This script systematically compares SAE checkpoints with different training times
(v0-v4) to demonstrate that correlation quality improves with SAE training duration.

The paper claims:
- "correlation drastically improves with longer SAE training times" (Appendix C)
- "The correlation distribution is strongly bimodal for the 'under-trained' SAEs"

This script provides empirical evidence for these claims by comparing:
- v0: 1x training (under-trained)
- v1: 2x training
- v2: 4x training
- v3: 8x training
- v4: 16x training (well-trained)

All checkpoints are from fw-medium layer 12 with expansion=16.
Default dataset: FineWeb-Edu (sampled).

REUSES verify_correlation infrastructure for memory-efficient streaming (DRY).

Usage:
    python scripts/figures/sae_training_time_analysis.py
    python scripts/figures/sae_training_time_analysis.py --device mps --n-features 300
"""

import sys
from pathlib import Path
import json
import argparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

import torch

from language.transformer import Transformer
from sae.sae import SAE, SAEConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model

from src.utils import track_emissions
from src.language.verify_correlation import (
    verify_correlation,
    create_validation_dataloader,
)


def load_sae_with_tag(repo: str, layer: int, expansion: int, k: int, tag: str) -> SAE:
    """Load SAE with a specific training tag (v0, v1, v2, v3, v4)."""
    point = ('mlp-out', layer)
    config = SAEConfig(point=point, expansion=expansion, k=k, d_model=0)
    base_name = f"{config.name}-{tag}"
    
    config_path = hf_hub_download(repo_id=repo, filename=f"{base_name}/config.json")
    model_path = hf_hub_download(repo_id=repo, filename=f"{base_name}/model.safetensors")
    
    sae = SAE.from_config(**json.load(open(config_path)))
    load_model(sae, model_path)
    return sae


def run_analysis(args):
    """Run the actual analysis using verify_correlation infrastructure (DRY)."""
    metric_label = "Cosine similarity" if args.metric == "cosine" else "Pearson correlation"
    print("=" * 70)
    print(f"SAE Training Time Analysis (Figure 10) - {metric_label}")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Metric: {metric_label}")
    print(f"Features: {args.n_features} (-1=all)")
    print(f"Dataset: {args.dataset}")
    print(f"Batches: {args.n_batches}, Batch size: {args.batch_size}")
    print(f"Exact chunk size: {args.exact_chunk_size}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    model_name = 'tdooms/fw-medium'
    layer = 12
    expansion = 16
    k = 30
    sae_versions = ['v0', 'v1', 'v2', 'v3', 'v4']
    ranks = [1, 2, 4, 8, 16, 30]
    n_ctx = 256  # Match Figure 9's SAE config
    
    # Load model on the target device
    print("Loading model...")
    model = Transformer.from_pretrained(model_name, device=args.device)
    repo = f'{model.config.repo}-scope'
    print(f"  Model: {model_name}")
    print(f"  Layer: {layer}, Expansion: {expansion}")
    print(f"  n_ctx: {n_ctx}")
    print(f"  Device: {args.device}")
    
    # Create config dict expected by create_validation_dataloader
    config = {"sae": {"n_ctx": n_ctx}}
    
    # Create dataloader using verify_correlation's helper
    print("\nCreating validation dataloader...")
    dataloader = create_validation_dataloader(
        tokenizer=model.tokenizer,
        config=config,
        device=args.device,
        n_samples=args.n_batches * args.batch_size,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
    )
    
    # Estimate tokens
    estimated_tokens = args.n_batches * args.batch_size * n_ctx
    print(f"  Estimated tokens: {estimated_tokens:,} (~{estimated_tokens/1e6:.2f}M)")
    
    # Results container
    all_results = {
        'metadata': {
            'model': model_name,
            'layer': layer,
            'expansion': expansion,
            'k': k,
            'n_tokens': estimated_tokens,
            'ranks': ranks,
            'metric': args.metric,
            'metric_label': metric_label,
            'dataset': args.dataset,
            'n_batches': args.n_batches,
            'batch_size': args.batch_size,
            'exact_chunk_size': args.exact_chunk_size,
            'timestamp': datetime.now().isoformat(),
        },
        'versions': {}
    }
    
    # Analyze each SAE version using verify_correlation (reuse existing code)
    for version in sae_versions:
        print(f"\n{'='*60}")
        print(f"Analyzing SAE version: {version}")
        print(f"{'='*60}")
        
        # Load SAE
        print(f"  Loading SAE {version}...")
        sae = load_sae_with_tag(repo, layer, expansion, k, version)
        
        # Use verify_correlation with exact_streaming for memory efficiency
        print(f"  Running correlation analysis with exact_streaming...")
        results, feature_results, scatter_files, run_info = verify_correlation(
            model=model,
            sae_out=sae,
            layer=layer,
            dataloader=dataloader,
            feature_indices=None,  # Auto-detect active features
            ranks=ranks,
            device=args.device,
            min_active_per_feature=args.min_active,
            target_samples_per_feature=-1,  # All samples
            max_batches=args.n_batches,
            max_features=args.n_features,
            save_scatter=False,
            model_name=f"fw-medium-{version}",
            use_streaming=True,
            chunk_size=256,
            metric=args.metric,
            batch_size=args.batch_size,
            n_ctx=n_ctx,
            streaming_threshold_tokens=0,  # Always use streaming
            exact_streaming=True,  # Memory-safe exact computation
            exact_chunk_size=args.exact_chunk_size,
        )
        
        # Convert results to summary format expected by Figure 10
        summary = {}
        for rank in ranks:
            if rank in results and results[rank]:
                corrs = results[rank]
                import numpy as np
                summary[f'rank_{rank}'] = {
                    'mean': float(np.mean(corrs)),
                    'std': float(np.std(corrs)),
                    'median': float(np.median(corrs)),
                    'above_75_pct': float(np.mean(np.array(corrs) > 0.75) * 100),
                    'n_features': len(corrs),
                }
        
        all_results['versions'][version] = {
            'summary': summary,
            'per_feature': feature_results,
            'n_active_features': len(feature_results),
            'run_info': run_info,
        }
        
        # Print summary
        print(f"\n  Summary for {version}:")
        for rank in [1, 2, 8]:
            if f'rank_{rank}' in summary:
                s = summary[f'rank_{rank}']
                print(f"    Rank-{rank}: mean={s['mean']:.3f}, median={s['median']:.3f}, "
                      f">75%: {s['above_75_pct']:.1f}%")
        
        # Clean up SAE memory
        del sae
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("SUMMARY: SAE Training Time Effect")
    print(f"{'='*70}")
    print(f"{'Version':<10} {'Rank-1':>10} {'Rank-2':>10} {'Rank-8':>10} {'%>0.75':>10}")
    print("-" * 50)
    for version in sae_versions:
        v_results = all_results['versions'][version]['summary']
        r1 = v_results.get('rank_1', {}).get('mean', 0)
        r2 = v_results.get('rank_2', {}).get('mean', 0)
        r8 = v_results.get('rank_8', {}).get('mean', 0)
        pct = v_results.get('rank_2', {}).get('above_75_pct', 0)
        print(f"{version:<10} {r1:>10.3f} {r2:>10.3f} {r8:>10.3f} {pct:>9.1f}%")
    print("-" * 50)
    print(f"{'Paper':>10} {'~0.65':>10} {'>0.75':>10} {'--':>10} {'69%':>10}")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results, args.output


def main():
    parser = argparse.ArgumentParser(description="SAE Training Time Analysis (Figure 10)")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--n-features', type=int, default=-1, 
                        help='Number of features to analyze (-1 for all)')
    parser.add_argument('--n-batches', type=int, default=128, 
                        help='Number of batches (default 128, matching Figure 9)')
    parser.add_argument('--batch-size', type=int, default=48, 
                        help='Batch size (default 48, matching Figure 9)')
    parser.add_argument('--min-active', type=int, default=50, 
                        help='Minimum active samples per feature')
    parser.add_argument('--exact-chunk-size', type=int, default=64,
                        help='Features per chunk in exact streaming (smaller=less memory)')
    parser.add_argument('--output', type=str, 
                        default='results/language/sae_training_time_comparison.json',
                        help='Output JSON file')
    parser.add_argument('--metric', type=str, default='pearson', 
                        choices=['pearson', 'cosine'],
                        help="Similarity metric: 'pearson' (default) or 'cosine'")
    parser.add_argument('--dataset', type=str, default='fineweb',
                        choices=['fineweb', 'tinystories'],
                        help="Dataset to sample from (default: fineweb)")
    args = parser.parse_args()
    
    # Adjust output path for cosine metric
    if args.metric == "cosine":
        output_path = Path(args.output)
        cosine_dir = output_path.parent / "cosine"
        args.output = str(cosine_dir / output_path.name)
        print(f"Cosine metric: output will be saved to {args.output}")
    
    # Run analysis with emissions tracking
    with track_emissions("bilinear-mlp-reproduction") as tracker:
        all_results, output_file = run_analysis(args)
    
    # Add emissions to results
    all_results['emissions'] = {
        'co2_kg': tracker.result.emissions_kg,
        'wall_time_hours': tracker.result.wall_time_hours,
        'gpu_hours': tracker.result.gpu_hours,
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"\nEmissions:")
    print(f"  CO2 (kg): {tracker.result.emissions_kg:.6f}")
    print(f"  Wall time: {tracker.result.wall_time_hours:.2f} hours")
    print(f"  GPU hours: {tracker.result.gpu_hours:.3f}")


if __name__ == "__main__":
    main()
