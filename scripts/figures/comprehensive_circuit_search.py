#!/usr/bin/env python3
"""
Comprehensive circuit search for fw-medium - optimized for MPS and CUDA.

Searches ALL output features for AND-gate structure, with optimizations:
- Batch processing to reduce memory transfers
- Caching of repeated computations
- Progress saving for resumability
- Streaming Q computation for CUDA (reduces memory from 17GB to 1GB)
"""

import sys
from pathlib import Path
import time
import json
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

import torch
import numpy as np

from src.utils import track_emissions
from src.language.streaming_tracer import q_streaming


def search_all_circuits(
    device: str = "mps",
    batch_size: int = 50,
    save_interval: int = 100,
    resume: bool = True,
    max_features: int = None,
    use_streaming: bool = False,
    chunk_size: int = 256,
):
    """
    Search all output features for AND-gate structure.
    
    Optimizations:
    - MPS: Keep model and latents on GPU, batch eigenvalue computations
    - CUDA (streaming): Chunk Q computation to reduce memory from 17GB to 1GB
    
    Args:
        device: Device to run on (mps, cuda, cpu)
        batch_size: Not currently used (reserved for future batch processing)
        save_interval: Save checkpoint every N features
        resume: Resume from checkpoint if exists
        max_features: Limit number of features to search
        use_streaming: Use streaming Q computation (reduces memory, enables CUDA)
        chunk_size: Chunk size for streaming (256=1GB, 128=0.5GB)
    """
    print(f"=" * 70)
    print(f"COMPREHENSIVE CIRCUIT SEARCH")
    print(f"=" * 70)
    print(f"Device: {device}")
    print(f"Streaming: {use_streaming} (chunk_size={chunk_size})")
    print(f"Batch size: {batch_size}")
    
    # Output paths
    results_dir = PROJECT_ROOT / "results/language"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "circuit_search_checkpoint.json"
    final_path = results_dir / "circuit_search_complete.json"
    
    # Load checkpoint if resuming
    processed_features = set()
    results = []
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        processed_features = set(checkpoint.get("processed", []))
        results = checkpoint.get("results", [])
        print(f"Resuming from checkpoint: {len(processed_features)} features already processed")
    
    print(f"\nLoading model on {device}...")
    start_time = time.time()
    
    from language.transformer import Transformer
    from sae import Tracer
    
    torch.set_grad_enabled(False)
    
    # Load model - MPS optimization: load to CPU first, then move
    if device == "mps":
        model = Transformer.from_pretrained("tdooms/fw-medium", device="cpu")
        model = model.to(device)
        # Ensure model is in eval mode
        model.eval()
    else:
        model = Transformer.from_pretrained("tdooms/fw-medium", device=device)
    
    # Create tracer
    tracer = Tracer(model, layer=7, inp=dict(expansion=8), out=dict(expansion=8), device=device)
    
    # Cache latents on CPU for stability (MPS can be flaky with large tensors)
    inp_latents = tracer.inp_latents.float().cpu()
    # BUG FIX: shape[0] is n_features (8192), shape[1] is d_model (1024)
    n_out = tracer.out_latents.shape[0]
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Total output features: {n_out}")
    
    # Determine features to process
    if max_features:
        features_to_process = list(range(min(max_features, n_out)))
    else:
        features_to_process = list(range(n_out))
    
    # Filter out already processed
    features_to_process = [f for f in features_to_process if f not in processed_features]
    print(f"Features to process: {len(features_to_process)}")
    
    # Estimate time
    est_time_per_feature = 1.5  # seconds
    est_total = len(features_to_process) * est_time_per_feature / 3600
    print(f"Estimated time: {est_total:.1f} hours")
    
    print(f"\n{'='*70}")
    print("Starting search...")
    print(f"{'='*70}\n")
    
    batch_start = time.time()
    
    for i, out_feat in enumerate(features_to_process):
        try:
            # Get Q matrix - this is the main computation
            # Streaming mode for CUDA: reduces memory from 17GB to ~1GB
            if use_streaming and device == "cuda":
                Q = q_streaming(tracer, out_feat, chunk_size=chunk_size)
                Q_cpu = Q.cpu()
            else:
                # Original path for MPS/CPU
                Q = tracer.q(out_feat, project=False).float()
                Q_cpu = Q.cpu()
            
            # Find top self-interacting input features
            # MPS optimization: do matrix ops on GPU
            inp_Q = torch.mm(Q_cpu, inp_latents)
            self_int = (inp_latents * inp_Q).sum(dim=0)
            top_inputs = self_int.abs().argsort(descending=True)[:20].tolist()
            
            # Compute submatrix for top features
            top_lat = inp_latents[:, top_inputs]
            Q_V = torch.mm(Q_cpu, top_lat)
            Q_sub = torch.mm(top_lat.T, Q_V)
            Q_sub = 0.5 * (Q_sub + Q_sub.T)
            
            # Eigendecomposition (CPU for stability)
            eigs = torch.linalg.eigvalsh(Q_sub)
            sorted_eigs = eigs.abs().sort(descending=True).values
            
            # Compute metrics
            top_ratio = (sorted_eigs[0] / (sorted_eigs[1:5].sum() + 1e-8)).item()
            
            # Block structure analysis
            half = 10
            self_1 = Q_sub[:half, :half].abs().mean().item()
            self_2 = Q_sub[half:, half:].abs().mean().item()
            cross = Q_sub[:half, half:].abs().mean().item()
            
            and_score = cross / ((self_1 + self_2)/2 + 1e-8)
            
            results.append({
                "feature": out_feat,
                "top_eigenvalue": sorted_eigs[0].item(),
                "eigen_ratio": top_ratio,
                "and_score": and_score,
                "cross_interaction": cross,
                "self_interaction": (self_1 + self_2)/2,
                "top_inputs": top_inputs[:15],
            })
            
            processed_features.add(out_feat)
            
        except Exception as e:
            print(f"  Error on feature {out_feat}: {e}")
            continue
        
        # Progress reporting
        if (i + 1) % 20 == 0:
            elapsed = time.time() - batch_start
            rate = (i + 1) / elapsed
            remaining = (len(features_to_process) - i - 1) / rate / 3600
            print(f"  Processed {i+1}/{len(features_to_process)} "
                  f"({rate:.1f} feat/s, ~{remaining:.1f}h remaining)")
        
        # Save checkpoint
        if (i + 1) % save_interval == 0:
            checkpoint = {
                "processed": list(processed_features),
                "results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f)
            print(f"  Checkpoint saved ({len(results)} results)")
    
    # Final save
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Features processed: {len(results)}")
    
    # Sort and analyze results
    by_and = sorted(results, key=lambda x: x["and_score"], reverse=True)
    by_rank = sorted(results, key=lambda x: x["eigen_ratio"], reverse=True)
    
    print(f"\nTOP 20 BY AND-GATE SCORE:")
    for r in by_and[:20]:
        print(f"  Feature {r['feature']:5d}: AND={r['and_score']:.2f}, "
              f"cross={r['cross_interaction']:.4f}, λ={r['top_eigenvalue']:.4f}")
    
    print(f"\nTOP 20 BY LOW-RANK (eigenvalue ratio):")
    for r in by_rank[:20]:
        print(f"  Feature {r['feature']:5d}: λ_ratio={r['eigen_ratio']:.2f}, "
              f"λ={r['top_eigenvalue']:.4f}, AND={r['and_score']:.2f}")
    
    # Save final results
    final_results = {
        "total_features": len(results),
        "total_time_hours": total_time / 3600,
        "top_by_and_score": by_and[:100],
        "top_by_eigen_ratio": by_rank[:100],
        "all_results": results,
    }
    
    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {final_path}")
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    return final_results


def analyze_top_candidate(device: str, feature_idx: int, n_top: int = 15, 
                          use_streaming: bool = False, chunk_size: int = 256):
    """
    Detailed analysis of a specific output feature.
    
    Args:
        device: Device to run on (mps, cuda, cpu)
        feature_idx: Output feature index to analyze
        n_top: Number of top input features to include
        use_streaming: Use streaming Q computation (reduces memory, enables CUDA)
        chunk_size: Chunk size for streaming (256=1GB, 128=0.5GB)
    """
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS: Output Feature {feature_idx}")
    print(f"{'='*70}")
    
    from language.transformer import Transformer
    from sae import Tracer
    
    torch.set_grad_enabled(False)
    
    if device == "mps":
        model = Transformer.from_pretrained("tdooms/fw-medium", device="cpu")
        model = model.to(device)
    else:
        model = Transformer.from_pretrained("tdooms/fw-medium", device=device)
    
    tracer = Tracer(model, layer=7, inp=dict(expansion=8), out=dict(expansion=8), device=device)
    inp_latents = tracer.inp_latents.float().cpu()
    
    # Get Q matrix - streaming mode for CUDA
    if use_streaming and device == "cuda":
        Q = q_streaming(tracer, feature_idx, chunk_size=chunk_size).cpu()
    else:
        Q = tracer.q(feature_idx, project=False).float().cpu()
    
    # Find top features
    inp_Q = torch.mm(Q, inp_latents)
    self_int = (inp_latents * inp_Q).sum(dim=0)
    top_features = self_int.abs().argsort(descending=True)[:n_top].tolist()
    
    print(f"Top {n_top} input features: {top_features}")
    
    # Compute submatrix
    top_lat = inp_latents[:, top_features]
    Q_V = torch.mm(Q, top_lat)
    Q_sub = torch.mm(top_lat.T, Q_V)
    Q_sub = 0.5 * (Q_sub + Q_sub.T)
    
    # Eigendecomposition
    eigs, vecs = torch.linalg.eigh(Q_sub)
    sorted_idx = eigs.abs().argsort(descending=True)
    eigs = eigs[sorted_idx]
    vecs = vecs[:, sorted_idx]
    
    v1 = vecs[:, 0]
    v2 = vecs[:, 1]
    
    # Cluster by v1 sign
    cluster_pos = [top_features[i] for i in range(n_top) if v1[i] > 0]
    cluster_neg = [top_features[i] for i in range(n_top) if v1[i] <= 0]
    
    print(f"\nCluster 1 (v1 > 0): {cluster_pos}")
    print(f"Cluster 2 (v1 <= 0): {cluster_neg}")
    print(f"\nTop eigenvalues: {eigs[:5].tolist()}")
    
    # Save detailed results
    results = {
        "output_feature": feature_idx,
        "top_input_features": top_features,
        "cluster_pos": cluster_pos,
        "cluster_neg": cluster_neg,
        "eigenvalues": eigs[:10].tolist(),
        "Q_submatrix": Q_sub.tolist(),
        "feature_projections": {
            str(top_features[i]): (v1[i].item(), v2[i].item())
            for i in range(n_top)
        },
    }
    
    out_path = PROJECT_ROOT / f"results/language/circuit_analysis_{feature_idx}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive circuit search for fw-medium")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=None, 
                        help="Limit number of features to search (default: all)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--analyze", type=int, default=None,
                        help="Analyze specific feature instead of searching")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: only search 10 features")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming Q computation (reduces memory, enables CUDA)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Chunk size for streaming (256=1GB, 128=0.5GB)")
    args = parser.parse_args()
    
    if args.test:
        print("=" * 70)
        print("TEST MODE: Searching only 10 features to verify setup")
        print("=" * 70)
        args.max_features = 10
        args.no_resume = True
    
    if args.analyze is not None:
        analyze_top_candidate(args.device, args.analyze, 
                             use_streaming=args.streaming, chunk_size=args.chunk_size)
    else:
        # Track emissions for the full search
        with track_emissions("bilinear-mlp-reproduction") as tracker:
            final_results = search_all_circuits(
                device=args.device,
                batch_size=args.batch_size,
                save_interval=args.save_interval,
                resume=not args.no_resume,
                max_features=args.max_features,
                use_streaming=args.streaming,
                chunk_size=args.chunk_size,
            )
        
        # Add emissions to final results and re-save
        if final_results is not None:
            final_results["emissions"] = {
                "co2_kg": tracker.result.emissions_kg,
                "wall_time_hours": tracker.result.wall_time_hours,
                "gpu_hours": tracker.result.gpu_hours,
            }
            final_path = PROJECT_ROOT / "results/language/circuit_search_complete.json"
            with open(final_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"\nEmissions added to results:")
            print(f"  CO2 (kg): {tracker.result.emissions_kg:.6f}")
            print(f"  Wall time: {tracker.result.wall_time_hours:.2f} hours")
            print(f"  GPU hours: {tracker.result.gpu_hours:.3f}")


if __name__ == "__main__":
    main()
