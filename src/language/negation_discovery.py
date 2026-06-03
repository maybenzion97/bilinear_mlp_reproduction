"""
Negation circuit discovery in bilinear transformers.

Uses the original paper's SAE code to find negation features:
- Feature activating on "not + negative words" (e.g., "not lost")
- Feature activating on "not + positive words" (e.g., "not free")

Paper identifies features 3834 and 751 as the key negation features.

Usage:
    python src/language/negation_discovery.py --config configs/language_negation.yaml
    python src/language/negation_discovery.py --config configs/language_negation.yaml --use-pretrained
"""

import sys
from pathlib import Path
import argparse
import time
import warnings
import yaml
import json

# Suppress torchvision image extension warning (libjpeg not needed for our use case)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from language.transformer import Transformer
from sae.sae import SAE, SAEConfig
from datasets import load_dataset

from src.utils import (
    get_device,
    load_config,
    track_emissions,
    setup_mps_fallbacks,
    is_mps_device,
    init_wandb,
    finish_wandb,
)
from src.language.context import LanguageContext


# Negation and sentiment word lists (from paper analysis)
NEGATION_WORDS = ["not", "no", "never", "neither", "none", "without", "hardly", "barely", "n't"]
POSITIVE_WORDS = ["good", "happy", "free", "nice", "great", "love", "wonderful", "beautiful", "kind", "joy",
                   "glad", "fine", "well", "better", "best", "fun", "safe", "warm", "soft", "sweet"]
NEGATIVE_WORDS = ["bad", "sad", "lost", "wrong", "hate", "fear", "terrible", "awful", "angry", "hurt",
                   "scared", "worried", "sorry", "hard", "cold", "dark", "alone", "tired", "sick", "dead"]


def load_sae(config: dict, device: str):
    """Load SAE from checkpoint or pretrained."""
    sae_config = config.get("sae", {})

    # Try to load pretrained first
    if sae_config.get("use_pretrained", False):
        model_name = config.get("model", {}).get("pretrained", "tdooms/ts-medium")
        repo = f"{model_name}-scope"
        layer = sae_config.get("layer", 2)
        point = sae_config.get("point", "mlp-out")
        expansion = sae_config.get("expansion", 4)
        k = sae_config.get("k", 30)

        print(f"Loading pretrained SAE from {repo}")
        sae = SAE.from_pretrained(repo, point=(point, layer), expansion=expansion, k=k)
        return sae.to(device)

    # Load from checkpoint
    checkpoint_path = sae_config.get("checkpoint")
    if checkpoint_path:
        print(f"Loading SAE from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        sae_cfg = SAEConfig(**ckpt["sae_config"])
        sae = SAE(sae_cfg)
        sae.load_state_dict(ckpt["sae_state_dict"])
        return sae.to(device)

    raise ValueError("Must specify either use_pretrained=True or checkpoint path")


@torch.no_grad()
def collect_activations_with_patterns(model, sae, dataloader, layer: int, point: str, device: str, n_samples: int):
    """
    Collect SAE activations and classify samples by negation patterns.

    Returns:
        feature_activations: dict mapping pattern -> mean activation per feature
        sample_counts: dict mapping pattern -> count
    """
    # Use PyTorch hooks instead of nnsight (more reliable with custom models)
    mlp_output_cache = {}

    def capture_mlp_output(module, input, output):
        mlp_output_cache["output"] = output

    # Register hook on the target MLP layer
    target_mlp = model.transformer.h[layer].mlp
    hook = target_mlp.register_forward_hook(capture_mlp_output)

    # Accumulators for each pattern
    patterns = ["neg_positive", "neg_negative", "baseline"]
    accum = {p: [] for p in patterns}
    counts = {p: 0 for p in patterns}

    model.eval()
    total_processed = 0

    try:
        for batch in tqdm(dataloader, desc="Collecting activations"):
            if total_processed >= n_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)

            # Forward pass to capture MLP output via hook
            _ = model(input_ids)
            mlp_out = mlp_output_cache["output"]

            # Flatten to [batch * seq, d_model]
            acts = mlp_out.reshape(-1, mlp_out.shape[-1])

            # Encode through SAE
            _, sae_acts = sae(acts)

            # Mean activation per sample (average over sequence)
            batch_size, seq_len = input_ids.shape
            sae_acts_reshaped = sae_acts.reshape(batch_size, seq_len, -1)
            # Mask padded positions
            mask = attention_mask.unsqueeze(-1).float()
            mean_acts = (sae_acts_reshaped * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Classify samples by pattern
            for i, ids in enumerate(input_ids):
                text = model.tokenizer.decode(ids.tolist()).lower()

                has_negation = any(neg in text for neg in NEGATION_WORDS)
                has_positive = any(pos in text for pos in POSITIVE_WORDS)
                has_negative = any(neg in text for neg in NEGATIVE_WORDS)

                if has_negation and has_positive and not has_negative:
                    pattern = "neg_positive"
                elif has_negation and has_negative and not has_positive:
                    pattern = "neg_negative"
                elif not has_negation:
                    pattern = "baseline"
                else:
                    continue  # Skip ambiguous samples

                accum[pattern].append(mean_acts[i].cpu())
                counts[pattern] += 1

            total_processed += batch_size

    finally:
        # Remove the hook
        hook.remove()

    # Compute mean activations per pattern
    feature_activations = {}
    for pattern in patterns:
        if accum[pattern]:
            feature_activations[pattern] = torch.stack(accum[pattern]).mean(dim=0)
        else:
            feature_activations[pattern] = None

    return feature_activations, counts


def find_negation_features(feature_activations: dict, top_k: int = 20):
    """
    Find SAE features that differentially activate on negation patterns.

    Returns:
        results: dict with top features for each negation pattern
    """
    baseline = feature_activations.get("baseline")
    neg_pos = feature_activations.get("neg_positive")
    neg_neg = feature_activations.get("neg_negative")

    if baseline is None or neg_pos is None or neg_neg is None:
        raise ValueError("Not enough samples for pattern analysis")

    # Differential activations
    neg_pos_diff = neg_pos - baseline
    neg_neg_diff = neg_neg - baseline

    # Top features for each pattern
    _, neg_pos_top_idx = neg_pos_diff.topk(top_k)
    _, neg_neg_top_idx = neg_neg_diff.topk(top_k)

    return {
        "not_positive_features": neg_pos_top_idx.tolist(),
        "not_negative_features": neg_neg_top_idx.tolist(),
        "not_positive_activations": neg_pos_diff[neg_pos_top_idx].tolist(),
        "not_negative_activations": neg_neg_diff[neg_neg_top_idx].tolist(),
    }


def check_opposing_directions(sae, feature_idx_1: int, feature_idx_2: int) -> float:
    """
    Check if two features form opposing directions in decoder space.

    Paper claim: negation features should have negative cosine similarity.
    """
    # Decoder weights represent feature directions
    dir_1 = sae.w_dec.weight[:, feature_idx_1]
    dir_2 = sae.w_dec.weight[:, feature_idx_2]

    cosine_sim = F.cosine_similarity(dir_1.unsqueeze(0), dir_2.unsqueeze(0)).item()
    return cosine_sim


def main():
    parser = argparse.ArgumentParser(description="Negation circuit discovery")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="results/language/negation_analysis.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-pretrained", action="store_true", help="Use pretrained SAEs from HuggingFace")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--n-samples", type=int, default=None, help="Override n_samples from config")
    args = parser.parse_args()

    # Auto-detect device (includes MPS support)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Setup MPS fallbacks if needed
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - some operations may use CPU fallback")

    # Load config
    config = load_config(args.config)
    config_name = config.get("name", Path(args.config).stem)

    # Override pretrained flag if specified
    if args.use_pretrained:
        config.setdefault("sae", {})["use_pretrained"] = True

    # Initialize wandb
    wandb_enabled = init_wandb(
        name=config_name,
        config=config,
        device=device,
        enabled=not args.no_wandb,
        tags=["language", "negation"],
    )

    # Run analysis with emissions tracking
    with track_emissions("bilinear-mlp-reproduction") as tracker:
        # Use LanguageContext for unified model/SAE loading
        ctx = LanguageContext(config, device)
        model = ctx.model
        model_name = ctx.model_name
        layer = ctx.layer
        point = ctx.output_sae_config.name

        # Load SAE via context
        sae = ctx.get_sae("mlp-out")

        # Create dataloader
        print("Loading TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        # CLI argument overrides config
        n_samples = args.n_samples if args.n_samples else config.get("analysis", {}).get("n_samples", 50000)
        print(f"Using n_samples: {n_samples}")
        dataset = dataset.select(range(min(n_samples, len(dataset))))

        def tokenize(examples):
            return model.tokenizer(
                examples["text"],
                truncation=True,
                max_length=model.config.n_ctx,
                padding="max_length",
                return_tensors="pt",
            )

        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataset.set_format("torch")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Collect activations
        print(f"Analyzing activations at ({point}, layer {layer})...")
        feature_activations, counts = collect_activations_with_patterns(
            model, sae, dataloader, layer, point, device, n_samples
        )

        print(f"\nSample counts:")
        for pattern, count in counts.items():
            print(f"  {pattern}: {count}")

        # Find negation features
        top_k = config.get("analysis", {}).get("top_k", 20)
        results = find_negation_features(feature_activations, top_k)
        results["sample_counts"] = counts

        # Check opposing directions for top features
        cosine_sim = None
        if results["not_positive_features"] and results["not_negative_features"]:
            feat_pos = results["not_positive_features"][0]
            feat_neg = results["not_negative_features"][0]
            cosine_sim = check_opposing_directions(sae, feat_pos, feat_neg)

            results["top_pair_analysis"] = {
                "not_positive_feature": feat_pos,
                "not_negative_feature": feat_neg,
                "cosine_similarity": cosine_sim,
                "opposing_directions": cosine_sim < 0,
                "paper_claim": "Negation features should have negative cosine similarity",
            }

            print(f"\n{'='*60}")
            print(f"TOP NEGATION FEATURE PAIR")
            print(f"{'='*60}")
            print(f"Not + Positive feature: {feat_pos}")
            print(f"Not + Negative feature: {feat_neg}")
            print(f"Cosine similarity: {cosine_sim:.4f}")
            print(f"Opposing directions: {cosine_sim < 0}")
            print(f"{'='*60}")

    # Build metrics for results and wandb
    results["metrics"] = {
        "wall_time_seconds": tracker.result.wall_time_seconds,
        "co2_kg": tracker.result.emissions_kg,
    }
    
    # Root-level emissions for validation script compatibility
    results["emissions"] = {
        "co2_kg": tracker.result.emissions_kg,
        "wall_time_hours": tracker.result.wall_time_hours,
        "gpu_hours": tracker.result.gpu_hours,
    }

    # Finalize wandb with results
    if wandb_enabled:
        extra_summary = {
            "n_samples": n_samples,
            "not_positive_feature": results.get("not_positive_features", [None])[0],
            "not_negative_feature": results.get("not_negative_features", [None])[0],
        }
        if cosine_sim is not None:
            extra_summary["cosine_similarity"] = cosine_sim
            extra_summary["opposing_directions"] = cosine_sim < 0
        finish_wandb(tracker.result, extra_summary=extra_summary)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
