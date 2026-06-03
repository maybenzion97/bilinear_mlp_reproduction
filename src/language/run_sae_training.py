"""
SAE Training wrapper using original paper code (Section 5: Language).

Uses the SAE implementation from bilinear-decomposition-main/sae/sae.py

Usage:
    python src/language/run_sae_training.py --config configs/language_sae.yaml
    python src/language/run_sae_training.py --config configs/language_sae.yaml --no-wandb
"""

import sys
from pathlib import Path
import argparse
import warnings

# Suppress torchvision image extension warning (libjpeg not needed for our use case)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ORIG_PATH = PROJECT_ROOT / "bilinear-decomposition-main"
sys.path.insert(0, str(ORIG_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

from language.transformer import Transformer
from sae.sae import SAE, SAEConfig
from datasets import load_dataset

from src.utils import (
    get_device,
    load_config,
    track_emissions,
    init_wandb,
    finish_wandb,
    setup_mps_fallbacks,
    is_mps_device,
)


def create_dataset(tokenizer, config: dict, device: str):
    """Create TinyStories dataset for SAE training.

    Note: Returns the dataset directly (not a DataLoader) because the SAE sampler
    wraps it in its own DataLoader internally.
    """
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    n_samples = config.get("data", {}).get("n_samples", 100000)
    if n_samples and n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))
        print(f"Using {n_samples} samples")

    n_ctx = config.get("sae", {}).get("n_ctx", 256)

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

    return dataset


def train_sae(config: dict, device: str):
    """
    Train SAE using original paper code.

    Args:
        config: Experiment configuration
        device: Device to train on

    Returns:
        Tuple of (sae, model, sae_config_dict)
    """
    # Load pretrained bilinear transformer
    model_name = config.get("model", {}).get("pretrained", "tdooms/ts-medium")
    print(f"Loading pretrained model: {model_name}")
    model = Transformer.from_pretrained(model_name, device=device)

    # Create dataset (not DataLoader - SAE sampler wraps it internally)
    train_dataset = create_dataset(model.tokenizer, config, device)

    # Create validation batch from first few samples
    batch_size = config.get("training", {}).get("batch_size", 32)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_iter = iter(val_loader)
    validate = next(val_iter)
    validate = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in validate.items()}
    # Add labels for loss computation (labels = input_ids for LM next-token prediction)
    validate["labels"] = validate["input_ids"]

    # SAE configuration
    sae_config = config.get("sae", {})
    layer = sae_config.get("layer", 2)
    point_name = sae_config.get("point", "mlp-out")
    expansion = sae_config.get("expansion", 8)
    k = sae_config.get("k", 32)

    training_config = config.get("training", {})
    n_buffers = training_config.get("n_buffers", 100)
    lr = training_config.get("lr", 1e-4)

    # Create SAE using original code
    print(f"Creating SAE at point=({point_name}, {layer}), expansion={expansion}, k={k}")
    sae_cfg = SAEConfig(
        point=(point_name, layer),
        target=(point_name, layer),
        expansion=expansion,
        k=k,
        d_model=model.config.d_model,
        n_ctx=model.config.n_ctx,
        lr=lr,
        n_buffers=n_buffers,
        in_batch=training_config.get("batch_size", 32),
        out_batch=training_config.get("out_batch", 4096),
        n_batches=training_config.get("n_batches", 256),
    )

    # Add device to config for sampler (original code defaults to cuda)
    sae_cfg.device = device

    sae = SAE(sae_cfg).to(device)

    # Train SAE (pass dataset, not DataLoader - SAE sampler wraps it internally)
    print(f"Training SAE for {n_buffers} buffers...")
    sae.fit(model, train_dataset, validate, project=None)

    # Config dict for checkpoint
    sae_config_dict = {
        "point": [point_name, layer],
        "target": [point_name, layer],
        "expansion": expansion,
        "k": k,
        "d_model": model.config.d_model,
        "n_ctx": model.config.n_ctx,
        "lr": lr,
        "n_buffers": n_buffers,
    }

    return sae, model, sae_config_dict


def save_sae_checkpoint(
    path: Path,
    sae,
    config: dict,
    sae_config_dict: dict,
    model_name: str,
):
    """Save SAE checkpoint."""
    sae_config = config.get("sae", {})
    layer = sae_config.get("layer", 2)
    point_name = sae_config.get("point", "mlp-out")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config,
        "sae_state_dict": sae.state_dict(),
        "sae_config": sae_config_dict,
        "model_name": model_name,
        "layer": layer,
        "point": point_name,
    }, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train SAE for Section 5")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint-dir", type=str, default="results/language")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Setup MPS fallbacks if needed
    if is_mps_device(device):
        setup_mps_fallbacks()
        print("MPS device detected - some operations may use CPU fallback")

    config = load_config(args.config)
    config_name = config.get("name", Path(args.config).stem)

    # Initialize wandb
    wandb_enabled = init_wandb(
        name=config_name,
        config=config,
        device=device,
        enabled=not args.no_wandb,
        tags=["language", "sae"],
    )

    # Train with tracking
    with track_emissions("bilinear-mlp-reproduction") as tracker:
        sae, model, sae_config_dict = train_sae(config, device)

    # Finalize wandb
    if wandb_enabled:
        finish_wandb(tracker.result)

    # Save checkpoint
    sae_config = config.get("sae", {})
    layer = sae_config.get("layer", 2)
    point_name = sae_config.get("point", "mlp-out")
    point_str = point_name.replace("-", "_")

    checkpoint_path = Path(args.checkpoint_dir) / f"sae_{point_str}_layer{layer}.pt"
    model_name = config.get("model", {}).get("pretrained", "tdooms/ts-medium")
    save_sae_checkpoint(checkpoint_path, sae, config, sae_config_dict, model_name)

    # Print summary
    result = tracker.result
    print(f"\n{'='*60}")
    print(f"SAE Training Complete")
    print(f"{'='*60}")
    print(f"Config: {config_name}")
    print(f"Point: ({point_name}, {layer})")
    print(f"Expansion: {sae_config.get('expansion', 8)}, k: {sae_config.get('k', 32)}")
    print(f"Wall Time: {result.wall_time_hours*60:.1f} minutes")
    print(f"CO2 (kg): {result.emissions_kg:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
