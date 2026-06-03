"""
Training script for Bilinear MLP experiments (Section 4: Vision).

Supports: 
- Dense models: MNIST, Fashion-MNIST, EMNIST Letters/Digits with optional CoM normalization
- CP models: CP-decomposed bilinear layers with rank control (Extension CP)

Usage (Dense mode - default):
    python src/train.py --config configs/mnist_dense_full.yaml --seed 42
    python src/train.py --config configs/mnist_dense_none.yaml --seed 42 --no-wandb --epochs 2
    python src/train.py --config configs/emnist_letters_regularized.yaml --seed 42

Usage (CP mode):
    python src/train.py --config configs/mnist_cp_r32.yaml --seed 42
    python src/train.py --mode cp --rank 32 --seed 42
    python src/train.py --mode cp --rank 32 --cp-init-mode gated --seed 42
"""

import sys
from pathlib import Path
import argparse
import torch

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bilinear-decomposition-main"))

from image.model import Model, Config

from src.paths import MNIST_CHECKPOINTS, FASHION_CHECKPOINTS, EXTENSION_CP_CHECKPOINTS
from src.models.cp_model import CPImageModel
from src.utils import (
    get_device,
    load_config,
    set_seed,
    track_emissions,
    init_wandb,
    finish_wandb,
    setup_mps_fallbacks,
    is_mps_device,
)
from src.training import (
    apply_variance_corrected_init,
    create_noise_transform,
    log_training_history,
    log_spectral_metrics,
    save_checkpoint,
)


def train_vision_model(config: dict, seed: int, device: str, epochs: int):
    """
    Train a bilinear vision model (dense or CP mode).

    Args:
        config: Experiment configuration
        seed: Random seed
        device: Device to train on
        epochs: Number of training epochs

    Returns:
        Tuple of (model, history, eigenvalues, eigenvectors)
    """
    set_seed(seed)
    
    # Get model mode (dense or cp)
    mode = config.get('model', {}).get('mode', 'dense')
    
    # Setup MPS fallbacks if needed (for CP eigendecomposition)
    if is_mps_device(device):
        setup_mps_fallbacks()
        if mode == 'cp':
            print("MPS device detected - using CPU fallback for eigendecomposition")
    
    # Get CoM normalization setting (default False for backward compatibility)
    apply_com = config.get('data', {}).get('apply_com', False)
    
    # Load dataset using unified data module
    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    
    if dataset_name == 'mnist':
        print("Loading MNIST data...")
        from src.data import MNIST
        train_data = MNIST(train=True, device=device, apply_com=apply_com)
        test_data = MNIST(train=False, device=device, apply_com=apply_com)
        d_output = 10
    elif dataset_name == 'fashion_mnist':
        print("Loading Fashion-MNIST data...")
        from src.data import FashionMNIST
        train_data = FashionMNIST(train=True, device=device, apply_com=apply_com)
        test_data = FashionMNIST(train=False, device=device, apply_com=apply_com)
        d_output = 10
    elif dataset_name == 'emnist_letters':
        print("Loading EMNIST Letters data...")
        from src.data import EMNISTLetters
        train_data = EMNISTLetters(train=True, device=device, apply_com=apply_com)
        test_data = EMNISTLetters(train=False, device=device, apply_com=apply_com)
        d_output = 26
    elif dataset_name == 'emnist_digits':
        print("Loading EMNIST Digits data...")
        from src.data import EMNISTDigits
        train_data = EMNISTDigits(train=True, device=device, apply_com=apply_com)
        test_data = EMNISTDigits(train=False, device=device, apply_com=apply_com)
        d_output = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if apply_com:
        print("  Center-of-Mass normalization applied")

    # Get common parameters
    d_hidden = config['model']['d_hidden']
    lr = config['training'].get('lr', 1e-3)
    weight_decay = config['regularization']['weight_decay']

    if mode == 'dense':
        # === DENSE MODEL (Original paper reproduction) ===
        model_config = Config(
            epochs=epochs,
            d_hidden=d_hidden,
            d_output=d_output,
            wd=weight_decay,
            lr=lr,
            seed=seed,
        )
        model = Model(model_config).to(device)

        # Apply variance-corrected initialization for Rich Training regime
        variance_corrected = config.get('model', {}).get('variance_corrected_init', False)
        if variance_corrected:
            print("Applying variance-corrected initialization (Rich Training regime):")
            apply_variance_corrected_init(model, enabled=True)

        # Create transform (noise augmentation)
        noise_std = config['regularization']['noise_std']
        transform = create_noise_transform(noise_std)

        # Train
        print(f"Training dense model for {epochs} epochs...")
        history = model.fit(train_data, test_data, transform=transform)

    elif mode == 'cp':
        # === CP MODEL (Extension CP) ===
        rank = config['model']['rank']
        cp_init_mode = config['model'].get('cp_init_mode', 'lambda')
        
        print(f"Creating CP model: rank={rank}, init_mode={cp_init_mode}")
        model = CPImageModel(
            d_hidden=d_hidden, 
            rank=rank, 
            n_classes=d_output, 
            cp_init_mode=cp_init_mode
        ).to(device)

        # Get CP-specific regularization parameters
        l1_coeff = config['regularization'].get('l1_coeff', 0.0)
        lambda_l1_coeff = config['regularization'].get('lambda_l1_coeff', 0.0)
        lambda_l0_coeff = config['regularization'].get('lambda_l0_coeff', 0.0)

        # Train (CP models use NO noise augmentation)
        print(f"Training CP model (rank={rank}) for {epochs} epochs...")
        history = model.fit(
            train_data, test_data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            l1_coeff=l1_coeff,
            lambda_l1_coeff=lambda_l1_coeff,
            lambda_l0_coeff=lambda_l0_coeff,
            transform=None,  # NO noise augmentation for CP
            verbose=True
        )
    else:
        raise ValueError(f"Unknown model mode: {mode}. Must be 'dense' or 'cp'")

    # Compute eigendecomposition (both models have .decompose() method)
    print("Computing eigendecomposition...")
    eigenvalues, eigenvectors = model.decompose()
    
    # Move to CPU if MPS for compatibility
    if is_mps_device(device):
        eigenvalues = eigenvalues.cpu()
        eigenvectors = eigenvectors.cpu()

    return model, history, eigenvalues, eigenvectors


def main():
    parser = argparse.ArgumentParser(description="Train Bilinear MLP (Dense or CP mode)")
    
    # Config and basic options
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not specified)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--checkpoint-dir", type=str, default=None, 
                        help="Checkpoint directory (default: results/vision/checkpoints for dense, results/extension_cp/checkpoints for CP)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    
    # Model options
    parser.add_argument("--mode", type=str, choices=["dense", "cp"], default=None,
                        help="Model mode: 'dense' (default) or 'cp' (overrides config)")
    parser.add_argument("--d-hidden", type=int, default=None, help="Hidden dimension (overrides config)")
    
    # CP-specific options
    parser.add_argument("--rank", type=int, default=None, help="CP rank (required for CP mode if not in config)")
    parser.add_argument("--cp-init-mode", type=str, choices=["fixed", "lambda", "gated"], default=None,
                        help="CP initialization mode (overrides config)")
    parser.add_argument("--l1-coeff", type=float, default=None, help="L1 penalty for CP factors B and C")
    parser.add_argument("--lambda-l1-coeff", type=float, default=None, help="L1 penalty for lambda vector")
    parser.add_argument("--lambda-l0-coeff", type=float, default=None, help="L0 proxy penalty for gate logits")
    
    # Training options
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay (overrides config)")
    
    # Data options
    parser.add_argument("--apply-com", type=str, choices=["true", "false"], default=None,
                        help="Override Center-of-Mass normalization (true/false)")
    
    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load config or create minimal config from CLI args
    if args.config:
        config = load_config(args.config)
        config_name_base = Path(args.config).stem
    else:
        # Create minimal config from CLI args (primarily for CP mode)
        mode = args.mode or 'cp'  # Default to CP if no config provided
        if mode == 'cp' and args.rank is None:
            raise ValueError("Must provide --rank or --config for CP mode")
        
        config = {
            'model': {
                'mode': mode,
                'd_hidden': args.d_hidden or 256,
            },
            'training': {
                'epochs': args.epochs or 100,
                'lr': args.lr or 1e-3,
            },
            'regularization': {
                'noise_std': 0.0 if mode == 'cp' else 0.5,
                'weight_decay': args.weight_decay or 0.1,
            },
            'data': {
                'dataset': 'mnist',
            },
        }
        if mode == 'cp':
            config['model']['rank'] = args.rank
            config['model']['cp_init_mode'] = args.cp_init_mode or 'lambda'
        config_name_base = f"mnist_{'cp_r' + str(args.rank) if mode == 'cp' else 'dense'}"

    # Apply CLI overrides to config
    if args.mode is not None:
        config['model']['mode'] = args.mode
    if args.d_hidden is not None:
        config['model']['d_hidden'] = args.d_hidden
    if args.rank is not None:
        config['model']['rank'] = args.rank
    if args.cp_init_mode is not None:
        config['model']['cp_init_mode'] = args.cp_init_mode
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.weight_decay is not None:
        config['regularization']['weight_decay'] = args.weight_decay
    if args.l1_coeff is not None:
        config['regularization']['l1_coeff'] = args.l1_coeff
    if args.lambda_l1_coeff is not None:
        config['regularization']['lambda_l1_coeff'] = args.lambda_l1_coeff
    if args.lambda_l0_coeff is not None:
        config['regularization']['lambda_l0_coeff'] = args.lambda_l0_coeff
    if args.apply_com is not None:
        if 'data' not in config:
            config['data'] = {}
        config['data']['apply_com'] = args.apply_com.lower() == 'true'
        print(f"  CoM override: {config['data']['apply_com']}")

    # Get final mode and epochs
    mode = config.get('model', {}).get('mode', 'dense')
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    
    # Validate CP mode has rank
    if mode == 'cp' and 'rank' not in config.get('model', {}):
        raise ValueError("CP mode requires 'rank' in config or --rank argument")

    # Determine checkpoint directory and naming
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        dataset_name = config.get('data', {}).get('dataset', 'mnist')
        if mode == 'cp':
            checkpoint_dir = EXTENSION_CP_CHECKPOINTS
        elif dataset_name == 'fashion':
            checkpoint_dir = FASHION_CHECKPOINTS
        else:
            checkpoint_dir = MNIST_CHECKPOINTS
    
    # Determine config name for checkpoint
    if mode == 'cp':
        rank = config['model']['rank']
        cp_init_mode = config['model'].get('cp_init_mode', 'lambda')
        config_name = f"mnist_cp_r{rank}_{cp_init_mode}"
    else:
        config_name = config_name_base

    # Determine dataset for tagging
    dataset_name = config.get('data', {}).get('dataset', 'mnist')
    tags = ["vision", dataset_name]
    if mode == 'cp':
        tags.append("cp")

    # Initialize wandb
    wandb_enabled = init_wandb(
        name=f"{config_name}_seed{args.seed}",
        config={**config, "seed": args.seed},
        device=device,
        enabled=not args.no_wandb,
        tags=tags,
    )

    # Train with tracking
    print(f"Training {config_name} with seed {args.seed}...")
    with track_emissions("bilinear-mlp-reproduction") as tracker:
        model, history, eigenvalues, eigenvectors = train_vision_model(
            config, args.seed, device, epochs
        )

    # Log training history (per-epoch metrics)
    log_training_history(history, wandb_enabled)

    # Log spectral metrics (comprehensive)
    spectral_metrics = log_spectral_metrics(eigenvalues, wandb_enabled)

    # Save checkpoint using consolidated function
    checkpoint_path = checkpoint_dir / f"{config_name}_seed{args.seed}.pt"
    checkpoint = save_checkpoint(
        checkpoint_path, config, model, history,
        eigenvalues, eigenvectors, args.seed, epochs,
        emissions={
            'wall_time_hours': tracker.result.wall_time_hours,
            'wall_time_seconds': tracker.result.wall_time_seconds,
            'gpu_hours': tracker.result.gpu_hours,
            'co2_kg': tracker.result.emissions_kg,
        }
    )
    print(f"Checkpoint saved to {checkpoint_path}")

    # Finalize wandb with all metrics
    if wandb_enabled:
        extra_summary = {
            "final_train_acc": checkpoint['metrics']['train_acc'],
            "final_val_acc": checkpoint['metrics']['val_acc'],
            "effective_rank": checkpoint['metrics']['effective_rank'],
            **spectral_metrics,  # Include all spectral metrics
        }
        finish_wandb(tracker.result, extra_summary=extra_summary)

    # Print summary
    metrics = checkpoint['metrics']
    result = tracker.result
    print(f"\n{'='*50}")
    print(f"Config: {config_name}")
    print(f"Mode: {mode}")
    print(f"Seed: {args.seed}")
    if mode == 'cp':
        print(f"CP Rank: {config['model']['rank']}")
        print(f"CP Init Mode: {config['model'].get('cp_init_mode', 'lambda')}")
    print(f"Final Val Accuracy: {metrics['val_acc']:.4f}")
    print(f"Effective Rank: {metrics['effective_rank']:.2f}")
    print(f"Wall Time: {result.wall_time_hours*60:.1f} minutes")
    if torch.cuda.is_available():
        print(f"GPU Hours: {result.gpu_hours:.3f}")
    print(f"CO2 (kg): {result.emissions_kg:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
