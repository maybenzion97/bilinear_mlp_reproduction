#!/bin/bash
# Unified vision experiment runner (Section 4)
#
# This script consolidates all vision-related experiments:
# - Base training (4 configs x 5 seeds for MNIST + Fashion)
# - Noise sweep (Figure 4)
# - Model size sweep (Figure 5)
# - Figure generation
#
# Usage:
#   ./scripts/train/run_vision.sh train base [options]   # Train base configs
#   ./scripts/train/run_vision.sh train noise [options]  # Noise sweep
#   ./scripts/train/run_vision.sh train size [options]   # Model size sweep
#   ./scripts/train/run_vision.sh train all [options]    # All training
#   ./scripts/train/run_vision.sh figures [options]      # Generate figures
#   ./scripts/train/run_vision.sh test                   # Quick MPS test
#   ./scripts/train/run_vision.sh all [options]          # Full pipeline
#   ./scripts/train/run_vision.sh help                   # Show help
#
# Options:
#   --quick       2 epochs, 1 seed (for testing)
#   --no-wandb    Disable wandb logging
#   --no-conda    Skip conda activation (for Snellius/HPC)
#   --mnist-only  Train only MNIST (for base)
#   --fashion-only Train only Fashion-MNIST (for base)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Ensure PYTHONPATH includes project root for src module imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default values
QUICK_MODE=false
WANDB_FLAG=""
EPOCHS_OVERRIDE=""
MNIST_ONLY=false
FASHION_ONLY=false
NO_CONDA=false
SEEDS=(42 43 44 45 46)
CONFIGS=("none" "noise" "wd" "full")

# Parse global options and extract command
COMMAND=""
SUBCOMMAND=""
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS_OVERRIDE="--epochs 2"
            SEEDS=(42)  # Only one seed in quick mode
            shift
            ;;
        --no-wandb)
            WANDB_FLAG="--no-wandb"
            shift
            ;;
        --no-conda)
            NO_CONDA=true
            shift
            ;;
        --mnist-only)
            MNIST_ONLY=true
            shift
            ;;
        --fashion-only)
            FASHION_ONLY=true
            shift
            ;;
        train|figures|test|all|help)
            if [ -z "$COMMAND" ]; then
                COMMAND=$1
            else
                REMAINING_ARGS+=("$1")
            fi
            shift
            ;;
        base|noise|size|challenge|adversarial|all)
            SUBCOMMAND=$1
            shift
            ;;
        --section|--sections)
            REMAINING_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default command
COMMAND=${COMMAND:-help}

# Check for conda environment (skip if --no-conda flag is set)
activate_conda() {
    if $NO_CONDA; then
        return 0  # Skip conda activation on Snellius/HPC
    fi
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate bilinear_mlp_cpu 2>/dev/null || conda activate bilinear_mlp 2>/dev/null || echo "Warning: Could not activate conda env"
    fi
}

# Print header
print_header() {
    echo "=========================================="
    echo "Vision Experiments"
    echo "=========================================="
    echo "Command: $COMMAND ${SUBCOMMAND:-}"
    if $QUICK_MODE; then echo "Mode: QUICK (2 epochs, 1 seed)"; fi
    if [ -n "$WANDB_FLAG" ]; then echo "wandb: disabled"; fi
    echo "=========================================="
    echo ""
}

# --- TRAIN BASE: 4 configs x 5 seeds ---
train_base() {
    print_header
    activate_conda
    
    local datasets=()
    if $MNIST_ONLY; then
        datasets=("mnist")
    elif $FASHION_ONLY; then
        datasets=("fashion")
    else
        datasets=("mnist" "fashion")
    fi
    
    for dataset in "${datasets[@]}"; do
        local ckpt_dir="checkpoints/vision/mnist"
        if [ "$dataset" == "fashion" ]; then
            ckpt_dir="checkpoints/vision/fashion"
        fi
        mkdir -p "$ckpt_dir"
        
        echo ">>> Training $dataset..."
        
        for config in "${CONFIGS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo ""
                echo ">>> [$dataset] Config: $config, Seed: $seed"
                
                python src/train.py \
                    --config "configs/${dataset}_dense_${config}.yaml" \
                    --seed "$seed" \
                    --checkpoint-dir "$ckpt_dir" \
                    $WANDB_FLAG \
                    $EPOCHS_OVERRIDE
            done
        done
    done
    
    echo ""
    echo "Base training complete!"
}

# --- TRAIN NOISE: Noise sweep (Figure 4) ---
train_noise() {
    print_header
    activate_conda
    
    local ckpt_dir="checkpoints/vision/noise_sweep"
    mkdir -p "$ckpt_dir"
    
    local noise_levels=(0.0 0.1 0.2 0.3 0.4 0.5)
    local seed=42
    
    echo ">>> Noise Sweep (Figure 4)"
    echo "    Noise levels: ${noise_levels[*]}"
    echo ""
    
    for noise in "${noise_levels[@]}"; do
        local config="configs/sweeps/mnist_noise_${noise}.yaml"
        
        echo ">>> Training with noise_std = $noise"
        
        python src/train.py \
            --config "$config" \
            --seed "$seed" \
            --checkpoint-dir "$ckpt_dir" \
            $WANDB_FLAG \
            $EPOCHS_OVERRIDE
    done
    
    echo ""
    echo "Noise sweep complete!"
    echo "To generate Figure 4: ./scripts/train/run_vision.sh figures --section regularization"
}

# --- TRAIN SIZE: Model size sweep (Figure 5) ---
train_size() {
    print_header
    activate_conda
    
    local ckpt_dir="checkpoints/vision/size_sweep"
    mkdir -p "$ckpt_dir"
    
    local sizes=(30 50 100 300 500 1000)
    
    echo ">>> Model Size Sweep (Figure 5)"
    echo "    Sizes: ${sizes[*]}"
    echo "    Seeds: ${SEEDS[*]}"
    echo ""
    
    for size in "${sizes[@]}"; do
        local config="configs/sweeps/mnist_size_${size}.yaml"
        
        for seed in "${SEEDS[@]}"; do
            echo ">>> Training d_hidden = $size, seed = $seed"
            
            python src/train.py \
                --config "$config" \
                --seed "$seed" \
                --checkpoint-dir "$ckpt_dir" \
                $WANDB_FLAG \
                $EPOCHS_OVERRIDE
        done
    done
    
    echo ""
    echo "Size sweep complete!"
    echo "To generate Figure 5: ./scripts/train/run_vision.sh figures --section truncation_similarity"
}

# --- TRAIN CHALLENGE (Figure 6) ---
train_challenge() {
    print_header
    activate_conda
    
    local ckpt_dir="checkpoints/vision/challenge"
    mkdir -p "$ckpt_dir"
    
    echo ">>> Challenge Task Training (Figure 6)"
    echo "    Using train_challenge_variants.py for binary classification task"
    echo "    Seeds: ${SEEDS[*]}"
    echo ""
    
    for seed in "${SEEDS[@]}"; do
        echo ">>> Training challenge task variants, seed = $seed"
        
        # Use the dedicated challenge training script (binary classification)
        python scripts/train/train_challenge_variants.py \
            --seed "$seed" \
            --out-dir "$ckpt_dir" \
            $EPOCHS_OVERRIDE
    done
    
    echo ""
    echo "Challenge training complete!"
    echo "To generate Figure 6: ./scripts/train/run_vision.sh figures --section challenge"
}

# --- TRAIN ADVERSARIAL (Figure 7) ---
train_adversarial() {
    print_header
    activate_conda
    
    local ckpt_dir="checkpoints/vision/mnist"
    mkdir -p "$ckpt_dir"
    
    echo ">>> Adversarial Training (Figure 7)"
    echo "    Config: noise_std=0.15"
    echo "    Seeds: ${SEEDS[*]}"
    echo ""
    
    for seed in "${SEEDS[@]}"; do
        echo ">>> Training noise015, seed = $seed"
        
        python src/train.py \
            --config "configs/mnist_dense_noise015.yaml" \
            --seed "$seed" \
            --checkpoint-dir "$ckpt_dir" \
            $WANDB_FLAG \
            $EPOCHS_OVERRIDE
    done
    
    echo ""
    echo "Adversarial training complete!"
    echo "To generate Figure 7: ./scripts/train/run_vision.sh figures --section adversarial"
}

# --- TRAIN ALL ---
train_all() {
    echo ">>> Running all training experiments..."
    echo ""
    train_base
    echo ""
    train_noise
    echo ""
    train_size
    echo ""
    train_challenge
    echo ""
    train_adversarial
    echo ""
    echo "All training complete!"
}

# --- FIGURES ---
generate_figures() {
    print_header
    activate_conda
    
    echo ">>> Generating figures..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/generate_vision_figures.py "${REMAINING_ARGS[@]}"
    
    echo ""
    echo "Figure generation complete!"
}

# --- TEST ---
run_test() {
    echo "=========================================="
    echo "Quick MPS Test (2 epochs)"
    echo "=========================================="
    echo ""
    
    activate_conda
    
    echo "1. Checking MPS availability..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print(f'MPS built: {torch.backends.mps.is_built()}')
    x = torch.randn(10, 10, device='mps')
    y = x @ x.T
    print(f'MPS tensor test: OK')
"
    echo ""
    
    echo "2. Testing training (2 epochs, MNIST, no regularization)..."
    mkdir -p results/test
    python src/train.py \
        --config configs/mnist_dense_none.yaml \
        --seed 42 \
        --epochs 2 \
        --checkpoint-dir results/test \
        --no-wandb
    
    echo ""
    echo "3. Verifying checkpoint..."
    python -c "
import torch
ckpt = torch.load('results/test/mnist_dense_none_seed42.pt', map_location='cpu', weights_only=False)
print(f'Checkpoint keys: {list(ckpt.keys())}')
print(f'Eigenvalues shape: {ckpt[\"eigenvalues\"].shape}')
print(f'Eigenvectors shape: {ckpt[\"eigenvectors\"].shape}')
print(f'Val accuracy: {ckpt[\"metrics\"][\"val_acc\"]:.4f}')
print(f'Effective rank: {ckpt[\"metrics\"][\"effective_rank\"]:.1f}')
"
    
    echo ""
    echo "=========================================="
    echo "MPS TEST PASSED!"
    echo "=========================================="
}

# --- ALL ---
run_all() {
    echo ">>> Running full vision pipeline..."
    echo ""
    train_all
    echo ""
    generate_figures
    echo ""
    echo "Full pipeline complete!"
}

# --- HELP ---
show_help() {
    echo "Usage: ./scripts/train/run_vision.sh <command> [subcommand] [options]"
    echo ""
    echo "Commands:"
    echo "  train base       Train 4 configs (none/noise/wd/full) x 5 seeds"
    echo "  train noise      Noise sweep for Figure 4 (6 noise levels)"
    echo "  train size       Model size sweep for Figure 5 (6 sizes x 5 seeds)"
    echo "  train challenge  Challenge task for Figure 6"
    echo "  train adversarial Noise015 models for Figure 7"
    echo "  train all        All training experiments (base + noise + size + challenge + adversarial)"
    echo "  figures          Generate all figures from checkpoints"
    echo "  extension_cross_dataset       Run cross-dataset experiments (delegates to run_extension_cross_dataset.sh)"
    echo "  test             Quick 2-epoch MPS verification"
    echo "  all              Full pipeline (train all + figures)"
    echo "  help             Show this help message"
    echo ""
    echo "Options:"
    echo "  --quick        2 epochs, 1 seed (for testing)"
    echo "  --no-wandb     Disable wandb logging"
    echo "  --mnist-only   Train only MNIST (for 'train base')"
    echo "  --fashion-only Train only Fashion-MNIST (for 'train base')"
    echo ""
    echo "Examples:"
    echo "  ./scripts/train/run_vision.sh test                    # Quick MPS test"
    echo "  ./scripts/train/run_vision.sh train base --quick      # Quick base training"
    echo "  ./scripts/train/run_vision.sh train noise             # Full noise sweep"
    echo "  ./scripts/train/run_vision.sh figures                 # Generate all figures"
    echo "  ./scripts/train/run_vision.sh all --no-wandb          # Full pipeline, no wandb"
    echo ""
    echo "Figure generation options:"
    echo "  --section <name>  Generate specific section:"
    echo "                    regularization, truncation_similarity,"
    echo "                    challenge, adversarial, appendix, hub"
}

# --- EXTENSION CROSS-DATASET ---
run_extension_cross_dataset() {
    echo ">>> Delegating to cross-dataset runner..."
    exec "$SCRIPT_DIR/run_extension_cross_dataset.sh" "${REMAINING_ARGS[@]}"
}

# --- MAIN ---
case $COMMAND in
    train)
        SUBCOMMAND=${SUBCOMMAND:-base}
        case $SUBCOMMAND in
            base)       train_base ;;
            noise)      train_noise ;;
            size)       train_size ;;
            challenge)  train_challenge ;;
            adversarial) train_adversarial ;;
            all)        train_all ;;
            *)
                echo "Unknown train subcommand: $SUBCOMMAND"
                echo "Valid subcommands: base, noise, size, challenge, adversarial, all"
                exit 1
                ;;
        esac
        ;;
    figures)
        generate_figures
        ;;
    test)
        run_test
        ;;
    all)
        run_all
        ;;
    extension_cross_dataset)
        run_extension_cross_dataset
        ;;
    help|*)
        show_help
        ;;
esac
