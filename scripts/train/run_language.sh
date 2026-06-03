#!/bin/bash
# Unified language experiment runner (Section 5)
#
# This script consolidates all language-related experiments:
# - Figure 9: Correlation sweep (ts-medium, fw-small, fw-medium)
# - Figure 8: Negation circuit visualization
# - Figure 10: SAE training time analysis
# - Negation discovery
# - Interaction analysis
#
# Usage:
#   ./scripts/train/run_language.sh figure9 [options]     # Correlation sweep
#   ./scripts/train/run_language.sh figure8 [options]     # Negation circuit viz
#   ./scripts/train/run_language.sh figure10 [options]    # SAE training time
#   ./scripts/train/run_language.sh negation [options]    # Negation discovery
#   ./scripts/train/run_language.sh interaction [options] # Interaction analysis
#   ./scripts/train/run_language.sh figures               # Generate all figures
#   ./scripts/train/run_language.sh test                  # Quick MPS test
#   ./scripts/train/run_language.sh all [options]         # Full pipeline
#   ./scripts/train/run_language.sh help                  # Show help
#
# Options:
#   --quick       Reduced samples/features for testing
#   --device      cpu|mps|cuda (default: auto-detect)
#   --no-wandb    Disable wandb logging
#   --no-conda    Skip conda activation (for Snellius/HPC)
#   --model       Specific model for figure9/negation/interaction
#   --sequential  Run figure9 sequentially (memory-safe)
#   --streaming   Use streaming Q computation for CUDA (Figure 8)
#   --chunk-size  Chunk size for streaming (256=1GB, 128=0.5GB)
#   --fig8-dataset Dataset for Figure 8 (tinystories|fineweb|fineweb-16k)
#   --batch-size  Batch size for figure9 validation
#   --max-batches Max batches for figure9 validation
#   --exact-accum Use exact streaming accumulation for Figure 9
#   --precompute  Precompute eigenpairs in 'all' pipeline
#   --exact-chunk Feature chunk size for exact streaming

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Ensure PYTHONPATH includes project root for src module imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default values
QUICK_MODE=false
DEVICE=""
WANDB_FLAG="--no-wandb"
MODEL=""
SEQUENTIAL=false
FEATURE=3834
NO_CONDA=false
STREAMING=false
CHUNK_SIZE=256
BATCH_SIZE=""
MAX_BATCHES=""
EXACT_ACCUM=false
PRECOMPUTE=false
EXACT_CHUNK=""
METRIC="pearson"
LOAD_EIGENPAIRS=""
LAYER=""
FIG8_DATASET=""

# Parse global options and extract command
COMMAND=""
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_FLAG="--no-wandb"
            shift
            ;;
        --no-conda)
            NO_CONDA=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --feature)
            FEATURE="$2"
            shift 2
            ;;
        --streaming)
            STREAMING=true
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --fig8-dataset)
            FIG8_DATASET="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-batches)
            MAX_BATCHES="$2"
            shift 2
            ;;
        --exact-accum)
            EXACT_ACCUM=true
            shift
            ;;
        --exact-chunk)
            EXACT_CHUNK="$2"
            shift 2
            ;;
        --precompute)
            PRECOMPUTE=true
            shift
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --load-eigenpairs)
            LOAD_EIGENPAIRS="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --float16)
            REMAINING_ARGS+=("$1")
            shift
            ;;
        precompute|figure9|correlation|figure8|negation-viz|figure10|sae-training|negation|interaction|figures|test|all|help)
            if [ -z "$COMMAND" ]; then
                COMMAND=$1
            else
                REMAINING_ARGS+=("$1")
            fi
            shift
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default command
COMMAND=${COMMAND:-help}

# Default device based on platform
if [ -z "$DEVICE" ]; then
    if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        DEVICE="mps"
    elif python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

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

# Best-effort device memory cleanup between stages
cleanup_memory() {
    local stage="${1:-stage}"
    echo "Clearing device memory after ${stage}..."
    python - <<'PY'
import gc
try:
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"Memory cleanup warning: {e}")
gc.collect()
PY
    # Give allocator/driver a moment to release
    sleep 3
}

# Print header
print_header() {
    echo "=========================================="
    echo "Language Experiments"
    echo "=========================================="
    echo "Command: $COMMAND"
    echo "Device: $DEVICE"
    echo "Metric: $METRIC"
    if $QUICK_MODE; then echo "Mode: QUICK"; fi
    if $STREAMING; then echo "Streaming: enabled (chunk_size=$CHUNK_SIZE)"; fi
    if [ -n "$MODEL" ]; then echo "Model: $MODEL"; fi
    echo "=========================================="
    echo ""
}

# --- FIGURE 9: Correlation Sweep ---
run_figure9() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    # Model configurations: "model layer expansion dataset"
    # ts-medium trained on TinyStories, fw-* trained on FineWeb
    declare -a MODELS=(
        "ts-medium 4 4 tinystories"
        "fw-small 8 4 fineweb"
        "fw-medium 10 8 fineweb"  # Layer 10 = 2/3 depth for 16-layer model
    )
    
    # Filter by model if specified
    if [ -n "$MODEL" ] && [ "$MODEL" != "all" ]; then
        case $MODEL in
            ts-medium|tdooms/ts-medium)
                MODELS=("ts-medium 4 4 tinystories")
                ;;
            fw-small|tdooms/fw-small)
                MODELS=("fw-small 8 4 fineweb")
                ;;
            fw-medium|tdooms/fw-medium)
                MODELS=("fw-medium 10 8 fineweb")
                ;;
            *)
                echo "Unknown model: $MODEL"
                echo "Valid models: ts-medium, fw-small, fw-medium, all"
                exit 1
                ;;
        esac
    fi
    
    # Default: analyze ALL features; quick mode uses subset
    local n_features="all"
    local ranks="1-60"
    local n_samples="all"
    local max_batches=128
    local target_samples="all"
    local batch_size=48
    local chunk_size="$CHUNK_SIZE"
    local scatter_flag=""
    local exact_flag=""
    local exact_chunk_flag=""
    
    # MPS optimization: larger batches since Apple Silicon has unified memory
    if [[ "$DEVICE" == "mps" ]]; then
        # Keep batch size modest for memory stability on MPS
        batch_size=48
        chunk_size="$CHUNK_SIZE"
    fi

    # Allow CLI overrides for hardware tuning
    if [ -n "$BATCH_SIZE" ]; then
        batch_size="$BATCH_SIZE"
    fi
    if [ -n "$MAX_BATCHES" ]; then
        max_batches="$MAX_BATCHES"
    fi
    
    if $QUICK_MODE; then
        n_features=50
        n_samples=1000
        max_batches=30
        target_samples=500
    else
        scatter_flag="--save-scatter --max-scatter-samples 1000"
    fi
    
    if $EXACT_ACCUM; then
        exact_flag="--streaming-exact"
    fi
    if [ -n "$EXACT_CHUNK" ]; then
        exact_chunk_flag="--exact-chunk-size $EXACT_CHUNK"
    fi
    
    # Build eigenpairs flag
    local eigenpairs_flag=""
    if [ -n "$LOAD_EIGENPAIRS" ]; then
        eigenpairs_flag="--load-eigenpairs $LOAD_EIGENPAIRS"
        echo ">>> Running Figure 9 Correlation Sweep (CACHED EIGENPAIRS)"
        echo "    Using cached eigenpairs: $LOAD_EIGENPAIRS"
    else
        echo ">>> Running Figure 9 Correlation Sweep"
    fi
    echo "    Models: ${#MODELS[@]}"
    echo "    Features: $n_features"
    echo "    Ranks: $ranks"
    echo "    Metric: $METRIC"
    echo "    Samples: $n_samples, Target: $target_samples per feature (max_batches=$max_batches)"
    echo "    Batch size: $batch_size, Chunk size: $chunk_size"
    echo ""
    
    for model_config in "${MODELS[@]}"; do
        read -r model layer expansion dataset <<< "$model_config"
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Running: $model (layer=$layer, expansion=$expansion, dataset=$dataset, metric=$METRIC)"
        if [ -n "$eigenpairs_flag" ]; then
            echo "Mode: CACHED (fast iteration)"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Model-specific memory tuning (keeps token counts unchanged)
        local model_chunk_size="$chunk_size"
        local model_exact_chunk_size="$EXACT_CHUNK"
        if $EXACT_ACCUM; then
            if [[ "$model" == "fw-medium" ]]; then
                # fw-medium is the largest model; smaller chunks reduce peak memory
                model_chunk_size=128
                if [ -z "$model_exact_chunk_size" ]; then
                    model_exact_chunk_size=32
                fi
            elif [[ "$model" == "fw-small" ]]; then
                # fw-small can still hit MPS memory limits in exact mode
                model_chunk_size=128
                if [ -z "$model_exact_chunk_size" ]; then
                    model_exact_chunk_size=32
                fi
            fi
        fi
        local model_exact_chunk_flag=""
        if [ -n "$model_exact_chunk_size" ]; then
            model_exact_chunk_flag="--exact-chunk-size $model_exact_chunk_size"
        fi

        python src/language/verify_correlation.py \
            --config "configs/language_correlation_fw.yaml" \
            --model "tdooms/$model" \
            --layer "$layer" \
            --expansion "$expansion" \
            --k 30 \
            --output "results/language/correlation_$model.json" \
            --plot "results/language/correlation_$model.png" \
            --device "$DEVICE" \
            $WANDB_FLAG \
            --n-features "$n_features" \
            --ranks "$ranks" \
            --n-samples "$n_samples" \
            --max-batches "$max_batches" \
            --target-samples "$target_samples" \
            --batch-size "$batch_size" \
            --chunk-size "$model_chunk_size" \
            --metric "$METRIC" \
            --dataset "$dataset" \
            $eigenpairs_flag \
            $scatter_flag \
            $exact_flag \
            $model_exact_chunk_flag
        
        echo "Completed: $model"
        cleanup_memory "figure9/$model"
        
        if $SEQUENTIAL; then
            echo "Waiting 5s for memory cleanup..."
            sleep 5
        fi
    done
    
    echo ""
    echo "Figure 9 sweep complete!"
    echo "Results: results/language/correlation_*.json"
    echo ""
    echo "To generate combined figures:"
    echo "  python scripts/figures/generate_language_figures.py"
}

# --- PRECOMPUTE: Eigenpairs Precomputation ---
# Precomputes eigendecompositions for rapid iteration with different metrics/thresholds

run_precompute() {
    print_header
    activate_conda
    
    mkdir -p results/language/eigenpairs
    
    # Model configurations: "model layer expansion"
    # Note: fw-medium layer 7 removed (Figure 8 computes on-the-fly for 2 features)
    declare -a PRECOMPUTE_MODELS=(
        "ts-medium 4 4"   # Figure 9 (paper's ts-tiny)
        "ts-medium 5 4"   # Figure 8 (only layer with mlp-in SAE)
        "fw-small 8 4"    # Figure 9
        "fw-medium 10 8"  # Figure 9 (2/3 depth for 16-layer model)
    )
    
    # Figure 10 SAE versions (fw-medium layer 12, expansion 16)
    declare -a FIGURE_10_SAE_VERSIONS=("v0" "v1" "v2" "v3" "v4")
    
    # Check for subcommand
    local subcmd="${REMAINING_ARGS[0]:-}"
    
    local float16_flag=""
    if [[ "${REMAINING_ARGS[*]}" == *"--float16"* ]]; then
        float16_flag="--float16"
    fi
    
    local max_features=""
    if $QUICK_MODE; then
        max_features="--max-features 100"
    fi
    
    # Filter by model if specified
    if [ -n "$MODEL" ] && [ "$MODEL" != "all" ]; then
        # Parse layer from --layer argument if provided (global or remaining args)
        local layer_arg="${LAYER:-}"
        if [ -z "$layer_arg" ]; then
            for i in "${!REMAINING_ARGS[@]}"; do
                if [[ "${REMAINING_ARGS[$i]}" == "--layer" ]]; then
                    layer_arg="${REMAINING_ARGS[$((i+1))]}"
                fi
            done
        fi
        
        case $MODEL in
            ts-medium|tdooms/ts-medium)
                if [ -n "$layer_arg" ]; then
                    PRECOMPUTE_MODELS=("ts-medium $layer_arg 4")
                else
                    # Default to layer 4 for ts-medium
                    PRECOMPUTE_MODELS=("ts-medium 4 4")
                fi
                ;;
            fw-small|tdooms/fw-small)
                PRECOMPUTE_MODELS=("fw-small 8 4")
                ;;
            fw-medium|tdooms/fw-medium)
                if [ -n "$layer_arg" ]; then
                    PRECOMPUTE_MODELS=("fw-medium $layer_arg 8")
                else
                    PRECOMPUTE_MODELS=("fw-medium 10 8")
                fi
                ;;
            *)
                echo "Unknown model: $MODEL"
                echo "Valid models: ts-medium, fw-small, fw-medium, all"
                exit 1
                ;;
        esac
    fi
    
    # Handle Figure 10 precomputation (SAE versions v0-v4)
    if [ "$subcmd" == "figure10" ]; then
        echo ">>> Running Figure 10 Eigenpairs Precomputation"
        echo "    Model: fw-medium layer 12 (expansion=16)"
        echo "    SAE versions: ${FIGURE_10_SAE_VERSIONS[*]}"
        echo "    Device: $DEVICE"
        if [ -n "$float16_flag" ]; then
            echo "    Storage: float16 (50% reduction)"
        fi
        echo ""
        echo "    Estimated storage: ~180 GB total (5 versions × ~36 GB each)"
        echo "    Or ~90 GB with --float16"
        echo ""
        
        python src/language/precompute_eigenpairs.py \
            --figure10 \
            --device "$DEVICE" \
            --chunk-size "$CHUNK_SIZE" \
            $float16_flag \
            $max_features
        
        echo ""
        echo "Figure 10 precomputation complete!"
        echo "Eigenpairs saved to: results/language/eigenpairs/fw-medium/12/{v0,v1,v2,v3,v4}/"
        return
    fi
    
    if [ "$subcmd" == "all" ] || [ -z "$subcmd" ]; then
        echo ">>> Running Eigenpairs Precomputation"
        echo "    Models: ${#PRECOMPUTE_MODELS[@]}"
        echo "    Device: $DEVICE"
        if [ -n "$float16_flag" ]; then
            echo "    Storage: float16 (50% reduction)"
        fi
        echo ""
        echo "    Estimated storage:"
        echo "      ts-medium (layer 4/5): ~2 GB each"
        echo "      fw-small (layer 8): ~7 GB"
        echo "      fw-medium (layer 10): ~34 GB"
        echo "      Figure 10 (fw-medium layer 12, 5 SAE versions): ~180 GB"
        echo "      Total: ~225 GB (or ~113 GB with --float16)"
        echo ""
        echo "    Note: Run 'precompute figure10' separately for Figure 10 SAE versions"
        echo ""
        
        for model_config in "${PRECOMPUTE_MODELS[@]}"; do
            read -r model layer expansion <<< "$model_config"
            
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "Precomputing: $model (layer=$layer, expansion=$expansion)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python src/language/precompute_eigenpairs.py \
                --model "tdooms/$model" \
                --layer "$layer" \
                --expansion "$expansion" \
                --device "$DEVICE" \
                --chunk-size "$CHUNK_SIZE" \
                $float16_flag \
                $max_features
            
            echo "Completed: $model layer $layer"
        done
        
        echo ""
        echo "Precomputation complete!"
        echo "Eigenpairs saved to: results/language/eigenpairs/"
        echo ""
        echo "To use cached eigenpairs:"
        echo "  ./scripts/train/run_language.sh figure9 --load-eigenpairs auto"
    else
        echo "Unknown precompute subcommand: $subcmd"
        echo "Usage: ./scripts/train/run_language.sh precompute [all]"
        exit 1
    fi
}

# --- FIGURE 8: Negation Circuit Visualization ---
# Now supports subcommands: search, analyze, figures, all

run_figure8_search() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    local max_features=""
    local streaming_flag=""
    if $QUICK_MODE; then
        max_features="--max-features 100"
    fi
    if $STREAMING; then
        streaming_flag="--streaming --chunk-size $CHUNK_SIZE"
    fi
    local dataset_flag=""
    if [ -n "$FIG8_DATASET" ]; then
        dataset_flag="--dataset $FIG8_DATASET"
    else
        dataset_flag="--dataset fineweb"
    fi
    
    echo ">>> Running Figure 8 Circuit Search (Full 8192 features)"
    echo "    Device: $DEVICE"
    if $STREAMING; then
        echo "    Streaming: enabled (chunk_size=$CHUNK_SIZE)"
        echo "    Estimated time: ~1-2 hours on CUDA A100"
    else
        echo "    Estimated time: ~8-10 hours on MPS"
    fi
    echo ""
    
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/comprehensive_circuit_search.py \
        --device "$DEVICE" \
        --batch-size 50 \
        --save-interval 100 \
        $max_features \
        $streaming_flag
    
    echo ""
    echo "Circuit search complete!"
    echo "Output: results/language/circuit_search_complete.json"
}

run_figure8_analyze() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    echo ">>> Analyzing top circuits from search results"
    echo ""
    
    # Check if search results exist
    if [ ! -f "results/language/circuit_search_complete.json" ]; then
        echo "ERROR: Search results not found. Run 'figure8 search' first."
        exit 1
    fi
    
    # Extract top features and analyze them
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -c "
import json
from pathlib import Path

PROJECT_ROOT = Path('.')
results_file = PROJECT_ROOT / 'results/language/circuit_search_complete.json'

with open(results_file) as f:
    results = json.load(f)

# Get top 5 by AND-score
top_features = [r['feature'] for r in results['top_by_and_score'][:5]]
print(f'Top 5 features by AND-score: {top_features}')

# Save for figure generation
with open(PROJECT_ROOT / 'results/language/top_circuit_features.json', 'w') as f:
    json.dump({
        'top_5_and_score': top_features,
        'best_feature': top_features[0],
        'tutorial_feature': 3834,  # For comparison
    }, f, indent=2)
print('Saved: results/language/top_circuit_features.json')
"
    
    # Analyze best feature
    local best_feature=$(python -c "
import json
with open('results/language/circuit_search_complete.json') as f:
    results = json.load(f)
print(results['top_by_and_score'][0]['feature'])
")
    
    echo ""
    echo "Analyzing best feature: $best_feature"
    
    local streaming_flag=""
    if $STREAMING; then
        streaming_flag="--streaming --chunk-size $CHUNK_SIZE"
    fi
    
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/comprehensive_circuit_search.py \
        --device "$DEVICE" \
        --analyze "$best_feature" \
        $streaming_flag
    
    echo ""
    echo "Analysis complete!"
}

run_figure8_figures() {
    print_header
    activate_conda
    
    mkdir -p results/language/figures
    mkdir -p Report/figures/language
    
    echo ">>> Generating Figure 8 variants"
    echo ""
    
    # Check if search results exist
    if [ ! -f "results/language/circuit_search_complete.json" ]; then
        echo "WARNING: Full search results not found."
        echo "Will generate figures with available data (feature 3834 only)."
    fi
    
    # Generate all figure 8 variants
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/generate_language_figures.py --figure8-only
    
    echo ""
    echo "Figure 8 generation complete!"
    echo "Output: Report/figures/language/"
}

run_figure8_all() {
    echo ">>> Running full Figure 8 pipeline: generate -> search -> analyze -> figures"
    echo "    Estimated total time: ~8-10 hours"
    echo ""
    
    # Step 1: Generate data for both datasets (TinyStories + FineWeb-16k)
    echo "Step 1/4: Generating Figure 8 data..."
    run_figure8_generate
    echo ""
    
    # Step 2: Comprehensive circuit search
    echo "Step 2/4: Running comprehensive circuit search..."
    run_figure8_search
    echo ""
    
    # Step 3: Analyze top circuits
    echo "Step 3/4: Analyzing top circuits..."
    run_figure8_analyze
    echo ""
    
    # Step 4: Generate all figure variants
    echo "Step 4/4: Generating all figure variants..."
    run_figure8_figures
    echo ""
    echo "Full Figure 8 pipeline complete!"
}

# Figure 8 data generation for features 3834 and 751
# Generates data for both TinyStories and FineWeb-16k datasets for comparison
run_figure8_generate() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    # Figure 8 uses memory-efficient iterative eigensolver
    # Best run on MPS (unified memory) or CUDA with streaming
    local fig8_device="$DEVICE"
    
    local n_samples="all"
    local streaming_flag=""
    local batch_size=16
    if $QUICK_MODE; then
        n_samples=1000
    fi
    if $STREAMING; then
        streaming_flag="--streaming --chunk-size $CHUNK_SIZE"
    fi
    
    # Helper function to generate data for a specific dataset
    generate_for_dataset() {
        local dataset_name="$1"
        local suffix="$2"
        
        echo ""
        echo "============================================================"
        echo ">>> Generating Figure 8 data for $dataset_name"
        echo "============================================================"
        echo "    Features: 3834 (not-good) and 751 (not-bad)"
        echo "    Device: $fig8_device"
        echo "    Dataset: $dataset_name"
        echo "    Batch size: $batch_size"
        echo "    Samples: $n_samples"
        echo "    Metric: $METRIC"
        if $STREAMING; then
            echo "    Streaming: enabled (chunk_size=$CHUNK_SIZE)"
        fi
        echo "    Memory: Uses iterative eigensolver (constant memory)"
        echo ""
        
        # Generate data for feature 3834 (not-good)
        echo "--- Feature 3834 (not-good) on $dataset_name ---"
        python src/language/negation_visualization.py \
            --config configs/language_negation_fw.yaml \
            --output "results/language/figure_8_data_fw_medium${suffix}.json" \
            --feature 3834 \
            --device "$fig8_device" \
            --batch-size "$batch_size" \
            --n-samples "$n_samples" \
            --metric "$METRIC" \
            --dataset "$dataset_name" \
            $streaming_flag
        
        echo ""
        cleanup_memory "figure8/3834_$dataset_name"
        
        # Generate data for feature 751 (not-bad)
        echo "--- Feature 751 (not-bad) on $dataset_name ---"
        python src/language/negation_visualization.py \
            --config configs/language_negation_fw.yaml \
            --output "results/language/figure_8_feature751${suffix}.json" \
            --feature 751 \
            --device "$fig8_device" \
            --batch-size "$batch_size" \
            --n-samples "$n_samples" \
            --metric "$METRIC" \
            --dataset "$dataset_name" \
            $streaming_flag
        
        echo ""
        cleanup_memory "figure8/751_$dataset_name"
    }
    
    # Check if specific dataset requested via --fig8-dataset
    if [ -n "$FIG8_DATASET" ]; then
        # Generate for single requested dataset
        case "$FIG8_DATASET" in
            tinystories)
                generate_for_dataset "tinystories" ""
                ;;
            fineweb-16k)
                generate_for_dataset "fineweb-16k" "_fineweb16k"
                ;;
            *)
                echo "Unknown dataset: $FIG8_DATASET"
                echo "Supported: tinystories, fineweb-16k"
                exit 1
                ;;
        esac
    else
        # Generate for BOTH datasets (default behavior)
        echo "============================================================"
        echo ">>> Figure 8: Generating data for BOTH datasets"
        echo "    1. TinyStories (cleaner semantic clustering)"
        echo "    2. FineWeb-16k (tutorial comparison)"
        echo "============================================================"
        
        # TinyStories (primary - used in main figure)
        generate_for_dataset "tinystories" ""
        
        # FineWeb-16k (for comparison with tutorial)
        generate_for_dataset "fineweb-16k" "_fineweb16k"
    fi
    
    echo ""
    echo "============================================================"
    echo "Figure 8 data generation complete!"
    echo "============================================================"
    echo "TinyStories outputs:"
    echo "  results/language/figure_8_data_fw_medium.json (feature 3834)"
    echo "  results/language/figure_8_feature751.json (feature 751)"
    if [ -z "$FIG8_DATASET" ] || [ "$FIG8_DATASET" = "fineweb-16k" ]; then
        echo ""
        echo "FineWeb-16k outputs (tutorial comparison):"
        echo "  results/language/figure_8_data_fw_medium_fineweb16k.json (feature 3834)"
        echo "  results/language/figure_8_feature751_fineweb16k.json (feature 751)"
    fi
    echo "============================================================"
    
    # Generate PDF figures for all available datasets
    echo ""
    echo ">>> Generating Figure 8 PDFs..."
    run_figure8_figures
    
    echo ""
    echo "============================================================"
    echo "Figure 8 complete!"
    echo "============================================================"
    echo "PDF outputs:"
    echo "  Report/figures/language/figure_8_final.pdf (TinyStories)"
    if [ -z "$FIG8_DATASET" ] || [ "$FIG8_DATASET" = "fineweb-16k" ]; then
        echo "  Report/figures/language/figure_8_final_fineweb16k.pdf (FineWeb-16k)"
    fi
    echo "============================================================"
}

run_figure8() {
    # Check for subcommand
    local subcmd="${REMAINING_ARGS[0]:-}"
    
    case $subcmd in
        search)
            run_figure8_search
            ;;
        analyze)
            run_figure8_analyze
            ;;
        figures)
            run_figure8_figures
            ;;
        all)
            run_figure8_all
            ;;
        generate|"")
            run_figure8_generate
            ;;
        *)
            echo "Unknown figure8 subcommand: $subcmd"
            echo ""
            echo "Usage: ./scripts/train/run_language.sh figure8 <subcommand>"
            echo ""
            echo "Subcommands:"
            echo "  generate  (default) Generate data for features 3834 & 751"
            echo "            Runs on both TinyStories and FineWeb-16k datasets"
            echo "  search    Run full circuit search (~8-10 hours)"
            echo "  analyze   Analyze top circuits from search results"
            echo "  figures   Generate all Figure 8 PDF variants"
            echo "  all       Full pipeline: generate -> search -> analyze -> figures"
            exit 1
            ;;
    esac
}

# --- FIGURE 10: SAE TRAINING TIME ---
run_figure10() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    # Figure 10: expansion=16 has ~16K features, but we sample for reasonable runtime
    # 1000 features is statistically representative for comparing v0-v4 SAE training effect
    # Target: ~14 hours on MPS (vs 17 days for all 16K features)
    local n_features=1000
    local batch_size=32
    local exact_chunk_size=32  # Larger chunks = faster (1000 features fits in memory)
    # 64 batches × 32 batch_size × 256 n_ctx = ~524K tokens (enough for correlation)
    local n_batches=${MAX_BATCHES:-64}
    if $QUICK_MODE; then
        n_features=100
        n_batches=10
        batch_size=24
        exact_chunk_size=16
    fi
    
    local n_tokens_approx=$((n_batches * batch_size * 256))  # 64×32×256 ≈ 524K tokens
    
    echo ">>> Running SAE Training Time Analysis (Figure 10)"
    echo "    Uses verify_correlation infrastructure (DRY, memory-efficient)"
    echo "    Model: fw-medium, Layer: 12, Expansion: 16"
    echo "    SAE versions: v0 (1x) -> v4 (16x training)"
    echo "    Features: $n_features (-1 = all)"
    echo "    Batches: $n_batches (batch_size=$batch_size)"
    echo "    Exact chunk size: $exact_chunk_size features/chunk"
    echo "    Approx tokens: ~${n_tokens_approx} (~$(echo "scale=2; $n_tokens_approx / 1000000" | bc)M)"
    echo "    Device: $DEVICE"
    echo "    Metric: $METRIC"
    echo "    Dataset: fineweb"
    echo ""
    
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/sae_training_time_analysis.py \
        --device "$DEVICE" \
        --n-features "$n_features" \
        --n-batches "$n_batches" \
        --batch-size "$batch_size" \
        --exact-chunk-size "$exact_chunk_size" \
        --metric "$METRIC" \
        --dataset "fineweb" \
        --output "results/language/sae_training_time_comparison.json"
    
    echo ""
    echo "Figure 10 data generated!"
    echo "Output: results/language/sae_training_time_comparison.json"
}

# --- NEGATION DISCOVERY ---
run_negation() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    local n_samples=50000
    if $QUICK_MODE; then
        n_samples=500  # Quick mode: ~16 batches instead of 1563
    fi
    
    local config="configs/language_negation_fw.yaml"
    if [ -n "$MODEL" ]; then
        case $MODEL in
            ts-medium|tdooms/ts-medium)
                config="configs/language_negation_ts.yaml"
                ;;
            fw-small|tdooms/fw-small)
                config="configs/language_negation_fw_small.yaml"
                ;;
            fw-medium|tdooms/fw-medium)
                config="configs/language_negation_fw.yaml"
                ;;
        esac
    fi
    
    echo ">>> Running Negation Discovery"
    echo "    Config: $config"
    echo "    Samples: $n_samples"
    echo ""
    
    python src/language/negation_discovery.py \
        --config "$config" \
        --output "results/language/negation_analysis.json" \
        --device "$DEVICE" \
        --use-pretrained \
        --n-samples "$n_samples" \
        $WANDB_FLAG
    
    echo ""
    echo "Negation discovery complete!"
    echo "Output: results/language/negation_analysis.json"
}

# --- INTERACTION ANALYSIS ---
run_interaction() {
    print_header
    activate_conda
    
    mkdir -p results/language
    
    local n_features=500
    if $QUICK_MODE; then
        n_features=50
    fi
    
    echo ">>> Running Interaction Analysis"
    echo "    Features: $n_features"
    echo ""
    
    python src/language/interaction_analysis.py \
        --config "configs/language_interaction.yaml" \
        --output "results/language/interaction_analysis.json" \
        --device "$DEVICE" \
        $WANDB_FLAG
    
    echo ""
    echo "Interaction analysis complete!"
    echo "Output: results/language/interaction_analysis.json"
}

# --- GENERATE FIGURES ---
generate_figures() {
    print_header
    activate_conda
    
    echo ">>> Generating language figures..."
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python scripts/figures/generate_language_figures.py
    
    echo ""
    echo "Figure generation complete!"
}

# --- TEST ---
run_test() {
    echo "=========================================="
    echo "Quick Language MPS Test"
    echo "=========================================="
    echo ""
    
    activate_conda
    
    echo "1. Checking device availability..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
    echo ""
    
    mkdir -p results/language/test
    
    # Test 1: Negation Discovery (minimal)
    echo "2. Testing negation discovery (500 samples)..."
    cat > /tmp/test_negation_config.yaml << 'EOF'
name: test_negation_mps
model:
  pretrained: "tdooms/ts-medium"
sae:
  use_pretrained: true
  point: "mlp-out"
  layer: 2
  expansion: 4
  k: 30
analysis:
  n_samples: 500
  top_k: 10
EOF
    
    python src/language/negation_discovery.py \
        --config /tmp/test_negation_config.yaml \
        --output results/language/test/negation_test.json \
        --use-pretrained \
        --device "$DEVICE" \
        --no-wandb \
        --n-samples 500 \
        && echo "Negation discovery test PASSED" \
        || { echo "Negation discovery test FAILED"; exit 1; }
    
    echo ""
    
    # Test 2: Interaction Analysis (minimal)
    echo "3. Testing interaction analysis (20 features)..."
    cat > /tmp/test_interaction_config.yaml << 'EOF'
name: test_interaction_mps
model:
  pretrained: "tdooms/ts-medium"
sae:
  layer: 2
  input:
    name: "mlp-in"
    expansion: 4
    k: 30
  output:
    name: "mlp-out"
    expansion: 4
    k: 30
analysis:
  n_features: 20
  rank_k: 2
EOF
    
    python src/language/interaction_analysis.py \
        --config /tmp/test_interaction_config.yaml \
        --output results/language/test/interaction_test.json \
        --device "$DEVICE" \
        --no-wandb \
        && echo "Interaction analysis test PASSED" \
        || { echo "Interaction analysis test FAILED"; exit 1; }
    
    echo ""
    echo "=========================================="
    echo "ALL LANGUAGE TESTS PASSED"
    echo "=========================================="
}

# --- ALL ---
run_all() {
    print_header
    activate_conda
    
    echo ">>> Running full language pipeline..."
    echo ""
    
    # Model configurations for Figure 9: "model layer expansion dataset"
    declare -a FIGURE9_MODELS=(
        "ts-medium 4 4 tinystories"
        "fw-small 8 4 fineweb"
        "fw-medium 10 8 fineweb"
    )
    
    if $PRECOMPUTE; then
        # Step 1: Check and precompute missing eigenpairs caches
        echo "Step 1: Checking eigenpairs caches..."
        for model_config in "${FIGURE9_MODELS[@]}"; do
            read -r model layer expansion dataset <<< "$model_config"
            cache_dir="results/language/eigenpairs/$model/$layer"
            
            if [ ! -d "$cache_dir" ] || [ -z "$(ls -A "$cache_dir" 2>/dev/null)" ]; then
                echo "  Cache missing for $model layer $layer, precomputing..."
                python src/language/precompute_eigenpairs.py \
                    --model "tdooms/$model" \
                    --layer "$layer" \
                    --expansion "$expansion" \
                    --device "$DEVICE"
            else
                echo "  Cache exists for $model layer $layer, skipping precompute"
            fi
        done
        echo ""
        
        # Step 2: Run experiments with cached eigenpairs
        echo "Step 2: Running correlation analysis with cached eigenpairs..."
        LOAD_EIGENPAIRS="auto"
    else
        echo "Step 1: Skipping eigenpairs precompute (use --precompute to enable)"
        echo "Step 2: Running correlation analysis without cached eigenpairs..."
        LOAD_EIGENPAIRS=""
    fi
    
    run_figure9
    echo ""
    cleanup_memory "figure9"
    
    # Step 3: Figure 8 data generation (needed for paper-style Figure 8)
    echo "Step 3: Generating Figure 8 data..."
    run_figure8
    echo ""
    cleanup_memory "figure8"
    
    # Step 4: Other experiments
    echo "Step 4: Running negation discovery..."
    run_negation
    echo ""
    cleanup_memory "negation"
    
    echo "Step 5: Running interaction analysis..."
    run_interaction
    echo ""
    cleanup_memory "interaction"
    
    # Step 6: Generate figures
    echo "Step 6: Generating figures..."
    generate_figures
    echo ""
    cleanup_memory "figures"
    
    echo "Full pipeline complete!"
    echo ""
}

# --- HELP ---
show_help() {
    echo "Usage: ./scripts/train/run_language.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  precompute    Precompute eigenpairs for fast iteration (Phase 1)"
    echo "  figure9       Correlation sweep for Figure 9 (all 3 models)"
    echo "  figure8       Negation circuit visualization (Figure 8)"
    echo "  figure10      SAE training time analysis (Figure 10, ~1.57M tokens)"
    echo "  negation      Negation feature discovery"
    echo "  interaction   Interaction matrix analysis"
    echo "  figures       Generate all language figures from results"
    echo "  test          Quick MPS verification tests"
    echo "  all           Full language pipeline (except Figure 8)"
    echo "  help          Show this help message"
    echo ""
    echo "Precompute Subcommands:"
    echo "  precompute all              Precompute all Figure 9 models (~8-10 hours)"
    echo "  precompute figure10         Precompute Figure 10 SAE versions v0-v4"
    echo "  precompute --model X        Precompute specific model"
    echo "  precompute --model X --layer Y  Specific model and layer"
    echo ""
    echo "Figure 8 Subcommands:"
    echo "  figure8 generate  Generate data for features 3834/751 (default)"
    echo "                    Runs on both TinyStories and FineWeb-16k datasets"
    echo "  figure8 search    Run comprehensive circuit search (~8-10 hours)"
    echo "  figure8 analyze   Analyze top circuits from search results"
    echo "  figure8 figures   Generate all Figure 8 PDF variants"
    echo "  figure8 all       Full pipeline: generate -> search -> analyze -> figures"
    echo ""
    echo "Options:"
    echo "  --quick           Reduced samples/features for testing"
    echo "  --device          cpu|mps|cuda (default: auto-detect)"
    echo "  --no-wandb        Disable wandb logging"
    echo "  --model           Specific model: ts-medium, fw-small, fw-medium, all"
    echo "  --layer           Layer index (for precompute)"
    echo "  --sequential      Run models sequentially (memory-safe for figure9)"
    echo "  --feature         Feature index for figure8 generate (default: 3834)"
    echo "  --streaming       Use streaming Q computation for CUDA (Figure 8)"
    echo "  --chunk-size      Chunk size for streaming (default: 256)"
    echo "  --fig8-dataset    Dataset for Figure 8 (tinystories|fineweb|fineweb-16k)"
    echo "  --batch-size      Batch size for figure9/figure10 validation"
    echo "  --max-batches     Max batches for figure9/figure10 (default: 128 for both)"
    echo "  --metric          pearson|cosine (default: pearson)"
    echo "  --load-eigenpairs Path to cached eigenpairs or 'auto' (fast iteration)"
    echo "  --float16         Save eigenpairs in float16 (50% storage reduction)"
    echo "  --exact-accum     Exact streaming accumulation for Figure 9"
    echo "  --precompute      Precompute eigenpairs in 'all' pipeline"
    echo ""
    echo "Examples:"
    echo "  # Precompute eigenpairs (one-time, slow)"
    echo "  ./scripts/train/run_language.sh precompute all         # Figure 9 models"
    echo "  ./scripts/train/run_language.sh precompute figure10    # Figure 10 SAE versions"
    echo "  ./scripts/train/run_language.sh precompute --model ts-medium --layer 5"
    echo ""
    echo "  # Use cached eigenpairs (fast iteration)"
    echo "  ./scripts/train/run_language.sh figure9 --load-eigenpairs auto --metric pearson"
    echo "  ./scripts/train/run_language.sh figure9 --load-eigenpairs auto --metric cosine"
    echo ""
    echo "  # Standard usage (without caching)"
    echo "  ./scripts/train/run_language.sh test                      # Quick tests"
    echo "  ./scripts/train/run_language.sh figure9 --quick           # Quick correlation sweep"
    echo "  ./scripts/train/run_language.sh figure9 --model fw-medium # Single model"
    echo "  ./scripts/train/run_language.sh figure8 all               # Full Figure 8 pipeline"
    echo "  ./scripts/train/run_language.sh figure10                  # SAE training analysis (~1.57M tokens)"
    echo "  ./scripts/train/run_language.sh figure10 --max-batches 50 # Fewer batches (faster)"
    echo "  ./scripts/train/run_language.sh all                       # Full pipeline"
    echo ""
    echo "Models (with correct validation datasets):"
    echo "  ts-medium  (6L, layer 4/5, expansion 4, tinystories) - Paper's 'ts-tiny'"
    echo "  fw-small   (12L, layer 8, expansion 4, fineweb)"
    echo "  fw-medium  (16L, layer 10, expansion 8, fineweb) - 2/3 depth"
}

# --- MAIN ---
case $COMMAND in
    precompute)
        run_precompute
        ;;
    figure9|correlation)
        run_figure9
        ;;
    figure8|negation-viz)
        run_figure8
        ;;
    figure10|sae-training)
        run_figure10
        ;;
    negation)
        run_negation
        ;;
    interaction)
        run_interaction
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
    help|*)
        show_help
        ;;
esac
