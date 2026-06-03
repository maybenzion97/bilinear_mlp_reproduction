#!/bin/bash
# Master script for ALL experiments with CodeCarbon tracking
#
# This script runs all experiments sequentially to ensure accurate per-experiment
# emissions measurements. It covers:
# - Vision experiments (MNIST, Fashion-MNIST, noise/size sweeps)
# - Extension 2 (Cross-dataset robustness with CoM)
# - Extension CP (CP decomposition rank sweep)
# - Language experiments (Figure 8, 9, 10, negation, interaction)
# - Figure generation for all sections
# - Emissions validation and aggregation
#
# Usage:
#   ./scripts/train/run_all.sh          # Full production run (~6-7h on A100)
#   ./scripts/train/run_all.sh --test   # Quick validation (~15-20 min)
#   ./scripts/train/run_all.sh --no-conda  # Skip conda activation (for Snellius)
#
# The --test flag runs minimal configurations to verify emissions tracking works:
# - 1 seed, 2 epochs for vision
# - 100 features for figure8 search
# - Single model for figure9
#
# After full run, results are available at:
# - checkpoints/vision/**/*.pt (with emissions field)
# - checkpoints/extension_cross_dataset/*.pt (with emissions field)
# - checkpoints/extension_cp/*.pt (with emissions field)
# - results/language/*.json (with co2_kg field)
# - results/emissions_summary.json (aggregated)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Ensure PYTHONPATH includes project root for src module imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Parse arguments
TEST_MODE=false
SKIP_VISION=false
SKIP_EXTENSION2=false
SKIP_EXTENSION_CP=false
SKIP_LANGUAGE=false
SKIP_FIGURES=false
SKIP_FIGURE8=false
NO_CONDA=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --no-conda)
            NO_CONDA=true
            shift
            ;;
        --skip-vision)
            SKIP_VISION=true
            shift
            ;;
        --skip-extension_cross_dataset)
            SKIP_EXTENSION2=true
            shift
            ;;
        --skip-extension-cp)
            SKIP_EXTENSION_CP=true
            shift
            ;;
        --skip-language)
            SKIP_LANGUAGE=true
            shift
            ;;
        --skip-figures)
            SKIP_FIGURES=true
            shift
            ;;
        --skip-figure8)
            SKIP_FIGURE8=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test              Quick validation mode (~15-20 min)"
            echo "  --no-conda          Skip conda activation (for Snellius/HPC)"
            echo "  --skip-vision       Skip vision experiments"
            echo "  --skip-extension_cross_dataset   Skip extension 2 experiments"
            echo "  --skip-extension-cp Skip extension CP experiments"
            echo "  --skip-language     Skip language experiments"
            echo "  --skip-figures      Skip figure generation"
            echo "  --skip-figure8      Skip Figure 8 (run separately on MPS)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Full production run"
            echo "  $0 --test           # Quick validation"
            echo "  $0 --no-conda       # For Snellius (conda already loaded)"
            echo "  $0 --skip-vision    # Skip vision, run everything else"
            echo "  $0 --skip-figure8   # Skip Figure 8 (run on MPS locally)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build conda flag for sub-scripts
CONDA_FLAG=""
if $NO_CONDA; then
    CONDA_FLAG="--no-conda"
fi

# Header
echo "========================================================================"
echo "COMPLETE EXPERIMENT PIPELINE"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Test mode: $TEST_MODE"
echo "No conda: $NO_CONDA"
echo "Start time: $(date)"
echo ""

if $TEST_MODE; then
    echo ">>> TEST MODE: Running minimal configurations for validation"
    echo "    - 1 seed, 2 epochs for vision"
    echo "    - 100 features for figure8 search"
    echo "    - Single model for figure9"
    echo ""
fi

# Create log directory
mkdir -p logs

# ============================================================================
# PHASE 1: VISION EXPERIMENTS
# ============================================================================
if ! $SKIP_VISION; then
    echo ""
    echo "========================================================================"
    echo "PHASE 1: VISION EXPERIMENTS"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running vision in test mode (1 seed, 2 epochs, base configs only)"
        ./scripts/train/run_vision.sh train base --quick --no-wandb $CONDA_FLAG
    else
        echo ">>> Running all vision experiments (base, noise, size, challenge, adversarial)"
        ./scripts/train/run_vision.sh train all $CONDA_FLAG
    fi
    
    echo ">>> Vision experiments complete"
else
    echo ""
    echo ">>> Skipping vision experiments (--skip-vision)"
fi

# ============================================================================
# PHASE 2: EXTENSION 2 (Cross-Dataset Robustness)
# ============================================================================
if ! $SKIP_EXTENSION2; then
    echo ""
    echo "========================================================================"
    echo "PHASE 2: EXTENSION 2 (Cross-Dataset Robustness)"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running cross-dataset in test mode (2 epochs, MNIST only)"
        ./scripts/train/run_extension_cross_dataset.sh test
    else
        echo ">>> Running all cross-dataset experiments"
        ./scripts/train/run_extension_cross_dataset.sh train all
    fi
    
    echo ">>> Extension 2 experiments complete"
else
    echo ""
    echo ">>> Skipping extension 2 experiments (--skip-extension_cross_dataset)"
fi

# ============================================================================
# PHASE 3: EXTENSION CP (CP Decomposition)
# ============================================================================
if ! $SKIP_EXTENSION_CP; then
    echo ""
    echo "========================================================================"
    echo "PHASE 3: EXTENSION CP (CP Decomposition)"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running extension CP in test mode (2 epochs)"
        ./scripts/train/run_extension_cp.sh test
    else
        echo ">>> Running all extension CP experiments"
        ./scripts/train/run_extension_cp.sh train all
    fi
    
    echo ">>> Extension CP experiments complete"
else
    echo ""
    echo ">>> Skipping extension CP experiments (--skip-extension-cp)"
fi

# ============================================================================
# PHASE 4: LANGUAGE EXPERIMENTS
# ============================================================================
if ! $SKIP_LANGUAGE; then
    echo ""
    echo "========================================================================"
    echo "PHASE 4: LANGUAGE EXPERIMENTS"
    echo "========================================================================"
    
    if $TEST_MODE; then
        echo ">>> Running language experiments in test mode"
        
        # Figure 9 (correlation sweep) - single model, quick mode
        echo ""
        echo ">>> Figure 9: Correlation sweep (fw-medium only, quick mode)"
        ./scripts/train/run_language.sh figure9 --quick --model fw-medium $CONDA_FLAG
        
        # Figure 8 (circuit search) - 100 features only, with streaming for CUDA
        echo ""
        echo ">>> Figure 8: Circuit search (100 features, streaming enabled)"
        ./scripts/train/run_language.sh figure8 search --quick --streaming $CONDA_FLAG
        
        # Negation discovery - quick mode
        echo ""
        echo ">>> Negation discovery (quick mode)"
        ./scripts/train/run_language.sh negation --quick $CONDA_FLAG
        
    else
        echo ">>> Running all language experiments"
        
        # Figure 9 (correlation sweep) - all 3 models (sequential to avoid OOM)
        echo ""
        echo ">>> Figure 9: Correlation sweep (all 3 models, sequential)"
        ./scripts/train/run_language.sh figure9 --sequential $CONDA_FLAG
        
        # Figure 8 (full circuit search) - all 8192 features
        # --streaming enables chunked Q computation, reducing memory from 17GB to ~1GB
        if ! $SKIP_FIGURE8; then
            echo ""
            echo ">>> Figure 8: Full circuit search (8192 features, streaming enabled, ~1-2h on A100)"
            ./scripts/train/run_language.sh figure8 all --streaming $CONDA_FLAG
        else
            echo ""
            echo ">>> Skipping Figure 8 (--skip-figure8 set, run separately on MPS)"
        fi
        
        # Figure 10 (SAE training time analysis)
        echo ""
        echo ">>> Figure 10: SAE training time analysis"
        ./scripts/train/run_language.sh figure10 $CONDA_FLAG
        
        # Negation discovery
        echo ""
        echo ">>> Negation discovery"
        ./scripts/train/run_language.sh negation $CONDA_FLAG
        
        # Interaction analysis
        echo ""
        echo ">>> Interaction analysis"
        ./scripts/train/run_language.sh interaction $CONDA_FLAG
    fi
    
    echo ">>> Language experiments complete"
else
    echo ""
    echo ">>> Skipping language experiments (--skip-language)"
fi

# ============================================================================
# PHASE 5: GENERATE ALL FIGURES
# ============================================================================
# NOTE: Figure generation is NOT tied to training skip flags.
# Even if training was skipped (resume mode), figures should be generated
# from existing checkpoints.
if ! $SKIP_FIGURES; then
    echo ""
    echo "========================================================================"
    echo "PHASE 5: GENERATE ALL FIGURES"
    echo "========================================================================"
    
    echo ">>> Generating vision figures"
    ./scripts/train/run_vision.sh figures $CONDA_FLAG
    
    echo ">>> Generating language figures"
    ./scripts/train/run_language.sh figures $CONDA_FLAG
    
    echo ">>> Generating cross-dataset figures"
    ./scripts/train/run_extension_cross_dataset.sh figures
    
    echo ">>> Generating extension CP figures"
    if [ -f "./scripts/train/run_extension_cp.sh" ]; then
        ./scripts/train/run_extension_cp.sh figures 2>/dev/null || echo "    (No extension CP figures script)"
    fi
    
    echo ">>> Figure generation complete"
else
    echo ""
    echo ">>> Skipping figure generation (--skip-figures)"
fi

# ============================================================================
# PHASE 6: VALIDATE & AGGREGATE EMISSIONS
# ============================================================================
echo ""
echo "========================================================================"
echo "PHASE 6: VALIDATE & AGGREGATE EMISSIONS"
echo "========================================================================"

echo ">>> Validating emissions data in all outputs"
python scripts/figures/validate_emissions.py

echo ""
echo ">>> Aggregating emissions data"
python scripts/figures/aggregate_emissions.py

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "========================================================================"
echo "ALL COMPLETE"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Vision checkpoints: checkpoints/vision/**/*.pt"
echo "  - Extension 2 checkpoints: checkpoints/extension_cross_dataset/*.pt"
echo "  - Extension CP checkpoints: checkpoints/extension_cp/*.pt"
echo "  - Language results: results/language/*.json"
echo "  - Emissions summary: results/emissions_summary.json"
echo "  - Figures: Report/figures/"
echo ""
echo "Next steps:"
echo "  1. Review results/emissions_summary.json for report Table 1"
echo "  2. Check Report/figures/ for generated figures"
echo "  3. Update Report/sections/06_fact_discussion.tex with emissions data"
