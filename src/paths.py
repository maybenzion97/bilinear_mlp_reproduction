"""
Centralized path constants for the Bilinear MLP project.

This module provides a single source of truth for all checkpoint and figure paths,
reducing duplication and making path changes easier to manage.

Usage:
    from src.paths import (
        MNIST_CHECKPOINTS,
        VISION_FIGURES,
        LANGUAGE_FIGURES,
    )
"""

from pathlib import Path

# Project root (bilinear_mlp_reproduction/)
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Checkpoints
# =============================================================================

CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"

# Vision checkpoints
VISION_CHECKPOINTS = CHECKPOINTS_ROOT / "vision"
MNIST_CHECKPOINTS = VISION_CHECKPOINTS / "mnist"
FASHION_CHECKPOINTS = VISION_CHECKPOINTS / "fashion"
NOISE_SWEEP_CHECKPOINTS = VISION_CHECKPOINTS / "noise_sweep"
SIZE_SWEEP_CHECKPOINTS = VISION_CHECKPOINTS / "size_sweep"
CHALLENGE_CHECKPOINTS = VISION_CHECKPOINTS / "challenge"

# Extension checkpoints
EXTENSION_CROSS_DATASET_CHECKPOINTS = CHECKPOINTS_ROOT / "extension_cross_dataset"
EXTENSION_CP_CHECKPOINTS = CHECKPOINTS_ROOT / "extension_cp"

# Backward compatibility alias (DEPRECATED - use EXTENSION_CROSS_DATASET_CHECKPOINTS)
EXTENSION2_CHECKPOINTS = EXTENSION_CROSS_DATASET_CHECKPOINTS

# =============================================================================
# Figures
# =============================================================================

FIGURES_ROOT = PROJECT_ROOT / "Report" / "figures"

# Figure subdirectories by paper section
VISION_FIGURES = FIGURES_ROOT / "vision"
LANGUAGE_FIGURES = FIGURES_ROOT / "language"
EXTENSION_CROSS_DATASET_FIGURES = FIGURES_ROOT / "extension_cross_dataset"
EXTENSION_CP_FIGURES = FIGURES_ROOT / "extension_cp"

# Backward compatibility alias (DEPRECATED - use EXTENSION_CROSS_DATASET_FIGURES)
EXTENSION2_FIGURES = EXTENSION_CROSS_DATASET_FIGURES

# =============================================================================
# Results (intermediate data, JSON outputs, etc.)
# =============================================================================

RESULTS_ROOT = PROJECT_ROOT / "results"

# Language results (JSON data files, NOT figures)
LANGUAGE_RESULTS = RESULTS_ROOT / "language"

# Language eigenpairs cache (precomputed eigendecompositions)
LANGUAGE_EIGENPAIRS = RESULTS_ROOT / "language" / "eigenpairs"

# Extension Cross-Dataset results (JSON data files)
EXTENSION_CROSS_DATASET_RESULTS = RESULTS_ROOT / "extension_cross_dataset"

# Backward compatibility alias (DEPRECATED - use EXTENSION_CROSS_DATASET_RESULTS)
EXTENSION2_RESULTS = EXTENSION_CROSS_DATASET_RESULTS

# =============================================================================
# Paper Hub
# =============================================================================

PAPER_HUB_ROOT = PROJECT_ROOT / "Report"
PAPER_HUB_BUNDLE = PAPER_HUB_ROOT / "paper_hub_bundle"
PAPER_HUB_ZIP = PAPER_HUB_ROOT / "paper_hub_bundle.zip"

# =============================================================================
# Original paper code (DO NOT MODIFY)
# =============================================================================

ORIGINAL_CODE_PATH = PROJECT_ROOT / "bilinear-decomposition-main"

# =============================================================================
# Google Drive Data (checkpoints and results)
# =============================================================================

# Shared folder containing checkpoints.zip and results.zip
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1et6EfHxvyEZKCZ1yXfA1EOHbcShCNWzt"
GDRIVE_FOLDER_ID = "1et6EfHxvyEZKCZ1yXfA1EOHbcShCNWzt"

# Individual file IDs (for direct download)
GDRIVE_CHECKPOINTS_ZIP_ID = "12RI9zXhgjXcrVO50qU3GGRvHNTyKw3ws"
GDRIVE_RESULTS_ZIP_ID = "1iy3BfsCVOOIt43yi9h9KNe31wwGPVFgK"

# Local zip file paths
CHECKPOINTS_ZIP = PROJECT_ROOT / "checkpoints.zip"
RESULTS_ZIP = PROJECT_ROOT / "results.zip"

# =============================================================================
# Helper functions
# =============================================================================


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_path(
    dataset: str,
    config: str,
    seed: int,
    extension: str = None,
) -> Path:
    """
    Get the path for a checkpoint file.
    
    Args:
        dataset: "mnist", "fashion", "emnist_digits", "emnist_letters"
        config: "none", "noise", "wd", "full", "regularized", etc.
        seed: Random seed (42, 43, 44, 45, 46)
        extension: Optional extension type ("cp", "cross_dataset"/"extension2", None for vision)
    
    Returns:
        Path to the checkpoint file
    
    Example:
        >>> get_checkpoint_path("mnist", "full", 42)
        Path('.../checkpoints/vision/mnist/mnist_dense_full_seed42.pt')
    """
    if extension == "cp":
        base = EXTENSION_CP_CHECKPOINTS
        filename = f"{dataset}_{config}_seed{seed}.pt"
    elif extension in ("cross_dataset", "extension2"):
        base = EXTENSION_CROSS_DATASET_CHECKPOINTS
        filename = f"{dataset}_{config}_seed{seed}.pt"
    elif dataset == "fashion":
        base = FASHION_CHECKPOINTS
        filename = f"fashion_dense_{config}_seed{seed}.pt"
    elif dataset in ("emnist_digits", "emnist_letters"):
        base = EXTENSION_CROSS_DATASET_CHECKPOINTS
        filename = f"{dataset}_{config}_seed{seed}.pt"
    else:
        base = MNIST_CHECKPOINTS
        filename = f"mnist_dense_{config}_seed{seed}.pt"
    
    return base / filename


def get_figure_path(name: str, section: str) -> Path:
    """
    Get the path for a figure file.
    
    Args:
        name: Figure filename (e.g., "figure_5a_similarity.pdf")
        section: "vision", "language", "cross_dataset"/"extension2", "extension_cp"
    
    Returns:
        Path to the figure file
    """
    section_map = {
        "vision": VISION_FIGURES,
        "language": LANGUAGE_FIGURES,
        "cross_dataset": EXTENSION_CROSS_DATASET_FIGURES,
        "extension2": EXTENSION_CROSS_DATASET_FIGURES,  # backward compatibility
        "extension_cp": EXTENSION_CP_FIGURES,
    }
    return section_map[section] / name
