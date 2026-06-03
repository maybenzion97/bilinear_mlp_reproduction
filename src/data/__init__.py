"""
Data utilities and dataset wrappers for Bilinear MLP experiments.

This module provides:
- MNIST/Fashion-MNIST wrappers with optional CoM normalization
- EMNIST Letters/Digits for cross-dataset experiments
- USPS with automatic upscaling (16x16 -> 28x28)
- Center-of-Mass transform for input geometry normalization
- Challenge dataset for Figure 6 experiments
"""

# Transforms
from .transforms import (
    CenterOfMassTransform,
    compute_center_of_mass,
    shift_to_center,
    apply_com_to_batch,
    apply_com_to_batch_cached,
    get_cache_dir,
    clear_com_cache,
)

# MNIST/Fashion-MNIST
from .mnist import (
    MNIST,
    FashionMNIST,
)

# EMNIST
from .emnist import (
    EMNISTLetters,
    EMNISTDigits,
    LETTER_DIGIT_SIMILARITY,
    EMNIST_LETTERS_CLASS_NAMES,
    EMNIST_DIGITS_CLASS_NAMES,
    EMNIST_CLASS_NAMES,
    extract_emnist_letters,
    get_emnist_letter_indices,
    load_emnist_letters_normalized,
    load_emnist_digits_normalized,
)

# USPS
from .usps import (
    USPS,
    load_usps_normalized,
)

# Challenge dataset (Figure 6)
from .challenge_dataset import (
    ChallengeDataset,
    ChallengeDatasetWrapper,
    create_challenge_datasets,
    get_target_image,
    cosine_similarity,
)

__all__ = [
    # Transforms
    "CenterOfMassTransform",
    "compute_center_of_mass",
    "shift_to_center",
    "apply_com_to_batch",
    "apply_com_to_batch_cached",
    "get_cache_dir",
    "clear_com_cache",
    # MNIST
    "MNIST",
    "FashionMNIST",
    # EMNIST
    "EMNISTLetters",
    "EMNISTDigits",
    "LETTER_DIGIT_SIMILARITY",
    "EMNIST_LETTERS_CLASS_NAMES",
    "EMNIST_DIGITS_CLASS_NAMES",
    "EMNIST_CLASS_NAMES",
    "extract_emnist_letters",
    "get_emnist_letter_indices",
    "load_emnist_letters_normalized",
    "load_emnist_digits_normalized",
    # USPS
    "USPS",
    "load_usps_normalized",
    # Challenge
    "ChallengeDataset",
    "ChallengeDatasetWrapper",
    "create_challenge_datasets",
    "get_target_image",
    "cosine_similarity",
]
