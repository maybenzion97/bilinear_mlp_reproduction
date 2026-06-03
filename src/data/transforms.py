"""
Image transforms for bilinear MLP experiments.

This module provides reusable transforms, particularly Center-of-Mass (CoM)
normalization for standardizing input geometry across datasets.

Includes caching support to avoid recomputing CoM transforms on every run.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
from pathlib import Path
import torch.nn.functional as F
import hashlib

# Default cache directory (relative to project root)
_CACHE_DIR: Optional[Path] = None


def get_cache_dir() -> Path:
    """Get or create the cache directory for preprocessed data."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        # Find project root (look for src/ directory)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "src").exists() and (parent / "configs").exists():
                _CACHE_DIR = parent / "data" / "cache" / "com"
                break
        if _CACHE_DIR is None:
            _CACHE_DIR = Path.home() / ".cache" / "bilinear_mlp" / "com"
    
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def get_cache_path(dataset_name: str, split: str) -> Path:
    """
    Get the cache file path for a dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'mnist', 'fashion', 'emnist_letters')
        split: 'train' or 'test'
        
    Returns:
        Path to the cache file
    """
    cache_dir = get_cache_dir()
    return cache_dir / f"{dataset_name}_{split}_com.pt"


def save_to_cache(tensor: Tensor, dataset_name: str, split: str) -> None:
    """Save preprocessed tensor to cache."""
    cache_path = get_cache_path(dataset_name, split)
    # Save to CPU to avoid device mismatches when loading
    torch.save(tensor.cpu(), cache_path)
    print(f"  Cached CoM-transformed data to {cache_path}")


def load_from_cache(dataset_name: str, split: str, device: str) -> Optional[Tensor]:
    """
    Load preprocessed tensor from cache if it exists.
    
    Returns:
        Cached tensor moved to specified device, or None if cache miss
    """
    cache_path = get_cache_path(dataset_name, split)
    if cache_path.exists():
        print(f"  Loading cached CoM data from {cache_path}")
        tensor = torch.load(cache_path, map_location=device, weights_only=True)
        return tensor
    return None


def clear_com_cache() -> None:
    """Clear all cached CoM-transformed data."""
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        for f in cache_dir.glob("*.pt"):
            f.unlink()
        print(f"Cleared CoM cache at {cache_dir}")


def compute_center_of_mass(image: Tensor) -> Tuple[float, float]:
    """
    Compute the center of mass of an image.
    
    Formula: CoM_x = sigma(x · I(x,y)) / sigma(I(x,y))
    
    IMPORTANT: This should be applied to raw tensor (0-1 range) BEFORE
    normalization. Normalized data contains negative values which breaks
    the physics analogy (mass cannot be negative).
    
    Args:
        image: Tensor of shape [H, W] or [C, H, W] with values in [0, 1]
        
    Returns:
        Tuple of (com_y, com_x) - center of mass coordinates
    """
    # Handle channel dimension
    if image.dim() == 3:
        image = image.squeeze(0)  # Remove channel dim for grayscale
    
    h, w = image.shape
    
    # Ensure non-negative values for mass calculation
    image = image.clamp(min=0)
    
    # Total mass
    total_mass = image.sum()
    
    if total_mass < 1e-8:
        # If image is empty, return center
        return h / 2, w / 2
    
    # Create coordinate grids
    y_coords = torch.arange(h, device=image.device, dtype=image.dtype)
    x_coords = torch.arange(w, device=image.device, dtype=image.dtype)
    
    # Compute center of mass
    # CoM_y = sigma(y · I(y,x)) / sigma(I(y,x))
    com_y = (y_coords.view(-1, 1) * image).sum() / total_mass
    # CoM_x = sigma(x · I(y,x)) / sigma(I(y,x))
    com_x = (x_coords.view(1, -1) * image).sum() / total_mass
    
    return com_y.item(), com_x.item()


def shift_to_center(image: Tensor, target_center: Tuple[float, float] = None) -> Tensor:
    """
    Shift an image so its center of mass is at the target center.
    
    Uses affine transformation with bilinear interpolation to maintain
    smooth gradients for training.
    
    Args:
        image: Tensor of shape [C, H, W] or [H, W] with values in [0, 1]
        target_center: Target (y, x) for CoM. If None, uses image center.
        
    Returns:
        Shifted image tensor with same shape as input
    """
    # Add channel dimension if needed
    squeeze_output = False
    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze_output = True
    
    c, h, w = image.shape
    
    # Default target is image center
    if target_center is None:
        target_center = (h / 2, w / 2)
    
    # Compute current center of mass
    com_y, com_x = compute_center_of_mass(image)
    
    # Compute shift needed (in pixels)
    shift_y = target_center[0] - com_y
    shift_x = target_center[1] - com_x
    
    # Normalize shift to [-1, 1] range for grid_sample
    # Note: grid_sample samples FROM input TO output. To shift content in
    # the positive direction, we need to sample from negative positions,
    # hence the negative sign.
    shift_x_norm = -2 * shift_x / w
    shift_y_norm = -2 * shift_y / h
    
    # Create identity affine matrix and add translation
    # Affine matrix: [[1, 0, tx], [0, 1, ty]]
    theta = torch.tensor([
        [1, 0, shift_x_norm],
        [0, 1, shift_y_norm]
    ], device=image.device, dtype=image.dtype)
    
    # Add batch dimension for grid_sample
    image_batch = image.unsqueeze(0)  # [1, C, H, W]
    theta_batch = theta.unsqueeze(0)  # [1, 2, 3]
    
    # Generate sampling grid and apply
    grid = F.affine_grid(theta_batch, image_batch.shape, align_corners=False)
    shifted = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # Remove batch dimension
    result = shifted.squeeze(0)
    
    # Remove channel dimension if input didn't have one
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


class CenterOfMassTransform:
    """
    Transform that shifts images so their center of mass is at the image center.
    
    This normalizes input geometry across datasets, allowing models to learn
    true shapes rather than overfitting to alignment artifacts.
    
    IMPORTANT: Apply this transform BEFORE any normalization (like ImageNet stats).
    The CoM calculation requires non-negative values (0-1 range).
    
    Usage:
        transform = CenterOfMassTransform()
        centered_image = transform(raw_image)  # raw_image should be in [0, 1]
    """
    
    def __init__(self, target_center: Tuple[float, float] = None):
        """
        Initialize the transform.
        
        Args:
            target_center: Target (y, x) coordinates for center of mass.
                          If None, uses image center.
        """
        self.target_center = target_center
    
    def __call__(self, image: Tensor) -> Tensor:
        """
        Apply center-of-mass centering to an image.
        
        Args:
            image: Tensor of shape [C, H, W] or [H, W] with values in [0, 1]
            
        Returns:
            Centered image tensor with same shape
        """
        return shift_to_center(image, self.target_center)
    
    def __repr__(self) -> str:
        return f"CenterOfMassTransform(target_center={self.target_center})"


def apply_com_to_batch(batch: Tensor, show_progress: bool = True) -> Tensor:
    """
    Apply center-of-mass centering to a batch of images.
    
    Args:
        batch: Tensor of shape [B, C, H, W] or [B, H, W] with values in [0, 1]
        show_progress: If True, show progress for large batches
        
    Returns:
        Centered batch tensor with same shape
    """
    transform = CenterOfMassTransform()
    n_samples = batch.shape[0]
    
    # Handle batched input
    if batch.dim() == 4:
        # [B, C, H, W]
        results = []
        for i, img in enumerate(batch):
            results.append(transform(img))
            if show_progress and (i + 1) % 10000 == 0:
                print(f"    Processed {i+1}/{n_samples} images...")
        return torch.stack(results)
    elif batch.dim() == 3:
        # [B, H, W] - add and remove channel dim
        batch_with_channel = batch.unsqueeze(1)  # [B, 1, H, W]
        results = []
        for i, img in enumerate(batch_with_channel):
            results.append(transform(img))
            if show_progress and (i + 1) % 10000 == 0:
                print(f"    Processed {i+1}/{n_samples} images...")
        result = torch.stack(results)
        return result.squeeze(1)  # [B, H, W]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {batch.dim()}D")


def apply_com_to_batch_cached(
    batch: Tensor,
    dataset_name: str,
    split: str,
    device: str = "cpu",
    use_cache: bool = True,
) -> Tensor:
    """
    Apply center-of-mass centering with caching support.
    
    This is the preferred function for dataset classes to use, as it handles
    caching automatically to avoid recomputing CoM transforms on every run.
    
    Args:
        batch: Tensor of shape [B, C, H, W] or [B, H, W] with values in [0, 1]
        dataset_name: Name of dataset (e.g., 'mnist', 'fashion', 'emnist_letters')
        split: 'train' or 'test'
        device: Device to load cached data onto
        use_cache: If True, use caching (default). Set to False to force recompute.
        
    Returns:
        Centered batch tensor with same shape
    """
    # Try loading from cache
    if use_cache:
        cached = load_from_cache(dataset_name, split, device)
        if cached is not None:
            # Verify shape matches
            if cached.shape == batch.shape:
                return cached
            else:
                print(f"  Cache shape mismatch ({cached.shape} vs {batch.shape}), recomputing...")
    
    # Compute CoM transform
    print(f"  Computing CoM transform for {batch.shape[0]} images...")
    result = apply_com_to_batch(batch, show_progress=True)
    
    # Save to cache
    if use_cache:
        save_to_cache(result, dataset_name, split)
    
    return result
