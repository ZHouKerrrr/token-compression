"""
G_Raw Methods for Conditional Precompression

This package contains implementations of different conditional precompression methods:
- A: Weighted Downsampling (stable baseline)
- I: Tiled Select-and-Place (strong coverage, geometric consistency)
- C: Content-Adaptive Kernel (high detail fidelity)
- F: Gaussian Splatting (extremely low budget friendly)
- B: Mixture-of-Scales (good stability)

Each method implements the BaseGRaw interface and focuses on query-conditional
pixel domain precompression to generate information-dense smaller images.
"""

# Import base classes first
from .base import BaseGRaw, register_graw, get_graw_class, RegularizationUtils

# Import all g_raw methods to trigger registration
from . import weighted_downsample

# Import the specific classes to make them available
from .weighted_downsample import WeightedDownsample

__all__ = [
    "BaseGRaw",
    "register_graw",
    "get_graw_class",
    "RegularizationUtils",
    "weighted_downsample",
    "WeightedDownsample",
]