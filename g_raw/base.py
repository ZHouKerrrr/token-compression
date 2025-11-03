"""Base class for g_raw methods

This module provides a common interface for all g_raw implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


# Registry for g_raw methods
_GRAW_REGISTRY = {}


def register_graw(name: str):
    """Decorator to register g_raw methods"""
    def decorator(cls):
        _GRAW_REGISTRY[name] = cls
        return cls
    return decorator


def get_graw_class(name: str):
    """Get g_raw class by name"""
    if name not in _GRAW_REGISTRY:
        raise ValueError(f"Unknown g_raw method: {name}. Available: {list(_GRAW_REGISTRY.keys())}")
    return _GRAW_REGISTRY[name]


class BaseGRaw(nn.Module, ABC):
    """Base class for g_raw pixel-level precompression
    
    All g_raw methods should inherit from this class and implement:
    - _setup_module(): Initialize method-specific components
    - forward(): Perform conditional precompression
    - compute_regularization_loss(): Compute method-specific losses
    """
    
    def __init__(self, config, context: Dict[str, Any]):
        """Initialize base g_raw module
        
        Args:
            config: Configuration object/dict with method-specific parameters
            context: Runtime context (device, dtype, etc.)
        """
        super().__init__()
        
        self.config = config
        self.context = context
        
        # Setup method-specific components
        self._setup_module()
    
    @abstractmethod
    def _setup_module(self) -> None:
        """Initialize method-specific components and parameters
        
        This method should be implemented by subclasses to create
        their specific networks and parameters.
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Forward pass for conditional precompression
        
        Args:
            images: Input images [B, C, H, W]
            text_embeddings: Query text embeddings [B, D] or [B, L, D]
            target_size: Target output size (H, W). If None, use config default
            
        Returns:
            Compressed images [B, C, H_out, W_out]
        """
        pass
    
    @abstractmethod
    def compute_regularization_loss(
        self,
        images: torch.Tensor,
        compressed_images: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute method-specific regularization losses
        
        Args:
            images: Original input images [B, C, H, W]
            compressed_images: Compressed output images [B, C, H_out, W_out]
            text_embeddings: Query embeddings [B, D] or [B, L, D]
            
        Returns:
            Dictionary of loss terms
        """
        pass


# Utility functions (can be used by subclasses)
class RegularizationUtils:
    """Utility functions for regularization losses"""
    
    @staticmethod
    def compute_smoothness_regularization(
        tensor: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute Total Variation (TV) loss for smoothness
        
        Args:
            tensor: Input tensor [B, C, H, W]
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            TV loss
        """
        # Compute differences
        diff_h = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        diff_w = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        
        # Total variation
        tv_loss = diff_h.mean() + diff_w.mean()
        
        if reduction == 'mean':
            return tv_loss
        elif reduction == 'sum':
            return tv_loss * tensor.shape[0]
        else:
            return tv_loss
