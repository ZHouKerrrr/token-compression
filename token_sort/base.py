"""Base class for Token Sort methods

This module provides a common interface for all token sorting implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


# Registry for token sort methods
_TOKEN_SORT_REGISTRY = {}


def register_token_sort(name: str):
    """Decorator to register token sort methods"""
    def decorator(cls):
        _TOKEN_SORT_REGISTRY[name] = cls
        return cls
    return decorator


def get_token_sort_class(name: str):
    """Get token sort class by name"""
    if name not in _TOKEN_SORT_REGISTRY:
        raise ValueError(f"Unknown token sort method: {name}. Available: {list(_TOKEN_SORT_REGISTRY.keys())}")
    return _TOKEN_SORT_REGISTRY[name]


class BaseTokenSorter(nn.Module, ABC):
    """Base class for token sorting/selection
    
    All token sort methods should inherit from this class and implement:
    - _setup_module(): Initialize method-specific components
    - forward(): Perform token sorting and selection
    - compute_budget_loss(): Compute budget-related losses
    """
    
    def __init__(self, config, context: Dict[str, Any]):
        """Initialize base token sorter
        
        Args:
            config: Configuration object/dict with method-specific parameters
            context: Runtime context (hidden_size, device, etc.)
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Forward pass for token sorting and selection
        
        Args:
            hidden_states: Token embeddings [B, N, D]
            attention_mask: Token validity mask [B, N]
            budget: Target number of tokens to select
            query_embeddings: Query embeddings [B, D]
            **kwargs: Additional method-specific arguments
            
        Returns:
            selected_tokens: Selected tokens [B, M, D]
            sort_indices: Sort indices [B, N]
            aux_outputs: Auxiliary outputs dictionary
        """
        pass

    
    def update_temperature(self, current_step: int, total_steps: int) -> None:
        """Update temperature parameter (for annealing)
        
        This is optional and can be overridden by subclasses that use
        temperature-based methods.
        
        Args:
            current_step: Current training step
            total_steps: Total training steps
        """
        pass
