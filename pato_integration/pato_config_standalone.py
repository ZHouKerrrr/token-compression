"""Standalone PATO Configuration

This is a simplified configuration that can work independently
for testing and development.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union


@dataclass
class GRawConfig:
    """Configuration for g_raw module (pixel-level precompression)"""
    
    # General settings
    enable: bool = True
    mode: str = 'A'  # Method A: Weighted Downsampling
    
    # Input/Output sizes
    target_size: Tuple[int, int] = (448, 448)
    
    # Feature dimensions
    text_dim: int = 3584  # Qwen2.5-VL language model hidden size
    vision_dim: int = 256  # LightCNN feature dimension
    
    # Network architecture
    density_hidden_dim: int = 256
    density_layers: int = 2
    
    # Regularization
    lambda_tv: float = 1e-4  # Total Variation
    lambda_area: float = 1e-3  # Area constraint
    min_area_ratio: float = 0.1  # Minimum density area ratio
    
    # Training
    learnable: bool = True


@dataclass
class TokenSortConfig:
    """Configuration for Token Sort module"""
    
    # General settings
    enable: bool = True
    mode: str = 'A'  # Method A: Differentiable Sorting
    
    # Budget settings
    budgets: Union[int, List[int]] = field(default_factory=lambda: [256])
    budget_min: int = 128
    budget_max: int = 512
    random_budget_training: bool = True  # Random budget during training
    
    # Ranker network
    scorer_hidden_dim: int = 256
    
    # Temperature (for differentiable sorting)
    tau_init: float = 1.0
    tau_final: float = 0.1
    tau_decay: str = 'linear'  # 'linear' or 'exponential'
    
    # Regularization
    lambda_entropy: float = 1e-3
    lambda_diversity: float = 1e-4
    
    # Sinkhorn iterations (for soft permutation)
    sinkhorn_iters: int = 5


@dataclass
class ProjectorConfig:
    """Configuration for Visual Projector"""
    
    # Projector type
    mode: str = 'A'  # A: Simplified Linear, B: Grid Reconstruction
    
    # Feature dimensions
    vision_dim: int = 1152  # Vision encoder output dim
    hidden_dim: int = 3584  # LLM hidden dim
    
    # Dropout
    dropout: float = 0.1


@dataclass
class PATOConfig:
    """Complete PATO configuration"""
    
    # Module configs
    g_raw: GRawConfig = field(default_factory=GRawConfig)
    token_sort: TokenSortConfig = field(default_factory=TokenSortConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    
    # Loss weights
    lambda_distill: float = 0.05  # Feature distillation
    lambda_contrast: float = 0.1  # Contrastive loss (query pairs)
    lambda_sort_reg: float = 0.01  # Token sort regularization
    
    # Training strategy
    freeze_vision_encoder: bool = True
    freeze_llm: bool = True
    freeze_embeddings: bool = True
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = False


# Helper function
def create_default_pato_config(**kwargs) -> PATOConfig:
    """Create a default PATO configuration
    
    Args:
        **kwargs: Override parameters
        
    Returns:
        PATOConfig instance
    """
    config = PATOConfig()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = create_default_pato_config()
    print("PATO Config created successfully!")
    print(f"g_raw enabled: {config.g_raw.enable}")
    print(f"token_sort enabled: {config.token_sort.enable}")
    print(f"Target size: {config.g_raw.target_size}")
    print(f"Token budgets: {config.token_sort.budgets}")
