"""PATO Configuration for Qwen2.5-VL

This module extends Qwen2.5-VL configuration with PATO-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig


@dataclass
class GRawConfig:
    """Configuration for g_raw module (pixel-level precompression)"""
    
    # General settings
    enable: bool = True
    mode: str = 'A'  # Method A: Weighted Downsampling
    
    # Input/Output sizes
    target_size: Tuple[int, int] = (448, 448)
    
    # Feature dimensions
    text_hidden_size: int = 2048  # Qwen2.5-VL language model hidden size
    vision_hidden_size: int = 1280  # LightCNN feature dimension
    
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
    mode: str = 'dynamic_token_sorter'  
    # Method hard_token_sorter: Hard token pruned by scores
    token_threshold: float = 0.5
    score_threshold: float = 0

    tau_init: float = 1.0 # tau of softmax in rate_loss
    tau_min: float = 0.05
    tau_decay: str = 'linear'  # 'linear' or 'exponential'

    # Ranker network
    scorer_hidden_dim: int = 256
    
    # Regularization
    lambda_entropy: float = 1e-3
    lambda_diversity: float = 1e-4
    
    # Sinkhorn iterations (for soft permutation)
    sinkhorn_iters: int = 5


@dataclass
class ProjectorConfig:
    """Configuration for Visual Projector"""
    # General settings
    enable: bool = True
    # Projector type
    mode: str = 'A'  # A: Simplified Linear, B: Grid Reconstruction
    
    # Feature dimensions
    vision_hidden_size: int = 1280  # Vision encoder output dim
    hidden_size: int = 2048  # LLM hidden dim

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
    lambda_rate: float = 0.3 # ratio of Rate Loss of token sort
    lambda_distortion: float = 2.0 # Distortion of main mission
    
    # Training strategy
    freeze_vision_encoder: bool = True
    freeze_llm: bool = True
    freeze_embeddings: bool = True
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = False

    # eval and skip LLM
    evaluate: bool = False

class PATOQwen2_5_VLConfig(Qwen2_5_VLConfig):
    """Extended Qwen2.5-VL configuration with PATO parameters"""
    
    model_type = "pato_qwen2_5_vl"
    
    def __init__(
        self,
        pato_config: Optional[PATOConfig] = None,
        **kwargs
    ):
        # Initialize PATO config
        if pato_config is None:
            self.pato_config = create_default_pato_config(**kwargs)
        elif isinstance(pato_config, dict):
            # Convert dict to dataclass
            self.pato_config = self._dict_to_pato_config(pato_config)
        else:
            self.pato_config = pato_config
        super().__init__(**kwargs)
    @staticmethod
    def _dict_to_pato_config(config_dict: dict) -> PATOConfig:
        """Convert dictionary to PATOConfig dataclass"""
        
        # Extract sub-configs
        g_raw_dict = config_dict.get('g_raw', {})
        token_sort_dict = config_dict.get('token_sort', {})
        projector_dict = config_dict.get('projector', {})
        
        # Create dataclass instances
        g_raw_config = GRawConfig(**g_raw_dict) if g_raw_dict else GRawConfig()
        token_sort_config = TokenSortConfig(**token_sort_dict) if token_sort_dict else TokenSortConfig()
        projector_config = ProjectorConfig(**projector_dict) if projector_dict else ProjectorConfig()
        
        # Main config parameters
        main_params = {
            k: v for k, v in config_dict.items() 
            if k not in ['g_raw', 'token_sort', 'projector']
        }
        
        return PATOConfig(
            g_raw=g_raw_config,
            token_sort=token_sort_config,
            projector=projector_config,
            **main_params
        )
    def _base_config(self):
        """Get base Qwen2.5-VL config without PATO params"""
        base_dict = self.to_dict()
        # Remove PATO-specific entries
        base_dict.pop('pato_config', None)
        base_config = Qwen2_5_VLConfig(**base_dict)
        return base_config
    def to_dict(self):
        """Override to include PATO config"""
        output = super().to_dict()
        
        # Add PATO config
        output['pato_config'] = {
            'g_raw': {
                k: v for k, v in self.pato_config.g_raw.__dict__.items()
            },
            'token_sort': {
                k: v for k, v in self.pato_config.token_sort.__dict__.items()
            },
            'projector': {
                k: v for k, v in self.pato_config.projector.__dict__.items()
            },
            'lambda_distill': self.pato_config.lambda_distill,
            'lambda_contrast': self.pato_config.lambda_contrast,
            'lambda_sort_reg': self.pato_config.lambda_sort_reg,
            'freeze_vision_encoder': self.pato_config.freeze_vision_encoder,
            'freeze_llm': self.pato_config.freeze_llm,
            'freeze_embeddings': self.pato_config.freeze_embeddings,
            'use_gradient_checkpointing': self.pato_config.use_gradient_checkpointing,
        }
        
        return output


# Helper functions
def create_default_pato_qwen_config(**kwargs) -> PATOQwen2_5_VLConfig:
    """Create a default PATO-Qwen2.5-VL configuration
    
    Args:
        **kwargs: Override parameters
        
    Returns:
        PATOQwen2_5_VLConfig instance
    """
    return PATOQwen2_5_VLConfig(**kwargs)

# Helper function
def create_default_pato_config(**kwargs) -> PATOConfig:
    """Create a default PATO configuration
    
    Args:
        **kwargs: Override parameters
        
    Returns:
        PATOConfig instance
    """
    config = PATOConfig()
    # Special parameters
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

if __name__ == "__main__":
    # Test configuration
    config = create_default_pato_config()
    print("PATO Config created successfully!")
    print(f"g_raw enabled: {config.pato_config.g_raw.enable}")
    print(f"token_sort enabled: {config.pato_config.token_sort.enable}")
    print(f"Target size: {config.pato_config.g_raw.target_size}")
    print(f"Token budgets: {config.pato_config.token_sort.budgets}")
