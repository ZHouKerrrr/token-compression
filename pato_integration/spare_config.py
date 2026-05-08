"""SPARE Configuration for Qwen2.5-VL

This module extends Qwen2.5-VL configuration with SPARE-specific parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Union
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig


@dataclass
class CompressorConfig:
    """Configuration for Token Sort module"""
    
    # General settings
    enable: bool = False
    mode: str = 'compressor'
    prune_depth_ratio: list[float] = field(default_factory=list)
    prune_depth: list[int] = field(default_factory=list)
    
    # old version
    prune_layers: list[int] = field(default_factory=list)

@dataclass
class SPARELossConfig:
    """Configuration for SPARE loss components"""
    # loss function weights
    lambda_kd_logits:float = 0.5
    lambda_kd_feature: float = 0.5
    lambda_distortion: float = 0.01   # 0.1
    lambda_compact: float = 0.01
    lambda_tv: float = 0.01
    lambda_rate: float = 0.4          # (1e-3)
    
    # temperature for distillation losses
    temperature_kd_logits:float = 2.0
    temperature_kd_feature: float = 2.0
    
@dataclass
class SPAREConfig:
    """Complete SPARE configuration"""
    
    # Module configs
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    
    # Loss weights
    loss_config: SPARELossConfig = field(default_factory=SPARELossConfig)
    
    # Training strategy
    freeze_vision_encoder: bool = True
    freeze_llm: bool = True
    freeze_embeddings: bool = True
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = False

    # eval and skip LLM
    evaluate: bool = False

class SPAREQwen2_5_VLConfig(Qwen2_5_VLConfig):
    """Extended Qwen2.5-VL configuration with SPARE parameters"""
    
    model_type = "spare_qwen2_5_vl"
    def __init__(
        self,
        spare_config: Optional[SPAREConfig] = None,
        **kwargs
    ):
        # Initialize SPARE config
        if spare_config is None:
            self.spare_config = create_default_spare_config(**kwargs)
        else:
            self.spare_config = spare_config
        super().__init__(**kwargs)

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict['spare_config'] = asdict(self.spare_config)
        return base_dict

# Helper functions
def create_default_spare_qwen_config(**kwargs) -> SPAREQwen2_5_VLConfig:
    """Create a default SPARE-Qwen2.5-VL configuration
    
    Args:
        **kwargs: Override parameters
        
    Returns:
        SPAREQwen2_5_VLConfig instance
    """
    return SPAREQwen2_5_VLConfig(**kwargs)

def create_default_spare_config(**kwargs) -> SPAREConfig:
    """Create a default SPARE configuration.
    """
    config = SPAREConfig()

    for key, value in kwargs.items():
        if not hasattr(config, key):
            continue

        current_attr = getattr(config, key)
        if isinstance(value, dict) and current_attr is not None:
            for sub_key, sub_value in value.items():
                if hasattr(current_attr, sub_key):
                    setattr(current_attr, sub_key, sub_value)
        else:
            setattr(config, key, value)

    return config

if __name__ == "__main__":
    # Test configuration
    config = create_default_spare_config()
    print("SPARE Config created successfully!")
