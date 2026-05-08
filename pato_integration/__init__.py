"""PATO Integration Package

Integration of PATO methods into Qwen2.5-VL model.
"""

from .pato_config import (
    PATOConfig,
    PATOQwen2_5_VLConfig,
    GRawConfig,
    TokenSortConfig,
    ProjectorConfig,
    create_default_pato_config,
)
from .pato_model import (
    PATOQwen2_5_VLForConditionalGeneration_ROUTE,
)
from .pato import (
    PATOQwen2_5_VLForConditionalGeneration,
    PATOQwen2_5_VisionTransformer,
)
# from .clip_qwen import (
#     CLIPQwen2_5_VLForConditionalGeneration
# )
from .spare_config import (
    SPAREConfig,
    SPAREQwen2_5_VLConfig,
    create_default_spare_config,
)
from .spare import (
    SPAREQwen2_5_VLForConditionalGeneration,
)
from .spare_loss import create_spare_loss
from .pato_loss import create_pato_loss
__all__ = [
    'PATOQwen2_5_VLForConditionalGeneration_ROUTE',
    'PATOQwen2_5_VLForConditionalGeneration',
    'PATOQwen2_5_VisionTransformer',
    'PATOConfig',
    'PATOQwen2_5_VLConfig',
    'GRawConfig',
    'TokenSortConfig',
    'ProjectorConfig',
    'create_default_pato_config',
    'create_pato_loss',
    
    'SPAREConfig',
    'SPAREQwen2_5_VLConfig',
    'create_default_spare_config',
    'create_spare_loss',
]
