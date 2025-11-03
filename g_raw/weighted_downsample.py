"""
G_Raw Method A: Weighted Downsampling

Implementation of weighted downsampling method for query-conditional precompression.
This is the most stable baseline method that learns significance density maps
for anti-aliased downsampling.

Architecture:
    Input Image -> LightCNN -> Text Fusion (FiLM) -> Density Map -> Weighted Downsampling
                                              |
                                      Query Text -> Text Projection ->
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base classes and utilities
try:
    from ..base import BaseGRaw, register_graw
    from ..utils import VisionUtils, MathUtils, RegularizationUtils
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from base import BaseGRaw, register_graw
    from utils import VisionUtils, MathUtils, RegularizationUtils


class LightCNN(nn.Module):
    """Lightweight CNN for feature extraction
    
    4-layer convolutional network with:
    - Progressive feature dimension increase (64 -> 256)
    - Spatial resolution preservation
    - Batch normalization and ReLU activation
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 feature_dim: int = 256,
                 hidden_dims: Optional[Tuple[int, ...]] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = (64, 128, 256, 256)
        
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        
        # Build convolutional layers
        layers = []
        channels = in_channels
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Conv + BatchNorm + ReLU
            layers.extend([
                nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            channels = hidden_dim
        
        # Final projection to feature_dim
        layers.append(
            nn.Conv2d(channels, feature_dim, kernel_size=3, padding=1, bias=True)
        )
        
        self.backbone = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Features [B, feature_dim, H, W]
        """
        return self.backbone(x)


class TextProjection(nn.Module):
    """Text projection module for query conditioning
    
    Projects text embeddings to the same dimension as visual features.
    Compatible with existing implementations in GlimpsePrune and VisionZip.
    """
    
    def __init__(self,
                 text_input_dim: int = 1536,  # Typical text embedding dimension
                 feature_dim: int = 256,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.text_input_dim = text_input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or (feature_dim * 2)
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(text_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, feature_dim * 2),  # For FiLM (gamma + beta)
            nn.LayerNorm(feature_dim * 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            text_embeddings: Text embeddings [B, D_text] or [B, L, D_text]
            
        Returns:
            FiLM parameters [B, 2*feature_dim]
        """
        # Handle multi-dimensional text embeddings (average pooling)
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.mean(dim=1)  # [B, D_text]
        
        return self.projection(text_embeddings)


class DensityPredictor(nn.Module):
    """Density map predictor using 1x1 convolution
    
    Takes FiLM-modulated features and produces significance density map.
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or 256
        
        # Density prediction network
        self.density_net = nn.Sequential(
            nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.hidden_dim // 2, 1, kernel_size=1, bias=True)  # Output density
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            features: FiLM-modulated features [B, feature_dim, H, W]
            
        Returns:
            Density map [B, 1, H, W] (values in [0, 1] after sigmoid)
        """
        density_logits = self.density_net(features)
        return torch.sigmoid(density_logits)


@register_graw("A")
class WeightedDownsample(BaseGRaw):
    """Method A: Weighted Downsampling
    
    Stable baseline method that learns significance density maps for
    query-conditional anti-aliased downsampling.
    
    Pipeline:
        1. Extract visual features with LightCNN
        2. Project text embeddings to feature space  
        3. Apply FiLM modulation for text conditioning
        4. Predict significance density map
        5. Perform normalized weighted downsampling
        6. Apply regularization (TV + area constraint)
    """
    
    def __init__(self, config, context: Dict[str, Any]):
        # Set configuration attributes BEFORE super().__init__() to avoid AttributeError
        # when BaseGRaw.__init__ calls self._setup_module()
        self.target_size = tuple(config.target_size)
        self.text_dim = config.text_dim
        self.vision_dim = config.vision_dim
        self.density_hidden_dim = config.density_hidden_dim
        self.density_layers = config.density_layers
        self.lambda_tv = config.lambda_tv
        self.lambda_area = config.lambda_area
        self.min_area_ratio = config.min_area_ratio
        
        super().__init__(config, context)
        
        # Initialize components (now called automatically by super().__init__)
        # self._setup_module() # Not needed - already called by BaseGRaw.__init__
    
    def _setup_module(self) -> None:
        """Initialize method-specific components and parameters"""
        
        # Visual feature extractor
        self.light_cnn = LightCNN(
            in_channels=3,
            feature_dim=self.vision_dim,
            hidden_dims=(64, 128, 256, self.vision_dim)
        )
        
        # Text projection for FiLM
        self.text_proj = TextProjection(
            text_input_dim=self.text_dim,
            feature_dim=self.vision_dim
        )
        
        # Density predictor
        self.density_predictor = DensityPredictor(
            feature_dim=self.vision_dim,
            hidden_dim=self.density_hidden_dim
        )
        
        # Device
        self.device = torch.device(self.context.get('device', 'cpu'))
        self.to(self.device)
    
    def forward(self, 
                images: torch.Tensor, 
                text_embeddings: torch.Tensor,
                target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass for conditional precompression
        
        Args:
            images: Input images [B, C, H, W]
            text_embeddings: Query text embeddings [B, D] or [B, L, D]
            target_size: Target output size (H, W). If None, use config default
            
        Returns:
            Compressed images [B, C, H_out, W_out]
        """
        if target_size is None:
            target_size = self.target_size
        
        # Step 1: Extract visual features using LightCNN
        visual_features = self.light_cnn(images)  # [B, vision_dim, H, W]
        
        # Step 2: Project text embeddings for FiLM modulation
        film_params = self.text_proj(text_embeddings)  # [B, 2*vision_dim]
        
        # Step 3: Apply FiLM modulation to condition visual features on text
        filmed_features = self._apply_film(visual_features, film_params)
        
        # Step 4: Predict significance density map (higher = more important regions)
        density_map = self.density_predictor(filmed_features)  # [B, 1, H, W]
        
        # Step 5: Apply minimum area constraint with gradient support
        density_map = self._apply_area_constraint(density_map)
        
        # Step 6: Perform density-aware weighted downsampling (FIXED: density now affects output)
        compressed_images = self._weighted_downsample(
            images, density_map, target_size
        )
        
        return compressed_images
    
    def _apply_film(self, 
                   features: torch.Tensor, 
                   film_params: torch.Tensor) -> torch.Tensor:
        """Apply Feature-wise Linear Modulation
        
        Args:
            features: Visual features [B, D, H, W]
            film_params: FiLM parameters [B, 2*D]
            
        Returns:
            FiLM-modulated features [B, D, H, W]
        """
        batch_size, feature_dim, height, width = features.shape
        
        # Split into gamma and beta
        gamma, beta = film_params.chunk(2, dim=1)  # [B, D] each
        
        # Reshape for broadcasting
        gamma = gamma.view(batch_size, feature_dim, 1, 1)
        beta = beta.view(batch_size, feature_dim, 1, 1)
        
        # Apply FiLM
        return features * gamma + beta
    
    def _apply_area_constraint(self, density_map: torch.Tensor) -> torch.Tensor:
        """Apply minimum area constraint to density map with gradient support
        
        Args:
            density_map: Density map [B, 1, H, W]
            
        Returns:
            Constrained density map [B, 1, H, W]
        """
        batch_size = density_map.shape[0]
        total_pixels = density_map.shape[2] * density_map.shape[3]
        min_area = total_pixels * self.min_area_ratio
        
        # Compute scaling factors while preserving gradients
        current_sums = density_map.view(batch_size, -1).sum(dim=1)  # [B]
        scale_factors = torch.where(
            current_sums < min_area,
            min_area / (current_sums + 1e-8),
            torch.ones_like(current_sums)
        )  # [B]
        
        # Apply scaling with gradient support
        scaled_density = density_map * scale_factors.view(-1, 1, 1, 1)
        
        # Clamp to [0, 1] while preserving gradients
        constrained_density = torch.clamp(scaled_density, max=1.0)
        
        return constrained_density
    
    def _weighted_downsample(self,
                            images: torch.Tensor,
                            density_map: torch.Tensor,
                            target_size: Tuple[int, int]) -> torch.Tensor:
        """Perform normalized weighted downsampling
        
        Args:
            images: Input images [B, C, H, W]
            density_map: Significance density [B, 1, H, W]
            target_size: Target size (H_out, W_out)
            
        Returns:
            Downsampled images [B, C, H_out, W_out]
        """
        batch_size, _, height, width = images.shape
        target_height, target_width = target_size
        
        # Create sampling grid for target size
        grid_h = torch.linspace(0, height - 1, target_height, device=images.device)
        grid_w = torch.linspace(0, width - 1, target_width, device=images.device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        
        # Convert to normalized coordinates for grid_sample
        grid_h = 2.0 * grid_h / (height - 1) - 1.0
        grid_w = 2.0 * grid_w / (width - 1) - 1.0
        
        # Create sampling grid [B, H_out, W_out, 2]
        grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0)
        grid = grid.expand(batch_size, -1, -1, -1)
        
        # Vectorized processing - handle all channels simultaneously
        # Sample images and density at target locations
        sampled_images = F.grid_sample(
            images, grid, mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, C, H_out, W_out]
        
        sampled_densities = F.grid_sample(
            density_map, grid, mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, 1, H_out, W_out]
        
        # Create local averaging kernel for proper normalization
        kernel_size = max(3, min(7, max(height // target_height, width // target_width)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian-like weights to density for smooth local averaging
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=images.device)
        kernel = kernel / kernel.sum()  # Normalize kernel
        
        # Compute local density sums for proper normalization
        local_density_sums = F.conv2d(density_map, kernel, padding=kernel_size//2)
        local_density_sums = F.grid_sample(
            local_density_sums, grid, mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, 1, H_out, W_out]
        
        # Apply density weighting: pixel values are weighted by local density importance
        # This ensures that regions with higher density get more importance in the output
        weighted_output = sampled_images * sampled_densities  # [B, C, H_out, W_out]
        
        # Normalize by local density to avoid bias and ensure consistent scaling
        normalized_output = weighted_output / (local_density_sums + 1e-8)
        
        return normalized_output
    
    def compute_regularization_loss(self, 
                                   images: torch.Tensor,
                                   compressed_images: torch.Tensor,
                                   text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute method-specific regularization losses
        
        Args:
            images: Original input images [B, C, H, W]
            compressed_images: Compressed output images [B, C, H_out, W_out]
            text_embeddings: Query embeddings [B, D] or [B, L, D]
            
        Returns:
            Dictionary of loss terms
        """
        losses = {}
        
        # Extract density map for regularization
        with torch.no_grad():
            visual_features = self.light_cnn(images)
            film_params = self.text_proj(text_embeddings)
            filmed_features = self._apply_film(visual_features, film_params)
            density_map = self.density_predictor(filmed_features)
        
        # Total Variation regularization
        if self.lambda_tv > 0:
            tv_loss = RegularizationUtils.compute_smoothness_regularization(
                density_map, reduction='mean'
            )
            losses['tv_loss'] = self.lambda_tv * tv_loss
        
        # Area constraint regularization (encourage compactness)
        if self.lambda_area > 0:
            # Penalize overly dispersed density maps
            mean_density = density_map.mean(dim=[2, 3])  # [B, 1]
            std_density = density_map.std(dim=[2, 3])   # [B, 1]
            
            # Want high mean density but low std (compact and concentrated)
            area_loss = torch.relu(0.1 - mean_density).mean() + std_density.mean()
            losses['area_loss'] = self.lambda_area * area_loss
        
        return losses


# ============================================================================
# Utility functions for testing and standalone usage
# ============================================================================

def create_weighted_downsample_config(**kwargs) -> Dict[str, Any]:
    """Create configuration for WeightedDownsample
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        'method': 'A',
        'target_size': [448, 448],
        'text_dim': 1536,
        'vision_dim': 256,
        'density_hidden_dim': 256,
        'density_layers': 2,
        'lambda_tv': 1e-4,
        'lambda_area': 1e-3,
        'min_area_ratio': 0.1,
    }
    
    default_config.update(kwargs)
    return default_config


# ============================================================================
# Module testing
# ============================================================================

if __name__ == "__main__":
    # Simple test without torch dependency
    print("WeightedDownsample module loaded successfully!")
    print("Available classes:")
    print("  - LightCNN")
    print("  - TextProjection") 
    print("  - DensityPredictor")
    print("  - WeightedDownsample")
    print("  - create_weighted_downsample_config")