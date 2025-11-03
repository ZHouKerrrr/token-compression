"""PATO Loss Functions

This module implements loss functions for training PATO components:
1. Feature Distillation Loss: Bridge g_raw output with standard downsampling
2. Token Sort Regularization Loss: Encourage efficient token selection
3. Contrastive Loss: Learn query-conditional representations
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PATOLoss(nn.Module):
    """Complete PATO loss computation
    
    Combines multiple loss terms with configurable weights:
    - Language modeling loss (from base model)
    - Feature distillation loss (g_raw)
    - Token sort regularization
    - Contrastive loss (optional)
    """
    
    def __init__(
        self,
        lambda_distill: float = 0.05,
        lambda_sort_reg: float = 0.01,
        lambda_contrast: float = 0.1,
        lambda_g_raw_reg: float = 0.001,
    ):
        super().__init__()
        
        self.lambda_distill = lambda_distill
        self.lambda_sort_reg = lambda_sort_reg
        self.lambda_contrast = lambda_contrast
        self.lambda_g_raw_reg = lambda_g_raw_reg
    
    def forward(
        self,
        lm_loss: torch.Tensor,
        aux_outputs: Optional[Dict] = None,
        g_raw_module: Optional[nn.Module] = None,
        token_sorter: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute complete PATO loss
        
        Args:
            lm_loss: Language modeling loss from base model
            aux_outputs: Auxiliary outputs from forward pass
            g_raw_module: g_raw module for regularization
            token_sorter: Token sorter module for regularization
            
        Returns:
            Dictionary of loss terms
        """
        losses = {'lm_loss': lm_loss}
        
        if aux_outputs is None:
            losses['total_loss'] = lm_loss
            return losses
        
        # ============================================================
        # Feature Distillation Loss (g_raw)
        # ============================================================
        if 'original_pixel_values' in aux_outputs and 'compressed_pixel_values' in aux_outputs:
            distill_loss = self.compute_distillation_loss(
                original_images=aux_outputs['original_pixel_values'],
                compressed_images=aux_outputs['compressed_pixel_values']
            )
            losses['distill_loss'] = distill_loss * self.lambda_distill
        
        # ============================================================
        # g_raw Regularization Loss
        # ============================================================
        if g_raw_module is not None and 'compressed_pixel_values' in aux_outputs:
            try:
                g_raw_reg = g_raw_module.compute_regularization_loss(
                    images=aux_outputs['original_pixel_values'],
                    compressed_images=aux_outputs['compressed_pixel_values'],
                    text_embeddings=aux_outputs.get('query_embeds', None)
                )
                
                for key, value in g_raw_reg.items():
                    losses[f'g_raw_{key}'] = value * self.lambda_g_raw_reg
            except Exception as e:
                print(f"Warning: Could not compute g_raw regularization: {e}")
        
        # ============================================================
        # Token Sort Regularization Loss
        # ============================================================
        if token_sorter is not None and aux_outputs:
            try:
                sort_loss = token_sorter.compute_budget_loss(aux_outputs)
                losses['sort_reg_loss'] = sort_loss * self.lambda_sort_reg
            except Exception as e:
                print(f"Warning: Could not compute token sort loss: {e}")
        
        # ============================================================
        # Total Loss
        # ============================================================
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    @staticmethod
    def compute_distillation_loss(
        original_images: torch.Tensor,
        compressed_images: torch.Tensor,
        feature_extractor: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Compute feature distillation loss
        
        This loss encourages the compressed image to preserve important
        visual features from the original image.
        
        Args:
            original_images: Original images [B, C, H, W]
            compressed_images: Compressed images [B, C, H', W']
            feature_extractor: Optional feature extractor (uses raw pixels if None)
            
        Returns:
            Distillation loss
        """
        if feature_extractor is not None:
            # Extract features
            with torch.no_grad():
                original_features = feature_extractor(original_images)
            compressed_features = feature_extractor(compressed_images)
            
            # MSE loss
            loss = F.mse_loss(compressed_features, original_features)
        else:
            # Simple pixel-level loss with resizing
            # Resize original to compressed size
            B, C, H_comp, W_comp = compressed_images.shape
            original_resized = F.interpolate(
                original_images,
                size=(H_comp, W_comp),
                mode='bilinear',
                align_corners=False
            )
            
            # MSE loss on resized
            loss = F.mse_loss(compressed_images, original_resized)
        
        return loss
    
    @staticmethod
    def compute_contrastive_loss(
        features: torch.Tensor,
        query_embeds: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Compute contrastive loss for query-conditional learning
        
        This loss encourages features to be similar for the same query
        and different for different queries.
        
        Args:
            features: Visual features [B, D]
            query_embeds: Query embeddings [B, D]
            temperature: Temperature for softmax
            
        Returns:
            Contrastive loss
        """
        # Normalize
        features = F.normalize(features, dim=1)
        query_embeds = F.normalize(query_embeds, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(features, query_embeds.T) / temperature  # [B, B]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(features.shape[0], device=features.device)
        
        # Symmetric loss
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i + loss_t) / 2
        
        return loss


class DistillationLoss(nn.Module):
    """Feature distillation loss module"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence distillation loss
        
        Args:
            student_logits: Student model logits [B, N, V]
            teacher_logits: Teacher model logits [B, N, V]
            
        Returns:
            Distillation loss
        """
        # Apply temperature
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence
        loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss


class BudgetRegularizationLoss(nn.Module):
    """Budget regularization loss to encourage token efficiency"""
    
    def __init__(self, target_budget: int, lambda_budget: float = 0.01):
        super().__init__()
        self.target_budget = target_budget
        self.lambda_budget = lambda_budget
    
    def forward(self, num_tokens: torch.Tensor) -> torch.Tensor:
        """Penalize deviation from target budget
        
        Args:
            num_tokens: Number of selected tokens [B] or scalar
            
        Returns:
            Budget loss
        """
        if isinstance(num_tokens, int):
            num_tokens = torch.tensor(num_tokens, dtype=torch.float32)
        
        # L1 loss from target
        loss = torch.abs(num_tokens - self.target_budget).mean()
        
        return loss * self.lambda_budget


# Helper function
def create_pato_loss(config) -> PATOLoss:
    """Create PATO loss from config
    
    Args:
        config: PATO configuration
        
    Returns:
        PATOLoss instance
    """
    return PATOLoss(
        lambda_distill=config.lambda_distill,
        lambda_sort_reg=config.lambda_sort_reg,
        lambda_contrast=config.lambda_contrast,
    )


__all__ = [
    'PATOLoss',
    'DistillationLoss',
    'BudgetRegularizationLoss',
    'create_pato_loss',
]
