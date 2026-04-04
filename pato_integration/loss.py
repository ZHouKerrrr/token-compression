"""
TODO 需要把所有的loss 都集成进来
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch


class RateLoss(nn.Module):
    """Loss to encourage the token selection to meet a target budget"""
    
    def __init__(self, lambda_rate: float = 0.01):
        super().__init__()
        self.lambda_rate = lambda_rate
    
    def forward(self, keep_ratio: torch.Tensor) -> torch.Tensor:
        """Compute rate loss
        
        Args:
            keep_prob: Soft probabilities of keeping tokens [B, N]
            
        Returns:
            Rate loss
        """
        loss = keep_ratio
        
        return loss * self.lambda_rate

class KDLoss(nn.Module):
    """Feature distillation loss module"""
    
    def __init__(
        self, 
        temperature: float = 1.0,
        lambda_kd: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_kd = lambda_kd

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence distillation loss
        
        Args:
            student_logits: Student model logits [B, N, V]
            teacher_logits: Teacher model logits [B, N, V]
            valid_mask: Optional mask for valid token positions [B, N]
        Returns:
            Distillation loss
        """
        if valid_mask is not None:
            # Mask out invalid positions
            student_logits = student_logits[valid_mask]
            teacher_logits = teacher_logits[valid_mask]
            
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        kd_loss = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="batchmean",
            log_target=True,
        ) * (self.temperature ** 2)

        return self.lambda_kd * kd_loss

class TokensSamplerRegularizationLoss(nn.Module):
    def __init__(self, lambda_compact: float = 0.01, lambda_tv: float = 0.01):
        super().__init__()
        self.lambda_compact = lambda_compact
        self.lambda_tv = lambda_tv

    def forward(
        self, 
        soft_scores: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        spatial_merge_size: int = 2,
        lambda_compact: float = 1.0,
        lambda_tv: float = 1.0,
        eps: float = 1e-6,
    )-> torch.Tensor:
        """
        soft_scores: [B, N] or [B, N, 1], soft mask / probabilities before hard top-k
        image_grid_thw: [B, 3] = (T, H, W)
        returns:
            loss_compact, loss_tv
        """
        if soft_scores.dim() == 3:
            soft_scores = soft_scores.squeeze(-1)

        B = soft_scores.size(0)
        device = soft_scores.device

        compact_losses = []
        tv_losses = []

        for b in range(B):
            t = int(image_grid_thw[b, 0].item())
            h = int(image_grid_thw[b, 1].item()) // spatial_merge_size
            w = int(image_grid_thw[b, 2].item()) // spatial_merge_size

            n = t * h * w
            s = soft_scores[b, :n].float().reshape(t, h, w).clamp_min(0.0)

            p = s / s.sum().clamp_min(eps)

            yy = torch.linspace(-1.0, 1.0, h, device=device).view(1, h, 1)
            xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, w)

            mu_x = (p * xx).sum()
            mu_y = (p * yy).sum()

            compact = (p * ((xx - mu_x) ** 2 + (yy - mu_y) ** 2)).sum()

            # TV smoothness
            tv = 0.0
            if h > 1:
                tv = tv + (s[:, 1:, :] - s[:, :-1, :]).abs().mean()
            if w > 1:
                tv = tv + (s[:, :, 1:] - s[:, :, :-1]).abs().mean()
            if t > 1:
                tv = tv + 0.3 * (s[1:, :, :] - s[:-1, :, :]).abs().mean()

            compact_losses.append(compact)
            tv_losses.append(tv)

        loss_compact = torch.stack(compact_losses).mean() * lambda_compact
        loss_tv = torch.stack(tv_losses).mean() * lambda_tv
        return (loss_compact + loss_tv)


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
        loss_config: Optional[Dict],
    ):
        super().__init__()
        self.kd_logits_loss_fct = KDLoss(
            temperature=loss_config.get("temperature_kd_logits", 2.0),
            lambda_kd=loss_config.get("lambda_kd_logits", 0.5)
        )
        self.kd_feature_loss_fct = KDLoss(
            temperature=loss_config.get("temperature_kd_feature", 2.0),
            lambda_kd=loss_config.get("lambda_kd_feature", 0.5)
        )
        self.reg_loss_fct = TokensSamplerRegularizationLoss(
            lambda_compact=loss_config.get("lambda_compact", 0.01),
            lambda_tv=loss_config.get("lambda_tv", 0.01)
        )
        self.rate_loss_fct = RateLoss(
            lambda_rate=loss_config.get("lambda_rate", 0.4)
        )
        self.lambda_distortion = loss_config.get("lambda_distortion", 0.01)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        students_outputs: Optional[Dict],
        teacher_outputs: Optional[Dict],
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
        losses = {}
        aux_outputs = students_outputs.get("aux_outputs", None)

        if students_outputs is not None:
            losses['task_loss'] = self.lambda_distortion * students_outputs['loss']
        
        if self.kd_loss_fct is not None and teacher_outputs is not None:
            student_logits = students_outputs.get("student_logits", None)
            teacher_logits = teacher_outputs.get("teacher_logits", None)
            student_hidden_states = students_outputs.get("hidden_states", None)
            teacher_hidden_states = teacher_outputs.get("hidden_states", None)
            
            labels = inputs["labels"]
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels.ne(-100)   # [B, T-1]
            if student_logits is not None and teacher_logits is not None:
                shift_student_logits = student_logits[:, :-1, :].contiguous()
                shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
            
                kd_logits_loss = self.kd_loss_fct(shift_student_logits, shift_teacher_logits, valid)
                losses['kd_logits_loss'] = kd_logits_loss
            
            if student_hidden_states is not None and teacher_hidden_states is not None:
                shift_student = student_hidden_states[:, :-1, :].contiguous()
                kd_feature_loss = self.kd_loss_fct(shift_student, teacher_hidden_states)
                losses['kd_feature_loss'] = kd_feature_loss

        
        if self.reg_loss_fct is not None and students_outputs is not None:
            soft_scores = students_outputs.get("soft_scores", None)
            image_grid_thw = students_outputs.get("image_grid_thw", None)
            if soft_scores is not None and image_grid_thw is not None:
                reg_loss = self.reg_loss_fct(soft_scores, image_grid_thw)
                losses['regularization_loss'] = reg_loss
        
        if self.rate_loss_fct is not None:
            keep_ratio = aux_outputs.get("keep_ratio", None)
            if keep_ratio is not None:
                rate_loss = self.rate_loss_fct(keep_ratio)
                losses['rate_loss'] = rate_loss
        
        return losses

# Helper function
def create_pato_loss(loss_config) -> PATOLoss:
    """Create PATO loss from config
    
    Args:
        config: PATO configuration
        
    Returns:
        PATOLoss instance
    """
    return PATOLoss(loss_config)


__all__ = [
    'PATOLoss',
    'DistillationLoss',
    'BudgetRegularizationLoss',
    'create_pato_loss',
]
