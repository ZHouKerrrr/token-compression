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
    
    def __init__(self, ):
        super().__init__()
        
    
    def forward(
        self, 
        keep_ratio: torch.Tensor,
        lambda_rate: float = 0.4,
    ) -> torch.Tensor:
        """Compute rate loss
        
        Args:
            keep_prob: Soft probabilities of keeping tokens [B, N]
            
        Returns:
            Rate loss
        """
        loss = keep_ratio.mean(dim=0)
        
        return loss * lambda_rate

class KLLoss(nn.Module):
    """Feature kl division loss module"""
    
    def __init__(self):
        super().__init__()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        lambda_kd: float = 1.0
    ) -> torch.Tensor:
        """Compute KL divergence distillation loss
        
        Args:
            student_logits: Student model logits [B, N, V]
            teacher_logits: Teacher model logits [B, N, V]
            valid_mask: Optional mask for valid token positions [B, N]
        Returns:
            Distillation loss
        """
        # 【修复】必须切断 teacher 的梯度传导！
        teacher_logits = teacher_logits.detach() 

        if valid_mask is not None:
            # Mask out invalid positions
            student_logits = student_logits[valid_mask]
            teacher_logits = teacher_logits[valid_mask]
            
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

        kd_loss = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="batchmean",
            log_target=True,
        ) * (temperature ** 2)

        return lambda_kd * kd_loss

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.MSELoss(reduction="mean")

    def forward(
        self, 
        student_feat, 
        teacher_feat, 
        valid_mask=None, 
        lambda_mse=1.0
    ):
        if valid_mask is not None:
            student_feat = student_feat[valid_mask]
            teacher_feat = teacher_feat[valid_mask]

        teacher_feat = teacher_feat.detach()

        student_feat = F.normalize(student_feat.float(), p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat.float(), p=2, dim=-1)

        return lambda_mse * self.loss_fct(student_feat, teacher_feat)

class TokensSamplerRegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        scores: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        spatial_merge_size: int = 2,
        lambda_compact: float = 0.01, 
        lambda_tv: float = 0.01,
        eps: float = 1e-6,
    )-> torch.Tensor:
        """
        scores: [B, N] or [B, N, 1], soft mask / probabilities before hard top-k
        image_grid_thw: [B, 3] = (T, H, W)
        returns:
            loss_compact, loss_tv
        """
        if scores.dim() == 3:
            scores = scores.squeeze(-1)

        B = scores.size(0)
        device = scores.device

        compact_losses = []
        tv_losses = []
        for b in range(B):
            t = int(image_grid_thw[b, 0])
            h = int(image_grid_thw[b, 1]) // spatial_merge_size
            w = int(image_grid_thw[b, 2]) // spatial_merge_size
            n = t * h * w
            s = scores[b, :n].float().reshape(t, h, w).clamp_min(0.0)

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
    
    def __init__(self):
        super().__init__()
        self.kd_loss_fct = KLLoss()
        self.mse_loss_fct = MSELoss()
        self.reg_loss_fct = TokensSamplerRegularizationLoss()
        self.rate_loss_fct = RateLoss()
        

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        labels,
        lambda_loss: Dict[str, float],
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
        # parameter list
        lambda_distortion = lambda_loss.get("lambda_distortion", None)
        lambda_kd_logits = lambda_loss.get("lambda_kd_logits", None)
        lambda_mse_feature = lambda_loss.get("lambda_mse_feature", None)
        lambda_compact = lambda_loss.get("lambda_compact", None)
        lambda_tv = lambda_loss.get("lambda_tv", None)
        lambda_rate = lambda_loss.get("lambda_rate", None)
        temperature_kd_logits = lambda_loss.get("temperature_kd_logits", 2.0)
        temperature_kd_feature = lambda_loss.get("temperature_kd_feature", 2.0)
        
        losses = {}
        aux_outputs = students_outputs.get("aux_outputs", None)

        if lambda_distortion is not None:
            losses['task_loss'] = lambda_distortion * students_outputs['loss']
        
        if teacher_outputs is not None:
            student_logits: torch.Tensor = students_outputs.get("logits", None)
            teacher_logits: torch.Tensor = teacher_outputs.get("logits", None)
            student_hidden: torch.Tensor = students_outputs.get("hidden_states", None)
            teacher_hidden: torch.Tensor = teacher_outputs.get("hidden_states", None)

            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels.ne(-100)   # [B, T-1]
            
            if lambda_kd_logits is not None:
                shift_student_logits = student_logits[:, :-1, :].float().contiguous()
                shift_teacher_logits = teacher_logits[:, :-1, :].float().contiguous()
            
                kd_logits_loss = self.kd_loss_fct(
                    student_logits=shift_student_logits,
                    teacher_logits=shift_teacher_logits,
                    valid_mask=valid,
                    temperature=temperature_kd_logits,
                    lambda_kd=lambda_kd_logits,
                )
                losses['kd_logits_loss'] = kd_logits_loss
            
            if lambda_mse_feature is not None:
                last_student_hidden = student_hidden[-1]
                last_teacher_hidden = teacher_hidden[-1]
                shift_student_feature = last_student_hidden[:, :-1, :].contiguous()
                shift_teacher_feature = last_teacher_hidden[:, :-1, :].contiguous()
                kd_feature_loss = self.mse_loss_fct(
                    student_feat=shift_student_feature, 
                    teacher_feat=shift_teacher_feature, 
                    valid_mask=valid,
                    lambda_mse=lambda_mse_feature,
                )
                losses['kd_feature_loss'] = kd_feature_loss

        
        if lambda_tv is not None and lambda_compact is not None:
            if isinstance(lambda_compact, (str)):
                lambda_compact = float(lambda_compact)
            if isinstance(lambda_tv, (str)):
                lambda_tv = float(lambda_tv)
            scores = aux_outputs.get("soft_prune_mask", None)
            image_grid_thw = inputs.get("image_grid_thw", None)
            if scores is not None and image_grid_thw is not None:
                reg_loss = self.reg_loss_fct(
                    scores=scores, 
                    image_grid_thw=image_grid_thw,
                    lambda_compact=lambda_compact,
                    lambda_tv=lambda_tv,
                )
                losses['reg_loss'] = reg_loss
        
        if lambda_rate is not None:
            keep_ratio = aux_outputs.get("keep_ratio", None)
            if keep_ratio is not None:
                rate_loss = self.rate_loss_fct(
                    keep_ratio,
                    lambda_rate,
                )
                losses['rate_loss'] = rate_loss
        return losses

# Helper function
def create_pato_loss() -> PATOLoss:
    """Create PATO loss from config
    
    Args:
        config: PATO configuration
        
    Returns:
        PATOLoss instance
    """
    return PATOLoss()


__all__ = [
    'PATOLoss',
    'DistillationLoss',
    'BudgetRegularizationLoss',
    'create_pato_loss',
]
