from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pato_integration.utils import *
from .base import BaseTokenSorter, register_token_sort


class TokenScorer(nn.Module):
    def __init__(
        self,
        token_dim,
    ):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        self.out_conv = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, token_dim // 4),
            nn.GELU(),
            nn.Linear(token_dim // 4, 2),
            nn.LogSoftmax(dim=-1),
        )


    def forward(self, x):
        """
        x : vision tokens
        query_embeds : text tokens
        """
        x = self.in_conv(x)
        B, N, C = x.shape
        local_x = x[:, :, :C // 2]  # [B, N, C//2]
        global_x = x[:, :, C // 2:].mean(dim=1, keepdim=True) # [B, 1, C//2]
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1) # B, N, 2C
        x = self.out_conv(x)
        
        return x

EVAL_KEEP_RATIO = {
    0 : 0.5,
    7 : 0.25,
    14 : 0.01,
}

@register_token_sort('compressor')
class Compressor(BaseTokenSorter):

    def _setup_module(self) -> None:
        self.token_dim = self.context.get('hidden_size', 2048)
        self.layer_idx = self.context.get('layer_idx', 0)
        self.token_scorer = TokenScorer(self.token_dim)
        
        # tau作退火
        self.current_progress = 0.0
        self.tau_start = getattr(self.config, 'tau_start', 1.0)
        self.tau_end = getattr(self.config, 'tau_end', 0.25)
        
        self._init_token_scorer()
        
    
    def _apply_freezing(self):
        for param in self.token_scorer.parameters():
            param.requires_grad = False
            
            
    def _init_token_scorer(self):
        for m in self.token_scorer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                
                
    def _get_current_tau(self):
        p = float(self.current_progress)
        p = max(0.0, min(1.0, p))  # clamp 到 [0, 1]
        tau = self.tau_start + (self.tau_end - self.tau_start) * p
        return tau
    
    
    def _gumbel_softmax(
        self,
        logits: torch.Tensor,
        tau: float = 1,
        hard: bool = False,
        eps: float = 1e-10,
        dim: int = -1,
        gumbel_tau: float = 1.0,
    ) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * gumbel_tau) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
            
        return ret
       

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        aux_outputs = {}
        device = hidden_states.device
        dtype = hidden_states.dtype
        B, N, D = hidden_states.shape
        
        idx = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
        valid_mask = idx < lengths.unsqueeze(-1)            # (1, N) < (B, 1) ==> (B, N)

        binary_scores = self.token_scorer(hidden_states) # (B, N, 2)
        keep_prob = F.softmax(binary_scores, dim=-1)

        topk = True
        if topk:
            keep_prob = keep_prob[:, :, 0] * valid_mask # (B, N)
            mask = torch.zeros_like(keep_prob, device=device, dtype=dtype)
            for i in range(B):
                length = lengths[i].item()
                _, indices = torch.topk(keep_prob[i], k=int(length * EVAL_KEEP_RATIO[self.layer_idx]), dim=-1) # (B, K)
                mask[i].scatter_(0, indices, 1.0)
        else:
            mask = self._gumbel_softmax(binary_scores, hard=True, gumbel_tau=0.0)[:, :, 0]
            mask = mask * valid_mask
            mask = mask.to(dtype=dtype)
        keep_ratio = (mask.sum(dim=1) / lengths.clamp(min=1)).sum()
        
        gumbel_mask = self._gumbel_softmax(binary_scores, hard=True, gumbel_tau=self._get_current_tau())[:, :, 0] # (B, N, 1)
        gumbel_mask = gumbel_mask * valid_mask
        gumbel_mask = gumbel_mask.to(dtype=dtype)
        _keep_ratio = (gumbel_mask.sum(dim=1) / lengths.clamp(min=1)).sum() # ratio 
        
        if self.training:
            aux_outputs = {
                'mask': gumbel_mask,
                'keep_ratio': keep_ratio,

                # "keep_prob": keep_prob,
                
                # '_keep_ratio': _keep_ratio,
            }
        else:
            aux_outputs = {
                'mask': mask,
                'keep_ratio': keep_ratio,
                
                # 'binary_scores': binary_scores,
                # "keep_prob": keep_prob,
                
                # '_keep_ratio': _keep_ratio,
            }
        return aux_outputs

        
__all__ = ['Compressor'] 
