from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import math

from .base import BaseTokenSorter, register_token_sort

class TokenScorer(nn.Module):
    def __init__(
        self,
        query_dim,
        token_dim,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(query_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.Dropout(0.1)
        )
        self.in_conv = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
        )
        self.out_conv = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim // 4),
            nn.GELU(),
            nn.Linear(token_dim // 4, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, query_embeds):
        """
        x : vision tokens
        query_embeds : text tokens
        """
        x = self.in_conv(x)
        B, N, C = x.shape
        local_x = x[:, :, :C // 2]
        global_x = x[:, :, C // 2:].sum(dim=1, keepdim=True)
        query_embeds = self.projector(query_embeds)
        x = torch.cat([local_x, global_x.expand(B, N, C//2), query_embeds], dim=-1) # B, N, 2C
        x = self.out_conv(x)
        return x
    
@register_token_sort("dynamic_token_sorter")
class DynamicTokenSorter(BaseTokenSorter):
    """
    """
    
    def _config_value(self, name: str, default: Any) -> Any:
        """Helper that supports both dataclass and dict configs."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return default

    def _setup_module(self) -> None:
        self.token_dim = self.context.get('out_hidden_size', 2048)
        self.query_dim = self.context.get('out_hidden_size', 2048)
        self.scorer_hidden_dim = self.context.get('scorer_hidden_dim', 256)
        self.token_scorer = TokenScorer(query_dim=self.query_dim, token_dim=self.token_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
        training: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """执行Token剪枝
        
        Args:
            hidden_states: (B, N, D) token embeddings
            lengths: (B,) tokens
            query_embeddings: (B, D) 查询嵌入
        
        Returns:
            sorted_tokens: (B, budget, D) 排序并截断后的 tokens
            sort_indices: (B, N) 排序索引
            aux_outputs: 辅助输出字典
        """
        B, N, D = hidden_states.shape
        if query_embeddings is None:
            query_embeddings = torch.zeros(
                B, self.query_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            if query_embeddings.dtype != hidden_states.dtype:
                query_embeddings = query_embeddings.to(dtype=hidden_states.dtype)

        aux_outputs = {}
        if training:
            device = hidden_states.device
            first_param = next(self.token_scorer.parameters())
            if first_param.device != device:
                self.token_scorer.to(device)
            
            B, N, D = hidden_states.shape
            idx = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
            valid_mask = idx < lengths.unsqueeze(-1)            # (1, N) < (B, 1) ==> (B, N)
            valid_mask = valid_mask.unsqueeze(-1)               # (B, N, -1)
            query_expanded = query_embeddings.unsqueeze(1).expand(B, N, D)
            binary_scores = self.token_scorer(hidden_states, query_expanded) # (B, N, 2)
            hard_mask = F.gumbel_softmax(binary_scores, hard=True)[:, :, 0:1] # (B, N, 1)
            hard_mask = hard_mask * valid_mask
            keep_ratio = hard_mask.sum(dim=1) / lengths.clamp(min=1) # ratio 
  
            aux_outputs = {
                'soft_prune_mask': hard_mask,
                'keep_ratio': keep_ratio,
            }
            return None, aux_outputs
        else:
            device = hidden_states.device
            first_param = next(self.token_scorer.parameters())
            if first_param.device != device:
                self.token_scorer.to(device)
            B, N, D = hidden_states.shape
            idx = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
            valid_mask = idx < lengths.unsqueeze(-1)            # (1, N) < (B, 1) ==> (B, N)
            valid_mask = valid_mask                             # (B, N)
            query_expanded = query_embeddings.unsqueeze(1).expand(B, N, D)
            binary_scores = self.token_scorer(hidden_states, query_expanded) # (B, N, 2)
            keep_prob = F.softmax(binary_scores, dim=-1)[:, :, 0] # (B, N)
            hard_mask = binary_scores.argmax(dim=-1) == 0
            hard_mask = hard_mask * valid_mask # (B, N) & (B, N) ==> (B, N)
            keep_ratio = hard_mask.sum(dim=1) / lengths.clamp(min=1) # ratio 
            filtered_hidden_list = [
                hidden_states[i][hard_mask[i]]
                for i in range(B)
            ]
            filtered_lengths = torch.as_tensor([v.size(0) for v in filtered_hidden_list], device=hidden_states.device)
            aux_outputs.update({
                "filtered_lengths": filtered_lengths,
                "hard_prune_mask": hard_mask,
                "keep_ratio": keep_ratio,
                "keep_prob": keep_prob,
            }) 
            #print(f"DynamicTokenSorter: keep_ratio={keep_ratio.mean().item():.4f}")
            # 把 batch 压扁（flat）成 1 维变长序列
            filtered_hidden = torch.cat(filtered_hidden_list, dim=0) 
            return filtered_hidden, aux_outputs
        
__all__ = ['DynamicTokenSorter'] 
