from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pato_integration.utils import *
from .base import BaseTokenSorter, register_token_sort


class NestDropout(nn.Module):

    def __init__(self, layers: int = 28):
        super().__init__()
        self.layers = layers

    def set_lengths(self, lengths: torch.Tensor):
        assert lengths.size(0) == 1, "Only support batch size of 1 for now"
        self.lengths = lengths
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        aux_outputs = {}
        device = hidden_states.device
        dtype = hidden_states.dtype
        B, N, D = hidden_states.shape
        
        idx = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
        valid_mask = idx < lengths.unsqueeze(-1)            # (1, N) < (B, 1) ==> (B, N)

        depth = ((1 - layer_idx / self.layers) * self.lengths).long()  # (B,)

        B, N = valid_mask.shape
        device = valid_mask.device

        # 防止 depth 超过每一行有效 token 数
        valid_counts = valid_mask.sum(dim=-1)  # (B,)
        depth = torch.minimum(depth, valid_counts)

        # 给每个位置生成随机分数
        rand = torch.rand(B, N, device=device)

        # 无效位置设为 -inf，保证不会被选中
        rand = rand.masked_fill(~valid_mask, float("-inf"))

        # 每行随机选最多 max_depth 个
        max_depth = int(depth.max().item())

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        if max_depth > 0:
            _, selected_idx = torch.topk(rand, k=max_depth, dim=-1)  # (B, max_depth)

            # 有些样本 depth 小于 max_depth，只保留前 depth[b] 个
            keep = torch.arange(max_depth, device=device).unsqueeze(0) < depth.unsqueeze(1)

            mask.scatter_(dim=-1, index=selected_idx, src=keep)

        keep_ratio = mask.sum(dim=1) / lengths.clamp(min=1)
        
        if self.training:
            aux_outputs = {
                'mask': mask,
                'keep_ratio': keep_ratio,
            }
        else:
            aux_outputs = {
                'mask': mask,
                'keep_ratio': keep_ratio,
            }
        return aux_outputs

        
__all__ = ['NestDropout'] 
