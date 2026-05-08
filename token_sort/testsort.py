from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import math
from pato_integration.utils import *
from .base import BaseTokenSorter, register_token_sort

class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(
        self, 
        x: torch.Tensor,  # (B, N, D):
    ) -> torch.Tensor:
        # x: (B, N, D) -> (N, B, D) for attention
        x = x.transpose(0, 1)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm(x + ffn_output)
        return x.transpose(0, 1)

class TokenScorer(nn.Module):
    def __init__(
        self,
        query_dim,
        token_dim,
        attn_config,
    ):
        super().__init__()
        self.config = attn_config
        # self.global_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid 后大约 0.5
        self.in_conv = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        # self.local_layernorm = nn.LayerNorm(token_dim // 2)
        # self.global_layernorm = nn.LayerNorm(token_dim // 2)
        if attn_config["enable"]:
            self.query_cls = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
            self.attn_layers = nn.ModuleList(
                [AttentionLayer(token_dim) for _ in range(attn_config["layers"])]
            )
            self.projector = nn.Sequential(
                nn.Linear(query_dim, token_dim),
                nn.LayerNorm(token_dim),
                nn.Dropout(0.1)
            )
            self.out_conv = nn.Sequential(
                nn.Linear(token_dim * 2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim // 4),
                nn.GELU(),
                nn.Linear(token_dim // 4, 2),
                nn.LogSoftmax(dim=-1),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(query_dim, token_dim),
                nn.LayerNorm(token_dim),
                nn.Dropout(0.1)
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
        if self.config["enable"]:
            query_embeds = torch.cat([query_embeds, self.query_cls.expand(query_embeds.size(0), -1, -1)], dim=1) # [B, N+1, D]
            for attn_layer in self.attn_layers:
                query_embeds = attn_layer(query_embeds)
            query_rep = query_embeds[:, -1] # (B, C)
        
            x = self.in_conv(x)
            B, N, C = x.shape
            local_x = x[:, :, :C // 2]  # [B, N, C//2]
            global_x = x[:, :, C // 2:].sum(dim=1, keepdim=True) # [B, 1, C//2]
            
            query_rep = self.projector(query_rep) # [B, C]
            query_rep_expanded = query_rep.unsqueeze(1).expand(B, N, C) # (B, N, C)
            
            x = torch.cat([local_x, global_x.expand(B, N, C//2), query_rep_expanded], dim=-1) # B, N, 2C
            x = self.out_conv(x)
        else:
            x = self.in_conv(x)
            B, N, C = x.shape
            local_x = x[:, :, :C // 2]  # [B, N, C//2]
            global_x = x[:, :, C // 2:].mean(dim=1, keepdim=True) # [B, 1, C//2]
            query_embeds = query_embeds.mean(dim=1, keepdim=True) # [B, 1, H]
            query_embeds = self.projector(query_embeds)
            # local_x = self.local_layernorm(local_x)
            # global_x = self.global_layernorm(global_x)
            # g = torch.sigmoid(self.global_gate)
            x = torch.cat([local_x, global_x.expand(B, N, C//2), query_embeds.expand(B, N, C)], dim=-1) # B, N, 2C
            x = self.out_conv(x)
        
        return x
    
@register_token_sort("test_sorter")
class TestTokenSorter(BaseTokenSorter):

    def _config_value(self, name: str, default: Any) -> Any:
        """Helper that supports both dataclass and dict configs."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return default

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

    def _setup_module(self) -> None:
        self.token_dim = self.context.get('out_hidden_size', 2048)
        self.query_dim = self.context.get('out_hidden_size', 2048)
        self.token_scorer = TokenScorer(
            query_dim=self.query_dim, 
            token_dim=self.token_dim, 
            attn_config={"enable": False, "layers": 2}
        )
        
        # tau作退火
        self.current_progress = 0.0
        self.tau_start = getattr(self.config, 'tau_start', 1.0)
        self.tau_end = getattr(self.config, 'tau_end', 0.5)
        self.anchor_idxs = getattr(self.config, 'anchor_idxs', None)

        self._init_token_scorer()
    
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
    ) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * self._get_current_tau()) / tau  # ~Gumbel(logits,tau)
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

    def _tokens_merge(
        self,
        scores: torch.Tensor,         # [B, N]
        mask: torch.Tensor,           # [B, N]
        hidden_states: torch.Tensor,  # [B, N, D]
        grid_thw: torch.Tensor,       # [B, 3]
        lengths: torch.Tensor,        # [B]
        high_t: float = 0.4,
        low_t: float = 0.0,
    ):
        B, N = scores.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        probs = torch.where((scores >= high_t) & (scores < 0.5), torch.tensor(1),
            torch.where((scores <= low_t) | (scores >= 0.5), torch.tensor(0), scores)).to(dtype=dtype, device=device) # [B, N]
        
        method = self.merge_config['method']
        if method == 'patch':
            
            size = self.merge_config['size']
            
            for b in range(B):
                
                t, h, w = grid_thw[b]
                length = lengths[b]
                new_hidden_states = hidden_states.clone()


                for i in range(length):
                    if not mask[b, i]:
                        continue
                    neighbor_idxs = []
                    
                    t_i = i // (h * w)
                    r = i % (h * w)
                    h_i = r // w
                    w_i = r % w
                    
                    h_grid = torch.arange(2 * size + 1, device=device) - size + h_i
                    w_grid = torch.arange(2 * size + 1, device=device) - size + w_i
                    
                    h_grid = h_grid[(h_grid >= 0) & (h_grid < h)]
                    w_grid = w_grid[(w_grid >= 0) & (w_grid < w)]
                    
                    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")   # [Kh, Kw]
                    hh = hh.reshape(-1)
                    ww = ww.reshape(-1)
                    
                    neighbor_idxs = t_i * (h * w) + hh * w + ww
                    p = probs[b, neighbor_idxs]
                    p_mask = (p >= 0.3) & (p <= 0.5)
                    neighbor_idxs = neighbor_idxs[p_mask]
                    
                    if neighbor_idxs.numel():
                        continue
                    weights = (probs[b, neighbor_idxs]) / (probs[b, neighbor_idxs].sum())
                    new_hidden_states[b, i] = 0.75 * hidden_states[b, i] + (hidden_states[b, neighbor_idxs] * weights[:, None]).sum(dim=0) * 0.25
                    
        return new_hidden_states                

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None, # [B, 3]
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
        
        device = hidden_states.device
        dtype = hidden_states.dtype
            
        B, N, D = hidden_states.shape
        
        idx = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
        valid_mask = idx < lengths.unsqueeze(-1)            # (1, N) < (B, 1) ==> (B, N)

        aux_outputs = {}
        if self.anchor_idxs is not None:
            _N = N - len(self.anchor_idxs)
            anchor_mask = torch.zeros(
                (B, N),
                device=device,
                dtype=dtype
            )

            anchor_mask[:, self.anchor_idxs] = 1              # (B, N)
            non_anchor_bool = ~(anchor_mask.bool())           # (B, N)
            non_anchor_mask = non_anchor_bool & valid_mask         # (B, N)
            
            anchor_valid_mask = valid_mask[non_anchor_bool].reshape(B, _N)     # (B, N')
            non_anchor_hidden_states = hidden_states[non_anchor_bool].reshape(B, _N, -1)
            binary_scores = self.token_scorer(non_anchor_hidden_states, query_embeddings) # (B, N', 2)
            
            keep_prob = F.softmax(binary_scores, dim=-1)

            if self.training:
                index = keep_prob.max(-1, keepdim=True)[1]
                y_hard = torch.zeros_like(
                    binary_scores, memory_format=torch.legacy_contiguous_format
                ).scatter_(-1, index, 1.0)
                ret = y_hard - keep_prob.detach() + keep_prob
                ret = ret[:, :, 0:1] * anchor_valid_mask.unsqueeze(-1)
                keep_ratio = (ret.sum(dim=1) + B * len(self.anchor_idxs)) / lengths.clamp(min=1)
                
                soft_mask = self._gumbel_softmax(binary_scores, hard=True)[:, :, 0] # (B, N) 
                valid_soft_mask = soft_mask[anchor_valid_mask].to(dtype=dtype)                  # [flatten_mask]
                anchor_mask[non_anchor_mask] = valid_soft_mask                  # (B, N)
                
                # TODO test what is the best
                # anchor_mask[:, self.anchor_idxs] = 0.0

                _keep_ratio = anchor_mask.sum(dim=1) / lengths.clamp(min=1)       # ratio
                
                aux_outputs = {
                    'soft_prune_mask': anchor_mask.unsqueeze(-1),
                    'keep_ratio': keep_ratio,
                    
                    'binary_scores': binary_scores,
                    "keep_prob": keep_prob,
                    
                    '_keep_ratio': _keep_ratio,
                }
                return None, aux_outputs
            else:
                # use to topk
                self.topk_ratio = 0.25
                self.high_t = 0.4
                self.low_t = 0.3
                if False:
                    k = max(1, int(N * self.topk_ratio))
                    scores = keep_prob[:, :, 0]
                    _, tokp_idxs = torch.topk(scores, k)
                    hard_mask = torch.zeros_like(scores, dtype=torch.bool)
                    hard_mask.scatter_(1, tokp_idxs, True)
                else:
                    # hard_mask = binary_scores.argmax(dim=-1) == 0
                    hard_mask = keep_prob[:, :, 0] > self.high_t
                anchor_mask = anchor_mask.bool()
                valid_hard_mask = hard_mask[anchor_valid_mask] # [flatten]
                anchor_mask[non_anchor_mask] = valid_hard_mask # [anchor mask]
                # anchor_mask[:, self.anchor_idxs] = False
                keep_ratio = anchor_mask.float().sum(dim=1) / lengths.clamp(min=1) # ratio
                
                # use merge
                self.merge_config = {
                    'method' : 'patch',
                    'size' : 1,
                }
                self.merge_config = None
                if self.merge_config:
                    scores = torch.zeros_like(
                        anchor_mask,
                        device=device,
                        dtype=dtype,
                    )
                    scores[non_anchor_mask] = keep_prob[:, :, 0]
                    scores[:, self.anchor_idxs] = 1.0

                    hidden_states = self._tokens_merge(
                        scores=scores, 
                        mask=anchor_mask, 
                        hidden_states=hidden_states, 
                        grid_thw=grid_thw,
                        lengths=lengths,
                        high_t=self.high_t,
                        low_t=self.low_t,
                    )
                    
                
                filtered_hidden_list = [
                    hidden_states[i][anchor_mask[i]]
                    for i in range(B)
                ]
                filtered_lengths = torch.as_tensor([v.size(0) for v in filtered_hidden_list], device=hidden_states.device)
                

                aux_outputs.update({
                    "filtered_lengths": filtered_lengths,
                    "hard_prune_mask": anchor_mask,
                    'keep_ratio': keep_ratio,
                    "keep_prob": keep_prob,
                }) 
                filtered_hidden = torch.cat(filtered_hidden_list, dim=0) 
                return filtered_hidden, aux_outputs
            
        else:
            binary_scores = self.token_scorer(hidden_states, query_embeddings) # (B, N, 2)
            keep_prob = F.softmax(binary_scores, dim=-1)

            if self.training:
                index = keep_prob.max(-1, keepdim=True)[1]
                y_hard = torch.zeros_like(
                    binary_scores, memory_format=torch.legacy_contiguous_format
                ).scatter_(-1, index, 1.0)
                ret = y_hard - keep_prob.detach() + keep_prob
                ret = ret[:, :, 0:1] * valid_mask.unsqueeze(-1)
                keep_ratio = ret.sum(dim=1) / lengths.clamp(min=1)
                
                soft_mask = self._gumbel_softmax(binary_scores, hard=True)[:, :, 0:1] # (B, N, 1)
                soft_mask = soft_mask * valid_mask.unsqueeze(-1)
                _keep_ratio = soft_mask.sum(dim=1) / lengths.clamp(min=1) # ratio 

                aux_outputs = {
                    'soft_prune_mask': soft_mask,
                    'keep_ratio': keep_ratio,
                    
                    'binary_scores': binary_scores,
                    "keep_prob": keep_prob,
                    
                    '_keep_ratio': _keep_ratio,
                }
                return None, aux_outputs
            else:
                """
                    1.将binary_scores[:,:,0] 离散地分为3档: 
                        丢弃:0-0.2 
                        融合:0.2-0.6 
                        保留:0.6-1.0
                    2. 在推理阶段，待融合的tokens通过image_grid_thw找到若干聚类中心，并将待融合的tokens与聚类中心进行融合（例如加权平均），得到新的token特征。
                    3. 第2点的聚类中心用矩阵乘法实现，并将矩阵保留
                """

                if self.use_topk:
                    k = max(1, int(N * self.topk_ratio))
                    scores = keep_prob[:, :, 0] # [B, N]
                    _, tokp_idxs = torch.topk(scores, k)
                    hard_mask = torch.zeros_like(scores, dtype=torch.bool)
                    hard_mask.scatter_(1, tokp_idxs, True)
                else:
                    hard_mask = binary_scores.argmax(dim=-1) == 0

                if self.merge_config:
                    pass
                
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
                    'keep_ratio': keep_ratio,
                    "keep_prob": keep_prob,
                }) 
                filtered_hidden = torch.cat(filtered_hidden_list, dim=0) 
                return filtered_hidden, aux_outputs
        
__all__ = ['TestTokenSorter'] 
