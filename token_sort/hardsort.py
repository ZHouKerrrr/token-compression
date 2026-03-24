from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import math

from .base import BaseTokenSorter, register_token_sort


@register_token_sort("hard_token_sorter")
class HardTokenSorter(BaseTokenSorter):
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
        """初始化模块组件"""
        # rate loss 参数
        # self.threshold = log(u) - log(1-u)
        self.token_threshold = self.context.get('token_threshold', 0.5)
        self.score_threshold = self.context.get('score_threshold', 0)
        
        self.tau = self.context.get('tau', 1.0)
        # 特征维度
        token_dim = self.context.get('out_hidden_size', 2048)
        query_dim = self.context.get('out_hidden_size', 2048)
        self.scorer_hidden_dim = self.context.get('scorer_hidden_dim', 256)
        # Token 评分器（Query-conditional）
        # 输入: [token_features, query_embeddings, mean_token]
        self.token_scorer = nn.Sequential(
            nn.Linear(token_dim + query_dim + token_dim, self.scorer_hidden_dim),
            nn.LayerNorm(self.scorer_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.scorer_hidden_dim, self.scorer_hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.scorer_hidden_dim // 2, 1),  # 输出单个分数
        )
        # 【高阶技巧】：初始化最后一层 bias，实现高存活率平滑冷启动
        nn.init.xavier_uniform_(self.token_scorer[-1].weight)
        nn.init.constant_(self.token_scorer[-1].bias, 2.0) # 1.5 对应的初始存活率约为 sigmoid(1.5) ≈ 81%
        self.tau_init = self.context.get('tau_init', 1.0)
        self.tau_min = self.context.get('tau_min', 0.05)
        self.current_progress = 0.0  
        self.current_tau = self.tau_init

    def _compute_token_scores(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算 token 重要性分数（Query-conditional）
        
        Args:
            hidden_states: (B, Max_len, D) token embeddings
            length: (B,) length of hidden_states
            query_embeddings: (B, D_q) 查询嵌入
        
        Returns:
            scores: (B, N) 每个 token 的重要性分数
        """
        # 确保模型在输入张量的设备上
        device = hidden_states.device
        # 检查第一个参数的设备，因为 Sequential 没有 device 属性
        first_param = next(self.token_scorer.parameters())
        if first_param.device != device:
            self.token_scorer.to(device)
        
        B, N, D = hidden_states.shape
        idx = torch.arange(N, device=device).unsqueeze(0)   # (1, N)
        valid_mask = idx < lengths.unsqueeze(1)             # (B, N)
        
        # masked mean token（只对有效 token 求均值）
        # 避免 length=0 时除 0（如果理论上不会出现也建议防御一下）
        denom = lengths.clamp(min=1).unsqueeze(1).to(hidden_states.dtype)  # (B, 1)
        masked_sum = (hidden_states * valid_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B,1,D)
        mean_token = masked_sum / denom.unsqueeze(-1)                              # (B,1,D)
        mean_token_expanded = mean_token.expand(B, N, D)

        # Expand query embeddings: (B, D_q) -> (B, N, D_q)
        query_expanded = query_embeddings.unsqueeze(1).expand(B, N, D)
        
        # 拼接特征: [token, query, mean_token]
        combined_features = torch.cat([
            hidden_states,
            query_expanded,
            mean_token_expanded,
        ], dim=2)  # (B, N, 3D)
        
        # Reshape for batch processing: (B*N, 3D)
        combined_flat = combined_features.view(B * N, -1)
        
        # 评分: (B*N, 1) -> (B*N,)
        scores_flat = self.token_scorer(combined_flat).squeeze(-1)
        
        # Reshape back: (B, N)
        scores = scores_flat.view(B, N)

        # 把 PAD 的分数置为极小（不会被剪枝选中）
        very_neg = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(~valid_mask, very_neg)
        return scores

    def epsilon_greedy_mask(logits, epsilon=0.1, temperature=1.0, is_training=True):
        """
        logits: 评分器输出的原始得分 (Batch, N)
        epsilon: 探索概率 (0.1 表示强制保留 10% 的额外 patch)
        """
        # 1. 正常的 Gumbel-Softmax 采样 (Hard Mask)
        # 假设 logits 是两维的 [stop_prob, keep_prob]
        # 或者如果你是单维的，直接做 binary gumbel
        mask_scorer = F.gumbel_softmax(logits, tau=temperature, hard=True)[:, :, 1] # 取 keep 分支
        

        # 2. 生成随机探索掩码 (Bernoulli 采样)
        # 创建一个和 mask_scorer 形状一样的全随机掩码
        # torch.rand_like 生成 0~1 均匀分布，小于 epsilon 的位置设为 1
        mask_rand = (torch.rand_like(mask_scorer) < epsilon).float()

        # 3. 合并掩码 (OR 逻辑)
        # 只要评分器选了，或者随机选了，都设为 1
        mask_combined = torch.max(mask_scorer, mask_rand)

        # 4. Straight-Through Estimator (STE) 技巧
        # 这一步非常关键！
        # 前向传播使用 mask_combined (带随机探索)
        # 反向传播使用 mask_scorer 的梯度 (让评分器感受到被随机选中的 patch 对 Loss 的贡献)
        mask_final = mask_combined.detach() - mask_scorer.detach() + mask_scorer

        return mask_final

    # 在模型 forward 中的用法：
    # logits = scorer(image_embeds) # (B, N, 2)
    # masks = epsilon_greedy_mask(logits, epsilon=0.1) # (B, N)
    # pruned_embeds = image_embeds * masks.unsqueeze(-1) 
    # # 接下来送入 LLM...
    def _compute_gumble_mask_loss(
        self,
        scores: torch.Tensor,
        lengths: torch.Tensor,
        training: bool = False,
        epsilon: float = 0.1, # 新增 epsilon 参数，代表随机探索的概率
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算掩码和剪枝率损失 (带 Epsilon-Greedy 探索)"""
        
        if training:
            # 1. 温度退火逻辑保持不变
            self.current_tau = self.tau_min + 0.5 * (self.tau_init - self.tau_min) * (
                1.0 + math.cos(math.pi * self.current_progress)
            )
            # Gumbel 噪声采样
            u = torch.empty_like(scores).uniform_(eps, 1.0 - eps)
            gumble = torch.log(u) - torch.log(1 - u)
            logits = (scores + gumble) / self.current_tau
        else:
            logits = scores / self.current_tau
            
        soft_mask = torch.sigmoid(logits)
        # 原始的硬掩码 (基于评分器和 Gumbel 噪声)
        hard_mask = (soft_mask > self.token_threshold).float()
        
        # --- 新增探索逻辑 ---
        if training:
            # 生成一个随机掩码：以 epsilon 的概率为 1
            rand_mask = (torch.rand_like(scores) < epsilon).float()
            
            # 使用 max(OR 逻辑)：只要评分器选了，或者随机选了，该位置就是 1
            # 这确保了原本被打低分的 patch 有 epsilon 的几率被 LLM “看到”
            final_hard_mask = torch.max(hard_mask, rand_mask)
        else:
            final_hard_mask = hard_mask
        # ------------------

        # 核心技巧：Straight-Through Estimator (STE)
        # 注意：这里减去的是 soft_mask.detach()，加上的是 soft_mask
        # 前向传播使用的是包含随机探索的 final_hard_mask
        # 反向传播时，由于 (final_hard_mask - soft_mask.detach()) 部分梯度为 0，
        # 梯度会直接流向最后的 + soft_mask，从而让评分器获得反馈。
        mask = final_hard_mask - soft_mask.detach() + soft_mask
        
        # Rate loss 建议依然只针对 soft_mask 计算，
        # 这样评分器优化的目标是“在尽量少选 token 的情况下降低损失”，
        # 而不是把随机探索的那部分也算进成本里。
        keep_ratio = soft_mask.sum(dim=1) / lengths.clamp(min=1)
        
        return final_hard_mask.bool(), mask, keep_ratio
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
        training: bool = False,
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
        # TODO 这里的N并不代表token数，因为有padding
        B, N, D = hidden_states.shape
        
        # 处理默认值
        if query_embeddings is None:
            query_dim = D
            query_embeddings = torch.zeros(
                B, query_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # 计算 token 分数
        scores = self._compute_token_scores(hidden_states, lengths, query_embeddings)

        # 优化点 4：务必把 training 标志传进去
        hard_mask, mask, keep_ratio = self._compute_gumble_mask_loss(
            scores, lengths, training=training
        )
        
        # 使用 STE 掩码进行特征放缩 (保留梯度的关键)
        hidden_states = hidden_states * mask.unsqueeze(-1)
        
        # 提取被保留的 Tokens
        filtered_hidden_list = [
            hidden_states[i][hard_mask[i]]
            for i in range(B)
        ]
        filtered_lengths = torch.as_tensor([v.size(0) for v in filtered_hidden_list], device=hidden_states.device)
        
        # 把 batch 压扁（flat）成 1 维变长序列
        filtered_hidden = torch.cat(filtered_hidden_list, dim=0) 
        
        aux_outputs = {
            'scores': scores.detach(),
            'sorter_mask': hard_mask,
            'filtered_lengths': filtered_lengths,  # 注意拼写：filtered
            'keep_ratio': keep_ratio,
        }
        
        return filtered_hidden, aux_outputs
       
__all__ = ['HardTokenSorter']
