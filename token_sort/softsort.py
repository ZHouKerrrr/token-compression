"""TokenSort Mode A: Differentiable Sorting (SoftSort/NeuralSort)

方案概述
-------
显式学习 token 的排序置换，使得按边际收益排序。任何前缀 [1:k] 都尽可能
优化最终任务损失（前缀最优）。使用可微的排序近似（如 SoftSort）进行端到端训练。

核心思想
-------
1. 为每个 token 计算重要性分数（基于查询条件化）
2. 使用可微排序生成近似置换矩阵 P
3. 应用排序：Z_sorted = P^T @ Z
4. 随机预算采样：取前 k 个 tokens
5. 正则化：排序熵、去冗余、温度退火

优势
----
- 直接对齐"前缀最优"目标
- 稳定训练，梯度平滑
- 支持任意预算 k
- O(N^2) 复杂度可接受（N 通常不大）

劣势
----
- 需要温度退火（tau: 1.0 -> 0.1）
- O(N^2) 对超长序列有压力
- 排序操作不如门控直观

"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import BaseTokenSorter, register_token_sort


@register_token_sort("A")
class DifferentiableSortingTokenSorter(BaseTokenSorter):
    """可微排序 Token 排序模块（Mode A）
    
    Pipeline:
        1. 计算 token 重要性分数（Query-conditional）
        2. 使用 SoftSort 生成近似置换矩阵
        3. 应用排序得到 Z_sorted
        4. 取前 k 个 tokens（k 可随机采样）
        5. 计算正则化损失（熵、去冗余）
    
    Architecture:
        scores = Scorer(hidden_states, query_embeddings)  # (B, N)
        P = SoftPermutation(scores, tau)                  # (B, N, N)
        sorted_tokens = P^T @ hidden_states               # (B, N, D)
        output_tokens = sorted_tokens[:, :budget, :]      # (B, budget, D)
    
    Config Parameters:
        - tau_init: 初始温度（默认 1.0）
        - tau_final: 最终温度（默认 0.1）
        - tau_decay: 温度衰减策略 ('linear' / 'exponential')
        - lambda_entropy: 排序熵正则系数（默认 1e-3）
        - lambda_diversity: 去冗余正则系数（默认 1e-4）
        - scorer_hidden_dim: 评分器隐藏维度（默认 256）
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
        # 基础参数（支持 dataclass / dict 配置）
        self.tau_init = self._config_value('tau_init', 1.0)
        self.tau_final = self._config_value('tau_final', 0.1)
        self.tau_decay = self._config_value('tau_decay', 'linear')
        self.lambda_entropy = self._config_value('lambda_entropy', 1e-3)
        self.lambda_diversity = self._config_value('lambda_diversity', 1e-4)
        self.scorer_hidden_dim = self._config_value('scorer_hidden_dim', 256)
        self.sinkhorn_iters = self._config_value('sinkhorn_iters', 5)
        
        # 特征维度
        token_dim = self.context.get('out_hidden_size', 2048)
        query_dim = self.context.get('out_hidden_size', 2048)
        
        # 当前温度（训练时动态更新）
        self.register_buffer('current_tau', torch.tensor(self.tau_init))
        self.register_buffer('training_step', torch.tensor(0))
        
        # Token 评分器（Query-conditional）
        # 输入: [token_features, query_embeddings, mean_token]
        self.token_scorer = nn.Sequential(
            nn.Linear(token_dim + query_dim + token_dim, self.scorer_hidden_dim),
            nn.LayerNorm(self.scorer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.scorer_hidden_dim, self.scorer_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.scorer_hidden_dim // 2, 1),  # 输出单个分数
        )
        
        # 设备分配延迟到 forward 调用时进行
        # 避免在 meta device 阶段进行设备转换
    
    def update_temperature(self, current_step: int, total_steps: int) -> None:
        """更新温度参数（用于温度退火）
        
        Args:
            current_step: 当前训练步数
            total_steps: 总训练步数
        """
        total_steps = max(total_steps, 1)
        if self.tau_decay == 'linear':
            # 线性退火
            progress = min(current_step / total_steps, 1.0)
            tau = self.tau_init + (self.tau_final - self.tau_init) * progress
        elif self.tau_decay == 'exponential':
            # 指数退火
            decay_rate = math.log(self.tau_final / self.tau_init) / total_steps
            tau = self.tau_init * math.exp(decay_rate * current_step)
        else:
            tau = self.tau_init
        
        self.current_tau.fill_(tau)
        self.training_step.fill_(current_step)
    
    def _compute_token_scores(
        self,
        hidden_states: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算 token 重要性分数（Query-conditional）
        
        Args:
            hidden_states: (B, N, D) token embeddings
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
        
        # 计算 mean token（全局上下文）
        mean_token = hidden_states.mean(dim=1, keepdim=True)  # (B, 1, D)
        mean_token_expanded = mean_token.expand(B, N, D)  # (B, N, D)
        
        # Expand query embeddings: (B, D_q) -> (B, N, D_q)
        query_expanded = query_embeddings.unsqueeze(1).expand(B, N, -1)
        
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
        
        return scores
    
    def _soft_permutation_matrix(
        self,
        scores: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """生成软置换矩阵（可微近似）
        
        使用 NeuralSort 风格的构造方法。
        
        Args:
            scores: (B, N) token 分数
            tau: 温度参数
        
        Returns:
            P: (B, N, N) 软置换矩阵
            
        Note:
            P[b, i, j] 表示第 j 个 token 被排到第 i 个位置的概率
            P^T @ hidden_states 得到排序后的 tokens
        """
        B, N = scores.shape
        
        # NeuralSort: 使用 pairwise 比较构造置换矩阵
        # 参考: https://arxiv.org/abs/1903.08850
        
        # 1. 计算 pairwise 差异: s_i - s_j
        scores_i = scores.unsqueeze(2)  # (B, N, 1)
        scores_j = scores.unsqueeze(1)  # (B, 1, N)
        pairwise_diff = scores_i - scores_j  # (B, N, N)
        
        # 2. Sigmoid with temperature: σ((s_i - s_j) / τ)
        # 表示 "token i 比 token j 更重要" 的概率
        pairwise_probs = torch.sigmoid(pairwise_diff / tau)  # (B, N, N)
        
        # 3. 计算每个 token 的期望排名（在 0~N-1 之间）
        rank_scores = pairwise_probs.sum(dim=2)  # (B, N)
        # 将最高分 token 对应到 rank ≈ 0，最低分 ≈ N-1
        # Normalize rank_scores into [0, N-1]
        rank_scores = (N - 1) - rank_scores
        
        # 4. 构造 token-位置的对数潜变量并做 Sinkhorn 正则化，使其近似双随机
        target_positions = torch.arange(
            N, device=scores.device, dtype=scores.dtype
        ).view(1, 1, N)  # (1, 1, N)
        
        logits = -((rank_scores.unsqueeze(2) - target_positions) ** 2) / tau  # (B, N, N)
        logits = logits - logits.amax(dim=1, keepdim=True)
        logits = logits - logits.amax(dim=2, keepdim=True)
        P = torch.exp(logits)
        
        eps = 1e-9
        for _ in range(self.sinkhorn_iters):
            P = P / (P.sum(dim=1, keepdim=True) + eps)
            P = P / (P.sum(dim=2, keepdim=True) + eps)
        
        # 返回位置 × token 的矩阵，便于直接左乘隐藏状态
        return P.transpose(1, 2).contiguous()
    
    def _compute_entropy_loss(
        self,
        P: torch.Tensor,
    ) -> torch.Tensor:
        """计算排序熵损失（鼓励确定性排序）
        
        Args:
            P: (B, N, N) 置换矩阵
        
        Returns:
            entropy_loss: 标量
        """
        # 计算每行的熵: -Σ_j P[i,j] * log(P[i,j])
        # 低熵 = 确定性排序
        epsilon = 1e-8
        entropy = -(P * torch.log(P + epsilon)).sum(dim=2)  # (B, N)
        entropy_loss = entropy.mean()
        
        return entropy_loss
    
    def _compute_diversity_loss(
        self,
        sorted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """计算前缀去冗余损失（鼓励多样性）
        
        Args:
            sorted_tokens: (B, N, D) 排序后的 tokens
        
        Returns:
            diversity_loss: 标量
        """
        # 计算相邻 tokens 的余弦相似度
        # 惩罚过高的相似度（鼓励多样性）
        B, N, D = sorted_tokens.shape
        
        if N < 2:
            return torch.tensor(0.0, device=sorted_tokens.device)
        
        # 归一化
        normalized = F.normalize(sorted_tokens, p=2, dim=2)  # (B, N, D)
        
        # 相邻 tokens 的相似度
        token_i = normalized[:, :-1, :]  # (B, N-1, D)
        token_j = normalized[:, 1:, :]   # (B, N-1, D)
        
        # 余弦相似度
        similarity = (token_i * token_j).sum(dim=2)  # (B, N-1)
        
        # 惩罚高相似度（我们希望多样性，所以相似度越低越好）
        # 但不要惩罚太重，因为某些相邻 tokens 可能确实相关
        diversity_loss = F.relu(similarity - 0.5).mean()  # 只惩罚 > 0.5 的相似度
        
        return diversity_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """执行可微排序 Token 选择
        
        Args:
            hidden_states: (B, N, D) token embeddings
            attention_mask: (B, N) token 有效性掩码（1=有效，0=padding）
            budget: 目标 token 数量
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
        
        if budget is None:
            config_budgets = self._config_value('budgets', None)
            if isinstance(config_budgets, (list, tuple)) and len(config_budgets) > 0:
                budget = int(config_budgets[0])
            elif isinstance(config_budgets, int):
                budget = int(config_budgets)
            else:
                budget = N
        
        budget = min(budget, N)  # 不能超过总 token 数

        # TODO 存疑：padding是以0进入计算，是否会影响最终结果，单靠将padding的分数置0，有效否？

        # ============================================================
        # Step 1: 计算 token 重要性分数
        # ============================================================
        scores = self._compute_token_scores(hidden_states, query_embeddings)  # (B, N)
        
        # 应用 attention_mask（如果提供）
        if attention_mask is not None:
            # 将 padding tokens 的分数设为很小的值
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # ============================================================
        # Step 2: 生成软置换矩阵
        # ============================================================
        tau = self.current_tau.item() if self.training else self.tau_final
        P = self._soft_permutation_matrix(scores, tau)  # (B, N, N)
        
        # ============================================================
        # Step 3: 应用排序
        # ============================================================
        # P^T @ hidden_states: (B, N, N) @ (B, N, D) -> (B, N, D)
        sorted_tokens_full = torch.bmm(P, hidden_states)  # (B, N, D)
        # ============================================================
        # Step 4: 截断到目标 budget
        # ============================================================
        sorted_tokens = sorted_tokens_full[:, :budget, :]  # (B, budget, D)
        # ============================================================
        # Step 5: 生成硬排序索引（用于推理和可视化）
        # ============================================================
        # 基于分数的硬排序
        sort_indices = torch.argsort(scores, dim=1, descending=True)  # (B, N)
        # ============================================================
        # Step 6: 计算正则化损失
        # ============================================================
        entropy_loss = self._compute_entropy_loss(P)
        diversity_loss = self._compute_diversity_loss(sorted_tokens_full)
        
        # ============================================================
        # Step 7: 辅助输出
        # ============================================================
        aux_outputs = {
            'scores': scores.detach(),
            'sort_indices': sort_indices.detach(),
            'permutation_matrix': P.detach(),
            'tau': tau,
            'num_tokens_before': N,
            'num_tokens_after': budget,
            'sparsity': 1.0 - (budget / N),
            # 正则化损失
            'entropy_loss': entropy_loss,
            'diversity_loss': diversity_loss,
        }
        
        return sorted_tokens, aux_outputs
    
    def compute_budget_loss(self, aux_outputs: Dict[str, Any]) -> torch.Tensor:
        """计算 token 预算约束损失
        
        Loss = λ_entropy * L_entropy + λ_diversity * L_diversity
        
        Args:
            aux_outputs: forward() 返回的辅助输出
        
        Returns:
            budget_loss: 标量损失张量
        """
        entropy_loss = aux_outputs.get('entropy_loss', 0.0)
        diversity_loss = aux_outputs.get('diversity_loss', 0.0)
        
        budget_loss = (
            self.lambda_entropy * entropy_loss +
            self.lambda_diversity * diversity_loss
        )
        
        return budget_loss


__all__ = ['DifferentiableSortingTokenSorter']
