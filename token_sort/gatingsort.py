"""TokenSort Mode B: Random Gating (Hard-Concrete/L0)

方案概述
-------
为每个 token 学习保留概率 p_i，训练时采样二值门 g_i ∈ {0,1}（使用
Hard-Concrete 或 L0 正则化实现可微近似）。推理时按概率排序取前 M 个
或使用阈值筛选。

核心思想
-------
1. 为每个 token 学习保留概率 p_i（基于查询条件化）
2. 训练时：采样二值门 g_i（Hard-Concrete 分布）
3. 应用门控：Z_gated = g ⊙ Z
4. 推理时：按 p_i 排序取前 budget 个
5. L0 正则：鼓励稀疏选择

优势
----
- 直接建模"包含/不包含"的边际贡献
- 对每个 token 的重要性独立评估
- 推理时灵活（排序或阈值）
- 稀疏性可解释

劣势
----
- 采样方差较大（需要多次采样或方差缩减）
- 温度调整敏感
- 不如 Mode A 显式对齐"排序"

Phase 4: P4-B
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTokenSorter


class RandomGatingTokenSorter(BaseTokenSorter):
    """随机门控 Token 排序模块（Mode B）
    
    Pipeline:
        1. 计算 token 保留概率（Query-conditional）
        2. 训练时：采样 Hard-Concrete 门控
        3. 应用门控：gated_tokens = gate * tokens
        4. 推理时：按概率排序取前 budget 个
        5. L0 正则化损失
    
    Architecture:
        logits = Gate(hidden_states, query_embeddings)    # (B, N)
        p_keep = sigmoid(logits)                           # (B, N)
        gates = HardConcrete(logits, tau)                  # (B, N) in {0,1}
        gated_tokens = gates.unsqueeze(2) * hidden_states # (B, N, D)
        # 推理: 按 p_keep 排序取前 budget
    
    Config Parameters:
        - tau: Hard-Concrete 温度参数（默认 0.5）
        - lambda_l0: L0 正则系数（默认 1e-3）
        - stretch_limits: Hard-Concrete 拉伸范围 (l, r)（默认 -0.1, 1.1）
        - gate_hidden_dim: 门控网络隐藏维度（默认 256）
        - use_straight_through: 是否使用 straight-through estimator（默认 True）
    """
    
    def _setup_module(self) -> None:
        """初始化模块组件"""
        # Hard-Concrete 参数
        self.tau = self.config.get('tau', 0.5)
        self.lambda_l0 = self.config.get('lambda_l0', 1e-3)
        stretch_limits = self.config.get('stretch_limits', [-0.1, 1.1])
        self.l = stretch_limits[0]  # left boundary
        self.r = stretch_limits[1]  # right boundary
        self.gate_hidden_dim = self.config.get('gate_hidden_dim', 256)
        self.use_straight_through = self.config.get('use_straight_through', True)
        
        # 特征维度
        token_dim = self.context.get('hidden_size', 768)
        query_dim = self.context.get('hidden_size', 768)
        
        # Token 门控网络（Query-conditional）
        # 输入: [token_features, query_embeddings, mean_token]
        self.gate_network = nn.Sequential(
            nn.Linear(token_dim + query_dim + token_dim, self.gate_hidden_dim),
            nn.LayerNorm(self.gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.gate_hidden_dim, self.gate_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.gate_hidden_dim // 2, 1),  # 输出 logit（未归一化）
        )
    
    def _compute_gate_logits(
        self,
        hidden_states: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算 token 门控 logits（Query-conditional）
        
        Args:
            hidden_states: (B, N, D) token embeddings
            query_embeddings: (B, D_q) 查询嵌入
        
        Returns:
            logits: (B, N) 每个 token 的门控 logits
        """
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
        
        # 门控 logits: (B*N, 1) -> (B*N,)
        logits_flat = self.gate_network(combined_flat).squeeze(-1)
        
        # Reshape back: (B, N)
        logits = logits_flat.view(B, N)
        
        return logits
    
    def _sample_hard_concrete_gates(
        self,
        logits: torch.Tensor,
        tau: float,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样 Hard-Concrete 门控
        
        Args:
            logits: (B, N) 门控 logits
            tau: 温度参数
            training: 是否训练模式
        
        Returns:
            gates: (B, N) 二值门 [0, 1]
            probs: (B, N) 保留概率
        """
        # 保留概率
        probs = torch.sigmoid(logits)  # (B, N)
        
        if training:
            # 训练时：采样 Hard-Concrete
            # 1. Sample from Uniform(0, 1)
            u = torch.rand_like(logits)
            
            # 2. Reparameterization: s = sigmoid((log(u/(1-u)) + logits) / tau)
            s = torch.sigmoid(
                (torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + logits) / tau
            )
            
            # 3. Stretch and clip: z = s * (r - l) + l
            z = s * (self.r - self.l) + self.l
            gates_soft = torch.clamp(z, 0, 1)  # (B, N)
            
            # 4. Straight-through estimator (可选)
            if self.use_straight_through:
                # Forward: binary, Backward: soft
                gates_hard = (gates_soft > 0.5).float()
                gates = gates_hard - gates_soft.detach() + gates_soft
            else:
                gates = gates_soft
        else:
            # 推理时：确定性
            s = torch.sigmoid(logits / tau)
            z = s * (self.r - self.l) + self.l
            gates = torch.clamp(z, 0, 1)
        
        return gates, probs
    
    def _compute_l0_loss(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """计算 L0 正则化损失
        
        L0 ≈ Σ_i P(gate_i = 1)
        
        Args:
            probs: (B, N) 保留概率
        
        Returns:
            l0_loss: 标量
        """
        # 期望的门数量
        expected_num_gates = probs.sum(dim=1).mean()  # 平均每个样本的门数
        
        # L0 正则：鼓励稀疏
        l0_loss = expected_num_gates
        
        return l0_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """执行随机门控 Token 选择
        
        Args:
            hidden_states: (B, N, D) token embeddings
            attention_mask: (B, N) token 有效性掩码
            budget: 目标 token 数量
            query_embeddings: (B, D) 查询嵌入
        
        Returns:
            selected_tokens: (B, budget, D) 选中的 tokens
            sort_indices: (B, N) 排序索引（基于概率）
            aux_outputs: 辅助输出字典
        """
        B, N, D = hidden_states.shape
        
        # 处理默认值
        if query_embeddings is None:
            query_dim = self.context.get('hidden_size', 768)
            query_embeddings = torch.zeros(
                B, query_dim,
                dtype=hidden_states.dtype,
            )
        
        if budget is None:
            budget = self.config.get('budgets', [N])[0]
        
        budget = min(budget, N)
        
        # ============================================================
        # Step 1: 计算门控 logits
        # ============================================================
        logits = self._compute_gate_logits(hidden_states, query_embeddings)  # (B, N)
        
        # 应用 attention_mask
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, -1e9)
        
        # ============================================================
        # Step 2: 采样 Hard-Concrete 门控
        # ============================================================
        gates, probs = self._sample_hard_concrete_gates(
            logits, self.tau, training=self.training
        )  # (B, N)
        
        # ============================================================
        # Step 3: 训练时 - 应用门控
        # ============================================================
        if self.training:
            # 门控 tokens
            gates_expanded = gates.unsqueeze(2)  # (B, N, 1)
            gated_tokens = gates_expanded * hidden_states  # (B, N, D)
            
            # 简化：直接取前 budget 个（按门控值排序）
            # 实际训练中，loss 会作用在所有 gated_tokens 上
            # 这里为了输出固定大小，我们按门控值排序
            gate_indices = torch.argsort(gates, dim=1, descending=True)  # (B, N)
            selected_indices = gate_indices[:, :budget]  # (B, budget)
            
            # 使用 gather 选择
            selected_indices_expanded = selected_indices.unsqueeze(2).expand(-1, -1, D)
            selected_tokens = torch.gather(gated_tokens, 1, selected_indices_expanded)
        else:
            # ============================================================
            # Step 4: 推理时 - 按概率排序取前 budget
            # ============================================================
            # 按保留概率排序
            prob_indices = torch.argsort(probs, dim=1, descending=True)  # (B, N)
            selected_indices = prob_indices[:, :budget]  # (B, budget)
            
            # 使用 gather 选择
            selected_indices_expanded = selected_indices.unsqueeze(2).expand(-1, -1, D)
            selected_tokens = torch.gather(hidden_states, 1, selected_indices_expanded)
        
        # ============================================================
        # Step 5: 排序索引（基于概率，用于可视化）
        # ============================================================
        sort_indices = torch.argsort(probs, dim=1, descending=True)  # (B, N)
        
        # ============================================================
        # Step 6: 计算 L0 损失
        # ============================================================
        l0_loss = self._compute_l0_loss(probs)
        
        # ============================================================
        # Step 7: 辅助输出
        # ============================================================
        # 统计实际保留的 token 数量
        actual_kept = (gates > 0.5).sum(dim=1).float().mean().item()
        
        aux_outputs = {
            'gate_probs': probs.detach(),
            'gates': gates.detach(),
            'sort_indices': sort_indices.detach(),
            'num_tokens_before': N,
            'num_tokens_after': budget,
            'sparsity': 1.0 - (budget / N),
            'actual_kept': actual_kept,
            # 正则化损失
            'l0_loss': l0_loss,
        }
        
        return selected_tokens, sort_indices, aux_outputs
    
    def compute_budget_loss(self, aux_outputs: Dict[str, Any]) -> torch.Tensor:
        """计算 token 预算约束损失
        
        Loss = λ_l0 * L0
        
        Args:
            aux_outputs: forward() 返回的辅助输出
        
        Returns:
            budget_loss: 标量损失张量
        """
        l0_loss = aux_outputs.get('l0_loss', 0.0)
        
        budget_loss = self.lambda_l0 * l0_loss
        
        return budget_loss


__all__ = ['RandomGatingTokenSorter']
