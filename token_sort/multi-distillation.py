"""TokenSort Mode C: Multi-Budget Distillation

方案概述
-------
多预算蒸馏是一个包装器/辅助模块，与 其他 token sort 方案组合使用。
用大预算 M* 的输出作为 teacher，指导小预算 M 的学习，保证前缀单调性
和小预算的性能。

核心思想
-------
1. 使用基础排序模块（其他 token sort 方案）
2. 对每个训练样本，采样一个学生预算 k
3. 同时计算教师预算 k* (通常为最大预算或全部 tokens)
4. 教师前向：Z_teacher = TokenSort(Z, budget=k*)
5. 学生前向：Z_student = TokenSort(Z, budget=k)
6. KL 散度损失：KL(teacher_logits || student_logits)

优势
----
- 提升小预算性能
- 保证前缀单调性（大预算 ⊃ 小预算的信息）
- 降低训练方差
- 与 其他 token sort 方案无缝集成

劣势
----
- 需要额外的 teacher 前向传播（计算开销 ~2×）
- 需要调整蒸馏温度和权重
- 依赖基础排序模块的质量

"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseTokenSorter


class MultiBudgetDistillationWrapper(BaseTokenSorter):
    """多预算蒸馏包装器（其他 token sort 方案）
    
    这是一个包装器模块，需要与基础排序模块（其他 token sort 方案）组合使用。
    
    Pipeline:
        1. 随机采样学生预算 k ~ U[k_min, k_max]
        2. 教师前向：logits_teacher = Model(tokens[:k*])
        3. 学生前向：logits_student = Model(tokens[:k])
        4. KL 蒸馏损失：KL(soft(teacher, T) || soft(student, T))
        5. 组合损失：CE(y, student) + β * KL
    
    Config Parameters:
        - teacher_budget: 教师预算（'max' 或具体数字，默认 'max'）
        - student_budget_range: 学生预算范围 [k_min, k_max]（默认 [16, 128]）
        - distill_temperature: 蒸馏温度 T（默认 2.0）
        - distill_weight: 蒸馏损失权重 β（默认 0.5）
        - base_sorter: 基础排序模块（必须提供）
    
    Note:
        此模块不直接实现排序逻辑，而是包装基础排序模块并添加蒸馏功能。
        实际的 token 选择由 base_sorter 完成。
    """
    
    def _setup_module(self) -> None:
        """初始化模块组件"""
        # 蒸馏参数
        self.teacher_budget = self.config.get('teacher_budget', 'max')
        self.student_budget_range = self.config.get('student_budget_range', [16, 128])
        self.distill_temperature = self.config.get('distill_temperature', 2.0)
        self.distill_weight = self.config.get('distill_weight', 0.5)
        
        # 基础排序模块
        # 注意：这里不直接实例化，而是在 forward 时从 context 获取
        # 或者作为参数传入
        self.base_sorter = None
        
        # 训练时的预算采样
        self.register_buffer('current_student_budget', torch.tensor(0))
    
    def set_base_sorter(self, base_sorter: BaseTokenSorter) -> None:
        """设置基础排序模块
        
        Args:
            base_sorter: Mode A 或 Mode B 的实例
        """
        self.base_sorter = base_sorter
    
    def sample_student_budget(self) -> int:
        """随机采样学生预算
        
        Returns:
            budget: 采样的预算值
        """
        k_min, k_max = self.student_budget_range
        budget = torch.randint(k_min, k_max + 1, (1,)).item()
        return budget
    
    def compute_kd_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """计算知识蒸馏损失（KL 散度）
        
        Args:
            teacher_logits: (B, vocab_size) 教师输出 logits
            student_logits: (B, vocab_size) 学生输出 logits
            temperature: 蒸馏温度
        
        Returns:
            kd_loss: 标量损失
        """
        # Soft targets
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence
        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)  # 温度平方缩放
        
        return kd_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        budget: Optional[int] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """执行多预算蒸馏 Token 选择
        
        Note:
            此方法只负责采样预算和记录蒸馏信息。
            实际的 token 选择由 base_sorter 完成。
            KL 损失的计算需要在训练循环中进行（因为需要 model logits）。
        
        Args:
            hidden_states: (B, N, D) token embeddings
            attention_mask: (B, N) token 有效性掩码
            budget: 目标 token 数量（如果为 None，随机采样）
            query_embeddings: (B, D) 查询嵌入
        
        Returns:
            selected_tokens: (B, budget, D) 选中的 tokens
            sort_indices: (B, N) 排序索引
            aux_outputs: 辅助输出字典（包含教师预算信息）
        """
        if self.base_sorter is None:
            raise ValueError(
                "base_sorter 未设置！请使用 set_base_sorter() 设置基础排序模块。"
            )
        
        B, N, D = hidden_states.shape
        
        # ============================================================
        # Step 1: 确定学生和教师预算
        # ============================================================
        if self.training and budget is None:
            # 训练时：随机采样学生预算
            student_budget = self.sample_student_budget()
        else:
            # 推理时：使用指定预算或默认值
            student_budget = budget if budget is not None else self.config.get('budgets', [N])[0]
        
        # 教师预算
        if self.teacher_budget == 'max':
            teacher_budget = N
        else:
            teacher_budget = min(int(self.teacher_budget), N)
        
        self.current_student_budget.fill_(student_budget)
        
        # ============================================================
        # Step 2: 使用基础排序模块选择 tokens
        # ============================================================
        # 学生前向（使用学生预算）
        selected_tokens, sort_indices, base_aux = self.base_sorter(
            hidden_states,
            attention_mask=attention_mask,
            budget=student_budget,
            query_embeddings=query_embeddings,
            **kwargs
        )
        
        # ============================================================
        # Step 3: 准备蒸馏信息（教师前向在训练循环中执行）
        # ============================================================
        # 这里只记录教师预算和学生预算
        # 实际的 KL 损失计算需要 model 的输出 logits
        # 由训练循环（GPTrainer）负责
        
        aux_outputs = base_aux.copy()
        aux_outputs.update({
            'student_budget': student_budget,
            'teacher_budget': teacher_budget,
            'distill_temperature': self.distill_temperature,
            'distill_weight': self.distill_weight,
            'use_distillation': self.training,  # 只在训练时使用蒸馏
        })
        
        return selected_tokens, sort_indices, aux_outputs
    
    def compute_budget_loss(self, aux_outputs: Dict[str, Any]) -> torch.Tensor:
        """计算 token 预算约束损失
        
        这里返回基础排序模块的预算损失。
        KL 蒸馏损失由训练循环单独计算。
        
        Args:
            aux_outputs: forward() 返回的辅助输出
        
        Returns:
            budget_loss: 标量损失张量（来自基础排序模块）
        """
        if self.base_sorter is None:
            return torch.tensor(0.0)
        
        # 使用基础排序模块的预算损失
        budget_loss = self.base_sorter.compute_budget_loss(aux_outputs)
        
        return budget_loss


class MultiBudgetDistillationHelper:
    """多预算蒸馏辅助类（用于训练循环）
    
    这个类提供了在训练循环中计算 KL 蒸馏损失的工具方法。
    由 GPTrainer 调用。
    """
    
    @staticmethod
    def should_use_distillation(aux_outputs: Dict[str, Any]) -> bool:
        """判断是否应该使用蒸馏
        
        Args:
            aux_outputs: TokenSort 的辅助输出
        
        Returns:
            use_distillation: 是否使用蒸馏
        """
        return aux_outputs.get('use_distillation', False)
    
    @staticmethod
    def get_teacher_budget(aux_outputs: Dict[str, Any]) -> int:
        """获取教师预算
        
        Args:
            aux_outputs: TokenSort 的辅助输出
        
        Returns:
            teacher_budget: 教师预算
        """
        return aux_outputs.get('teacher_budget', 0)
    
    @staticmethod
    def get_student_budget(aux_outputs: Dict[str, Any]) -> int:
        """获取学生预算
        
        Args:
            aux_outputs: TokenSort 的辅助输出
        
        Returns:
            student_budget: 学生预算
        """
        return aux_outputs.get('student_budget', 0)
    
    @staticmethod
    def compute_kd_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """计算 KL 蒸馏损失
        
        Args:
            teacher_logits: (B, ..., vocab_size) 教师输出
            student_logits: (B, ..., vocab_size) 学生输出
            temperature: 蒸馏温度
        
        Returns:
            kd_loss: 标量损失
        """
        # Flatten if needed
        if teacher_logits.dim() > 2:
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            student_logits = student_logits.view(-1, student_logits.size(-1))
        
        # Soft targets with temperature
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence
        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return kd_loss


__all__ = [
    'MultiBudgetDistillationWrapper',
    'MultiBudgetDistillationHelper',
]
