# PATO-Qwen2.5-VL V1.0 MVP

**PATO (Pixels As The Optimization)** 集成到 Qwen2.5-VL 的最小可行产品实现。

## 🎯 项目目标

将PATO方法集成到Qwen2.5-VL多模态大模型中，通过**像素域预压缩**和**token优化选择**实现端到端可训练的视觉效率提升。

## ✅ V1.0 实现状态

### 核心组件（全部完成）

- [x] **g_raw (像素域预压缩)** - Method A: Weighted Downsampling
  - 文本条件化的显著性预测
  - 可导的加权下采样
  - 完整梯度流支持
  - **效果**: 5.4× 像素压缩，80.9% 内存节省

- [x] **Token Sort (token选择)** - Method A: Differentiable Sorting  
  - Query-conditional token scoring
  - SoftSort可微排序
  - 前缀最优选择
  - **效果**: 75% token reduction (1024 → 256)

- [x] **Simplified Projector** - Method A: Linear Projection
  - 简单线性投影 (vision_dim → hidden_dim)
  - 快速验证，保留所有选中tokens

- [x] **PATO Loss Functions**
  - Feature distillation loss
  - Token sort regularization
  - G_raw regularization (TV + area constraint)

### 测试验证（全部通过）

- [x] **Component Tests** (`tests/test_components.py`)
  - g_raw独立测试 ✓
  - Token Sort独立测试 ✓  
  - PATO Loss测试 ✓

- [x] **Integration Demo** (`tests/test_pato_demo.py`)
  - 完整pipeline模拟 ✓
  - Training mode梯度流验证 ✓
  - **整体效率提升**: ~78% reduction

## 📊 性能指标

| 指标 | Baseline | PATO V1.0 | 改进 |
|------|----------|-----------|------|
| **Pixel Data** | 6.3M | 1.2M | **80.9%** ↓ |
| **Vision Tokens** | 1024 | 256 | **75.0%** ↓ |
| **Vision Encoder FLOPs** | 100% | ~19% | **~81%** ↓ |
| **LLM FLOPs** | 100% | ~25% | **~75%** ↓ |
| **Overall Efficiency** | - | - | **~78%** ↓ |

## 🏗️ 架构设计

```
输入图像 (1024×1024)
    ↓
┌────────────────────────────────────────┐
│ Stage 1: g_raw (Pixel Compression)     │
│   - Text-conditional saliency          │
│   - Weighted downsampling              │
│   - Output: 448×448 (5.4× compressed)  │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Stage 2: Vision Encoder (Frozen)       │
│   - Patch embedding (14×14)            │
│   - Vision transformer (32 layers)     │
│   - Output: 1024 tokens × 1152 dim     │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Stage 3: Token Sort (Selection)        │
│   - Query-conditional scoring          │
│   - Differentiable sorting             │
│   - Budget: 256 tokens                 │
│   - Output: 256 tokens × 1152 dim      │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Stage 4: Simplified Projector          │
│   - Linear: 1152 → 3584                │
│   - Output: 256 tokens × 3584 dim      │
└────────────────────────────────────────┘
    ↓
LLM (Frozen) → Generation
```

## 📁 项目结构

```
PATO/
├── g_raw/                          # 像素域预压缩
│   ├── base.py                     # 基类定义
│   ├── weighted_downsample.py      # Method A实现 ✓
│   └── __init__.py
│
├── token_sort/                     # Token排序选择
│   ├── base.py                     # 基类定义
│   ├── softsort.py                 # Method A实现 ✓
│   ├── gatingsort.py               # Method B实现
│   └── __init__.py
│
├── pato_integration/               # PATO集成
│   ├── pato_config.py              # 配置类
│   ├── pato_config_standalone.py   # 独立配置 ✓
│   ├── pato_model.py               # PATO模型 ✓
│   ├── loss.py                     # 损失函数 ✓
│   └── __init__.py
│
├── tests/                          # 测试脚本
│   ├── test_components.py          # 组件测试 ✓
│   ├── test_pato_demo.py           # 集成演示 ✓
│   └── test_real_integration.py    # 真实模型测试
│
├── training/                       # 训练脚本（待实现）
│   ├── data_loader.py
│   └── train.py
│
├── PATO.md                         # PATO方法文档
├── PATO_Qwen_Integration_Plan.md   # 集成计划
└── README.md                       # 本文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活qwen环境（已安装transformers）
conda activate qwen

# 进入项目目录
cd /home/baoyouneng/LLM_Compression/PATO
```

### 2. 运行测试

```bash
# 测试独立组件
python tests/test_components.py

# 运行完整集成演示
conda run -n qwen python tests/test_pato_demo.py
```

### 3. 预期输出

```
============================================================
✓ PATO INTEGRATION DEMO COMPLETED!
============================================================

PATO V1.0 MVP Summary:
  • g_raw: ✓ Conditional pixel compression (5.4× reduction)
  • Token Sort: ✓ Query-based token selection (75% reduction)
  • Simplified Projector: ✓ Linear projection
  • Gradient Flow: ✓ End-to-end trainable
  • Efficiency: ~78% overall reduction
```

## 🔧 配置参数

### G_Raw配置

```python
g_raw_config = {
    'enable': True,
    'mode': 'A',                    # Weighted Downsampling
    'target_size': (448, 448),      # 压缩后尺寸
    'text_dim': 3584,               # 文本嵌入维度
    'vision_dim': 256,              # 特征维度
    'lambda_tv': 1e-4,              # Total Variation正则
    'lambda_area': 1e-3,            # Area约束
}
```

### Token Sort配置

```python
token_sort_config = {
    'enable': True,
    'mode': 'A',                    # Differentiable Sorting
    'budgets': [256],               # Token数量
    'budget_min': 128,              # 最小budget（训练时随机）
    'budget_max': 512,              # 最大budget
    'tau_init': 1.0,                # 初始温度
    'tau_final': 0.1,               # 最终温度
    'lambda_entropy': 1e-3,         # 熵正则
    'lambda_diversity': 1e-4,       # 多样性正则
}
```

### 训练策略

```python
training_config = {
    'freeze_vision_encoder': True,  # 冻结视觉编码器
    'freeze_llm': True,             # 冻结语言模型
    'freeze_embeddings': True,      # 冻结嵌入层
    
    # 只训练PATO组件
    'trainable_components': [
        'g_raw',                    # ~3.7M params
        'token_sorter',             # ~0.9M params
        'projector',                # ~4.1M params
    ],
    'total_trainable': '~8.8M params'
}
```

## 📈 训练策略

### Phase 1: PATO组件训练（当前V1.0）

- **冻结**: Vision Encoder, LLM
- **训练**: g_raw, Token Sort, Simplified Projector
- **数据**: VQA数据集 (10k-50k样本)
- **Epochs**: 3-5
- **学习率**: 1e-4
- **目标**: 验证PATO有效性

### Phase 2: 端到端微调（V2.0计划）

- **解冻**: 全部参数 或 使用LoRA
- **数据**: 扩展到100k+样本
- **Epochs**: 5-10
- **目标**: 达到或超越baseline性能

## 🎓 技术细节

### 可微分设计

1. **G_raw**
   - 全程PyTorch可导算子
   - FiLM调制实现文本条件化
   - 归一化加权下采样保持梯度流

2. **Token Sort**
   - SoftSort + Sinkhorn算法
   - 温度退火策略 (τ: 1.0 → 0.1)
   - 双随机矩阵约束

3. **Loss Functions**
   - Feature distillation: 桥接压缩与原始特征
   - Budget regularization: 鼓励稀疏选择
   - G_raw TV loss: 平滑显著性图

### 关键创新

- **Query-conditional**: 所有操作都基于文本query条件化
- **Prefix-optimal**: Token排序保证任意前缀都最优
- **End-to-end trainable**: 完整梯度流，统一优化目标

## 📋 下一步计划

### 短期（1-2周）

- [ ] 实现VQA数据加载器
- [ ] 创建完整训练脚本
- [ ] 与真实Qwen2.5-VL集成测试
- [ ] 在小规模数据上验证训练

### 中期（1个月）

- [ ] 大规模VQA训练
- [ ] 评估VQA/Caption/OCR性能
- [ ] Ablation study
- [ ] 性能优化（推理加速）

### 长期（2-3个月）

- [ ] 实现Grid重建Projector (Method B)
- [ ] 多方法对比实验
- [ ] 论文撰写
- [ ] 开源发布

## 🔬 实验笔记

### 测试环境

- **硬件**: CPU测试（GPU版本待测）
- **模型**: Qwen2.5-VL-7B维度
- **Conda环境**: qwen (含transformers)
- **测试数据**: 合成数据

### 关键发现

1. **维度匹配**: Token sorter的query_embeddings需要与vision hidden_size匹配
2. **梯度流**: 所有PATO组件均有稳定梯度
3. **效率提升**: 理论上可达78%计算量减少

### 已知限制

1. 需要transformers库完整集成
2. 配置文件导入依赖问题（已有standalone版本）
3. 未在真实数据上验证性能

## 🤝 贡献

欢迎提Issue和PR！

## 📚 参考文献

- PATO论文: [待填充]
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- SoftSort: https://arxiv.org/abs/1903.08850

## 📄 License

[待定]

---

**Last Updated**: 2025-11-03  
**Version**: 1.0-MVP  
**Status**: ✅ Core components完成，ready for training
