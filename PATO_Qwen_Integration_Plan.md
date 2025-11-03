# PATO-Qwen2.5-VL 集成项目计划

## 📋 项目概述

**目标**: 将 PATO (Pixels As The Optimization) 方法集成到 Qwen2.5-VL 多模态大模型中，实现端到端可训练的视觉token优化。

**核心改造**:
1. **g_raw (像素域预压缩)** - 嵌入 Processor 阶段，实现可导的图像预处理
2. **Token Sort (前缀最优排序)** - 嵌入 Vision Encoder 之后，实现基于query的token选择，要求可微分
3. **保留 Patch Merger** - 维持原有的空间合并和维度投影逻辑

---

## 🏗️ 架构设计

### 整体Pipeline (完善版)

```
原始输入 (PIL Image / numpy array)
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 0: Processor (非可导预处理)                       │
├─────────────────────────────────────────────────────────┤
│ 输入: images (PIL/numpy), text (str)                    │
│                                                          │
│ 标准预处理流程:                                          │
│   - Resize to standard size (如 1024×1024)             │
│   - ToTensor: PIL → Tensor [B, 3, H, W]                │
│   - Normalize: ImageNet mean/std                        │
│                                                          │
│ 文本处理:                                                │
│   - Tokenization: text → input_ids                      │
│   - Add special tokens (<image>, <|im_start|> etc.)    │
│                                                          │
│ 输出:                                                    │
│   pixel_values: [B, 3, 1024, 1024] (Tensor, 可导✅)    │
│   input_ids: [B, seq_len]                               │
│   attention_mask: [B, seq_len]                          │
└─────────────────────────────────────────────────────────┘
    ↓ pixel_values [1, 3, 1024, 1024] - Tensor在计算图中
    
╔═════════════════════════════════════════════════════════╗
║              Model Forward (全程可导)                   ║
╠═════════════════════════════════════════════════════════╣
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 1A: 文本条件提取 (用于g_raw)              │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ 提前获取文本嵌入作为g_raw的条件:                │  ║
║  │                                                  │  ║
║  │ # 轻量级文本编码 (可冻结)                       │  ║
║  │ text_embeds = embed_tokens(input_ids)            │  ║
║  │   [B, seq_len, 3584]                            │  ║
║  │                                                  │  ║
║  │ # Pooling获取query embedding                    │  ║
║  │ eq = text_embeds.mean(dim=1)  # [B, 3584]       │  ║
║  │ 或: eq = attention_pool(text_embeds, mask)       │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ eq [B, 3584]                                   ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 1B: g_raw 像素域预压缩 (NEW - 可导✅)     │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ 输入: pixel_values [B, 3, 1024, 1024]           │  ║
║  │       eq (query embed) [B, 3584]                │  ║
║  │                                                  │  ║
║  │ 文本条件投影:                                    │  ║
║  │   text_cond = TextProj(eq)  # [B, 256]          │  ║
║  │   text_cond_spatial = expand to [B,256,H,W]     │  ║
║  │                                                  │  ║
║  │ 显著性预测 (方案A - 加权下采样):                │  ║
║  │   concat_input = cat([pixel_values,             │  ║
║  │                      text_cond_spatial], dim=1) │  ║
║  │     → [B, 3+256=259, 1024, 1024]                │  ║
║  │                                                  │  ║
║  │   saliency_map = SaliencyNet(concat_input)      │  ║
║  │     → [B, 1, 1024, 1024]  范围[0,1]             │  ║
║  │                                                  │  ║
║  │ 加权像素值:                                      │  ║
║  │   weighted = pixel_values * saliency_map        │  ║
║  │     → [B, 3, 1024, 1024]                        │  ║
║  │                                                  │  ║
║  │ 可导下采样 (使用PyTorch内置算子):               │  ║
║  │   compressed = F.adaptive_avg_pool2d(           │  ║
║  │       weighted, (448, 448))                     │  ║
║  │   weight_sum = F.adaptive_avg_pool2d(           │  ║
║  │       saliency_map, (448, 448))                 │  ║
║  │   compressed = compressed / (weight_sum+1e-8)   │  ║
║  │                                                  │  ║
║  │ 输出: pixel_values_compressed                    │  ║
║  │   [B, 3, 448, 448] (浓缩小图,非人眼可视导向)    │  ║
║  │                                                  │  ║
║  │ ✅ 全程Tensor操作,完全可导                       │  ║
║  │ ✅ 参数: SaliencyNet (CNN), TextProj (MLP)      │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ pixel_values_compressed [B, 3, 448, 448]       ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 2: Vision Encoding                        │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ Patch Embedding:                                 │  ║
║  │   Conv3d(3→1152, kernel=14×14, stride=14)       │  ║
║  │   [B, 3, 448, 448] → [B, 1024, 1152]            │  ║
║  │   (32×32 patches)                                │  ║
║  │                                                  │  ║
║  │ Position Encoding:                               │  ║
║  │   - 计算 grid_thw = [1, 32, 32]                 │  ║
║  │   - 生成 3D RoPE (temporal, height, width)      │  ║
║  │   - rotary_pos_emb: (cos, sin) [1024, dim]      │  ║
║  │                                                  │  ║
║  │ Vision Transformer Blocks (×32 layers):         │  ║
║  │   for layer in range(32):                        │  ║
║  │     # Self-Attention (sliding window)            │  ║
║  │     attn_out = Attention(hidden, pos_emb)        │  ║
║  │     hidden = hidden + attn_out                   │  ║
║  │                                                  │  ║
║  │     # MLP                                        │  ║
║  │     mlp_out = MLP(LayerNorm(hidden))            │  ║
║  │     hidden = hidden + mlp_out                    │  ║
║  │                                                  │  ║
║  │ 输出: image_embeds (原始vision tokens)           │  ║
║  │   Zv = [B, N=1024, d_v=1152]                    │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ Zv [B, 1024, 1152]                             ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 3: Token Sort (NEW - PATO核心,可导✅)     │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ 输入: Zv [B, 1024, 1152]                        │  ║
║  │       eq (query embed) [B, 3584] (Stage 1A复用) │  ║
║  │                                                  │  ║
║  │ 特征拼接:                                        │  ║
║  │   eq_expand = eq.unsqueeze(1).expand(B,N,3584)  │  ║
║  │   concat_feat = cat([Zv, eq_expand], dim=-1)    │  ║
║  │     → [B, 1024, 1152+3584=4736]                 │  ║
║  │                                                  │  ║
║  │ 相关性打分:                                      │  ║
║  │   scores = Ranker_phi(concat_feat)               │  ║
║  │   # MLP: 4736 → 512 → 128 → 1                   │  ║
║  │     → [B, 1024, 1] squeeze → [B, 1024]          │  ║
║  │                                                  │  ║
║  │ 可微排序 (方案A - SoftSort):                     │  ║
║  │   # Sinkhorn算法或Gumbel-Softmax                │  ║
║  │   P = SoftPermutation(scores, tau=2.0)           │  ║
║  │     → [B, 1024, 1024] (软置换矩阵)               │  ║
║  │                                                  │  ║
║  │   # 应用排序                                     │  ║
║  │   Z_sorted = bmm(P.transpose(-2,-1), Zv)        │  ║
║  │     → [B, 1024, 1152]                           │  ║
║  │                                                  │  ║
║  │ Budget采样 (训练时随机):                         │  ║
║  │   if training:                                   │  ║
║  │     M = random.randint(128, 512)                │  ║
║  │   else:                                          │  ║
║  │     M = config.default_budget  # 如 256         │  ║
║  │                                                  │  ║
║  │   Z_prefix = Z_sorted[:, :M, :]                 │  ║
║  │   sort_indices = argsort(scores)[:, :M]         │  ║
║  │                                                  │  ║
║  │ 输出: Z_prefix [B, M, 1152] (例: M=256)         │  ║
║  │       sort_indices [B, M] (原始位置索引)         │  ║
║  │                                                  │  ║
║  │ ✅ 全程可导: Ranker可学习, SoftSort可微         │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ Z_prefix [B, 256, 1152]                        ║
║       ↓ sort_indices [B, 256]                          ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 4: Projector / Patch Merger (改造)        │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ [问题] Token Sort后tokens不再是规则32×32 grid   │  ║
║  │                                                  │  ║
║  │ ━━━━━━━━━ Option A (推荐MVP) ━━━━━━━━━━━       │  ║
║  │ 简化线性投影:                                    │  ║
║  │   H_v = Linear_1152_to_3584(Z_prefix)           │  ║
║  │     → [B, 256, 3584]                            │  ║
║  │                                                  │  ║
║  │   优点: 简单,快速验证,保留所有选中tokens        │  ║
║  │   缺点: token数较多,无空间合并                  │  ║
║  │                                                  │  ║
║  │                                                  │  ║
║  │                                                  │  ║
║  │                                                  │  ║
║  │ 输出: H_v (视觉特征,对齐LLM维度)                │  ║
║  │   [B, num_visual_tokens, 3584]                  │  ║
║  │   num_visual_tokens = 256 (A) | 64 (B/C)        │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ H_v [B, num_visual_tokens, 3584]              ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 5: Position IDs 重建                      │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ Qwen2.5-VL使用4D position IDs:                  │  ║
║  │   [text_pos, temporal, height, width]           │  ║
║  │                                                  │  ║
║  │ 文本部分 (保持原有):                             │  ║
║  │   text_pos_ids: [B, seq_len]                    │  ║
║  │     值: [0, 1, 2, ..., seq_len-1]               │  ║
║  │   temporal: 全0                                 │  ║
║  │   height/width: 全0                             │  ║
║  │                                                  │  ║
║  │ 视觉部分 (从sort_indices重建):                  │  ║
║  │   # Option A: 简化 - 使用1D序列位置              │  ║
║  │   vision_pos_ids = torch.arange(num_tokens)     │  ║
║  │   temporal = 0                                  │  ║
║  │   height = vision_pos_ids // sqrt(num_tokens)   │  ║
║  │   width = vision_pos_ids % sqrt(num_tokens)     │  ║
║  │                                                  │  ║
║  │   # Option B: 从sort_indices反推原始2D位置       │  ║
║  │   original_h = sort_indices // 32               │  ║
║  │   original_w = sort_indices % 32                │  ║
║  │   # 归一化到grid size                           │  ║
║  │   height = original_h * grid_h / 32             │  ║
║  │   width = original_w * grid_w / 32              │  ║
║  │                                                  │  ║
║  │ 拼接 position_ids:                               │  ║
║  │   position_ids = stack([text_pos, temporal,     │  ║
║  │                        height, width], dim=0)   │  ║
║  │     → [4, B, total_len]                         │  ║
║  │   total_len = seq_len + num_visual_tokens       │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ position_ids [4, B, total_len]                ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 6: Multimodal Fusion                      │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ 文本嵌入:                                        │  ║
║  │   H_q = embed_tokens(input_ids)                 │  ║
║  │     → [B, seq_len, 3584]                        │  ║
║  │                                                  │  ║
║  │ 找到<image>占位符位置:                           │  ║
║  │   image_token_id = 151655  # <image>特殊token   │  ║
║  │   image_mask = (input_ids == image_token_id)    │  ║
║  │     → [B, seq_len] bool                         │  ║
║  │                                                  │  ║
║  │ 扩展mask到特征维度:                              │  ║
║  │   image_mask_expanded = image_mask.unsqueeze(-1)│  ║
║  │                        .expand(..., 3584)       │  ║
║  │     → [B, seq_len, 3584]                        │  ║
║  │                                                  │  ║
║  │ 插入视觉特征:                                    │  ║
║  │   # 方法1: masked_scatter (Qwen原生)            │  ║
║  │   inputs_embeds = H_q.masked_scatter(           │  ║
║  │       image_mask_expanded, H_v)                 │  ║
║  │                                                  │  ║
║  │   # 方法2: 显式拼接                             │  ║
║  │   pre_image = H_q[:, :image_start_pos, :]       │  ║
║  │   post_image = H_q[:, image_end_pos:, :]        │  ║
║  │   inputs_embeds = cat([pre_image, H_v,          │  ║
║  │                       post_image], dim=1)       │  ║
║  │                                                  │  ║
║  │ 输出: inputs_embeds                              │  ║
║  │   [B, seq_len+num_visual_tokens, 3584]          │  ║
║  │   例: [1, 50+256=306, 3584]                     │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ inputs_embeds [B, total_len, 3584]            ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 7: Language Model (80 layers)             │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ 准备RoPE:                                        │  ║
║  │   rotary_emb = Qwen2_5_VLRotaryEmbedding()      │  ║
║  │   position_embeddings = rotary_emb(             │  ║
║  │       inputs_embeds, position_ids)              │  ║
║  │     → (cos, sin) 用于multimodal RoPE            │  ║
║  │                                                  │  ║
║  │ Decoder Layers (×80):                           │  ║
║  │   hidden_states = inputs_embeds                 │  ║
║  │                                                  │  ║
║  │   for layer_idx in range(80):                   │  ║
║  │     layer = decoder_layers[layer_idx]           │  ║
║  │                                                  │  ║
║  │     # Self-Attention (GQA + Sliding Window)     │  ║
║  │     # num_heads=64, num_kv_heads=8              │  ║
║  │     attn_out = layer.self_attn(                 │  ║
║  │         hidden_states,                          │  ║
║  │         attention_mask=causal_mask,             │  ║
║  │         position_embeddings=position_embeddings,│  ║
║  │         use_sliding_window=(layer_idx not in    │  ║
║  │             config.fullatt_block_indexes)       │  ║
║  │     )                                            │  ║
║  │     hidden_states = hidden_states + attn_out    │  ║
║  │                                                  │  ║
║  │     # MLP                                        │  ║
║  │     mlp_out = layer.mlp(                        │  ║
║  │         layer.post_attention_layernorm(         │  ║
║  │             hidden_states))                     │  ║
║  │     hidden_states = hidden_states + mlp_out     │  ║
║  │                                                  │  ║
║  │ Final LayerNorm:                                │  ║
║  │   hidden_states = norm(hidden_states)           │  ║
║  │                                                  │  ║
║  │ 输出: [B, total_len, 8192]                      │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ hidden_states [B, total_len, 8192]            ║
║                                                         ║
║  ┌─────────────────────────────────────────────────┐  ║
║  │ Stage 8: LM Head                                │  ║
║  ├─────────────────────────────────────────────────┤  ║
║  │ 投影到词表:                                      │  ║
║  │   logits = lm_head(hidden_states)               │  ║
║  │     → [B, total_len, vocab_size=152064]         │  ║
║  │                                                  │  ║
║  │ 训练时: 计算交叉熵损失                          │  ║
║  │   loss = CrossEntropy(logits, labels)           │  ║
║  │                                                  │  ║
║  │ 推理时: 生成下一个token                         │  ║
║  │   next_token_logits = logits[:, -1, :]          │  ║
║  │   next_token = argmax(next_token_logits)        │  ║
║  └─────────────────────────────────────────────────┘  ║
║       ↓ logits [B, total_len, 152064]                 ║
╚═════════════════════════════════════════════════════════╝
    ↓
最终输出: 
  - 训练模式: loss (标量)
  - 推理模式: generated_ids [B, gen_len]
```

---



### 3. **维度变化完整追踪**
| Stage | 输入 | 操作 | 输出 |
|-------|------|------|------|
| 0 | PIL Image | Processor | `[B,3,1024,1024]` |
| 1A | `input_ids` | Text Embed | `[B,3584]` |
| 1B | `[B,3,1024,1024]`+`[B,3584]` | g_raw | `[B,3,448,448]` |
| 2 | `[B,3,448,448]` | Vision Encoder | `[B,1024,1152]` |
| 3 | `[B,1024,1152]`+`[B,3584]` | Token Sort | `[B,M,1152]` |
| 4 | `[B,M,1152]` | Projector | `[B,M,3584]` |
| 5 | - | Position IDs | `[4,B,seq+M]` |
| 6 | `[B,seq,3584]`+`[B,M,3584]` | Fusion | `[B,seq+M,3584]` |
| 7 | `[B,seq+M,3584]` | LLM | `[B,seq+M,8192]` |
| 8 | `[B,seq+M,8192]` | LM Head | `[B,seq+M,152064]` |

### 4. **三个Projector方案对比**
| 方案 | 输入→输出 | Token压缩 | 空间结构 | 复杂度 | 推荐阶段 |
|------|----------|----------|----------|--------|---------|
| A | `[256,1152]→[256,3584]` | 无 | 丢失 | ★ | MVP |
| B | `[256,1152]→[64,3584]` | 4:1 | 保留 | ★★★ | V2.0 |
| C | `[256,1152]→[64,3584]` | 4:1 | 部分 | ★★ | 可选 |





## 📁 项目文件结构

```
PATO/
├── README.md                          # 项目说明
├── PATO.md                           # PATO方法论文档 (已有)
├── PATO_Qwen_Integration_Plan.md    # 本文档
│
├── configuration_PATO_qwen2_5_vl.py       # 配置文件 (可在原qwen2_5基础上修改)
├── modeling_PATO_qwen2_5_vl.py            # 模型主文件 (可在原qwen2_5基础上修改)
├── modular_PATO_qwen2_5_vl.py            # 模块化模型 (可在原qwen2_5基础上修改)
├── processing_PATO_qwen2_5_vl.py         # 处理器文件 (可在原qwen2_5基础上修改)
│
├── g_raw/                            # g_raw模块 (已有)
│   ├── __init__.py
│   ├── weighted_downsample.py       # 加权下采样实现(已有)
│   └── base.py                       # 基础类 (待创建)
│
├── token_sort/                       # Token Sort模块 (已有)
│   ├── __init__.py
│   ├── gatingsort.py                # Gating-based排序
│   ├── softsort.py                  # Soft排序实现
│   ├── multi-distillation.py        # 多目标蒸馏
│   └── ranker.py                     # Ranker网络 (待创建)
│
├── pato_integration/                 # 集成代码 (新建)
│   ├── __init__.py
│   ├── pato_processor.py            # 扩展的Processor
│   ├── pato_model.py                # 扩展的Model
│   ├── pato_config.py               # PATO配置类
│   └── loss.py                      # 损失函数
│
├── training/                         # 训练脚本 (新建)
│   ├── train.py                     # 主训练脚本
│   ├── trainer.py                   # Trainer类
│   ├── data_loader.py               # 数据加载
│   └── utils.py                     # 训练工具
│
├── evaluation/                       # 评估脚本 (新建)
│   ├── eval.py                      # 主评估脚本
│   ├── metrics.py                   # 评估指标
│   └── benchmarks/                  # 基准测试
│       ├── vqa.py                   # VQA评估
│       ├── caption.py               # Caption评估
│       └── ocr.py                   # OCR评估
│
├── experiments/                      # 实验配置 (新建)
│   ├── configs/                     # 实验配置文件
│   │   ├── baseline.yaml            # 基线配置
│   │   ├── pato_stage1.yaml         # 阶段1配置
│   │   └── pato_full.yaml           # 完整配置
│   └── scripts/                     # 运行脚本
│       ├── run_baseline.sh
│       ├── run_pato.sh
│       └── ablation.sh
│
└── tests/                            # 单元测试 (新建)
    ├── test_g_raw.py
    ├── test_token_sort.py
    ├── test_integration.py
    └── test_pipeline.py
```

---

## 🎯 实施阶段


### Phase 1: g_raw集成

**目标**: 实现可导的像素域预压缩模块

#### 1.1 g_raw模块重构

**任务**:
- [ ] 设计g_raw基础接口
  ```python
  class GRawBase(nn.Module):
      def forward(images, text_embeds) -> compressed_images
  ```

- [ ] 实现g_raw方案
  - **方案A**: Adaptive (加权下采样) - 重点


- [ ] 可微分实现
  - 所有操作使用PyTorch可导算子
  - 避免硬编码索引和离散操作
  - 实现梯度回传路径

**关键设计点**:
- 输入: `[B, C, H, W]` 原始图像 + `[B, D]` 文本嵌入
- 输出: `[B, C, H*, W*]` 压缩图像 (H*×W* < H×W)
- 条件化: 文本嵌入通过attention或gating调制

#### 1.2 Processor集成

**任务**:
- [ ] 创建 `Qwen2_5_VLProcessorPATO` 类
- [ ] 在 `__call__` 方法中插入g_raw
- [ ] 处理PIL/numpy/Tensor格式转换
- [ ] 实现文本编码器接口

**代码位置**:
- 文件: `pato_integration/pato_processor.py`
- 继承: `Qwen2_5_VLProcessor`

**接口设计**:
```python
processor = Qwen2_5_VLProcessorPATO.from_pretrained(
    "Qwen/Qwen2.5-VL-7B",
    g_raw_config={
        'mode': 'A',              # I/C/F/A/B
        'target_size': (448, 448),
        'enable': True,
        'learnable': True,
    }
)
```

#### 1.3 测试与验证

**任务**:
- [ ] 单元测试: 测试各g_raw方案的输入输出
- [ ] 梯度测试: 验证反向传播正确性
- [ ] 性能测试: 对比压缩前后的推理速度
- [ ] 质量测试: 评估压缩后的VQA性能

**输出**:
- `tests/test_g_raw.py`
- g_raw性能报告
- 可视化对比 (原图 vs 压缩图)

---

### Phase 2: Token Sort集成

**目标**: 实现基于query的可微token排序

#### 2.1 Token Sort模块实现

**任务**:
- [ ] 设计Ranker网络
  - 输入: vision_tokens + query_embeds
  - 输出: relevance scores
  - 架构: MLP with LayerNorm

- [ ] 实现可微排序
  - **方案A**: SoftSort (Sinkhorn/Gumbel)
  - **方案B**: Gating-based (Hard gating + Sort)
  - **方案C**: Multi-distillation (多目标)

- [ ] Budget采样策略
  - 训练时: 随机采样 M ~ Uniform[M_min, M_max]
  - 推理时: 固定budget或自适应

**关键设计点**:
- 输入: `[B, N, d_v]` vision tokens + `[B, d_h]` query embedding
- 输出: `[B, M, d_v]` sorted tokens + `[B, M]` sort indices
- 温度参数: τ = 2.0 (可学习)
- 正则化: 熵正则 + 多样性损失

#### 2.2 Model集成

**任务**:
- [ ] 创建 `Qwen2_5_VLModelPATO` 类
- [ ] 在Vision Encoder后插入Token Sort
- [ ] 修改forward流程处理sorted tokens
- [ ] 处理position_ids重建问题

**代码位置**:
- 文件: `pato_integration/pato_model.py`
- 继承: `Qwen2_5_VLModel`

**关键实现**:
```python
# 在forward中
if self.enable_token_sort:
    sorted_tokens, sort_indices = self.token_sorter(
        vision_tokens=image_embeds,
        query_embeds=query_embeds,
        budget=token_budget,
    )
    image_embeds = sorted_tokens
```

#### 2.3 Patch Merger适配

**任务**:
- [ ] **方案A** (推荐起步): 简化线性投影
  - 实现: `nn.Linear(1152, 3584)`
  - 优点: 简单,保留所有选中tokens
  - 缺点: token数可能较多


**实施建议**:
- 第一阶段: 使用方案A快速验证

#### 2.4 Position IDs处理

**任务**:
- [ ] 理解Qwen2.5-VL的3D position IDs
- [ ] 设计sorted tokens的position重建策略
- [ ] 实现 `_get_position_ids_with_sort()` 方法

**挑战**:
- 原始: 规则grid [t, h, w] 对应每个patch
- 排序后: 打乱顺序,需从sort_indices反推
- 解决: 保存原始位置映射

#### 2.5 测试与验证

**任务**:
- [ ] 单元测试: Token Sort模块独立测试
- [ ] 集成测试: 完整forward流程测试
- [ ] 梯度测试: 验证端到端可导

**输出**:
- `tests/test_token_sort.py`


---

### Phase 3: 配置与损失函数

**目标**: 完善配置系统和训练损失

#### 3.1 配置系统扩展

**任务**:
- [ ] 扩展 `Qwen2_5_VLConfig`
  ```python
  class Qwen2_5_VLConfig:
      def __init__(self, ..., pato_config=None):
          self.pato_config = pato_config or {}
  ```

- [ ] 创建 `PATOConfig` 类
  ```python
  @dataclass
  class PATOConfig:
      # g_raw配置
      g_raw_mode: str = 'A'
      g_raw_target_size: Tuple[int, int] = (448, 448)
      g_raw_enable: bool = True
      
      # Token Sort配置
      token_sort_mode: str = 'A'
      token_budget_min: int = 128
      token_budget_max: int = 512
      token_sort_enable: bool = True
      
      # Merger配置
      use_simplified_merger: bool = True
      
      # 训练配置
      lambda_contrast: float = 0.1
      lambda_distill: float = 0.05
      lambda_sort_reg: float = 0.01
  ```

**代码位置**:
- 文件: `pato_integration/pato_config.py`

#### 3.2 损失函数设计

**任务**:
- [ ] 主损失: 语言建模损失 (保持原有)

- [ ] 特征蒸馏损失
  - 目的: 桥接g_raw输出与标准下采样
  - 实现: MSE between g_raw features and baseline features

- [ ] Token Sort正则
  - 熵正则: 鼓励明确排序
  - 多样性损失: 减少冗余token

- [ ] Budget正则 (可选)
  - 控制token使用量

**代码位置**:
- 文件: `pato_integration/loss.py`

**核心公式**:
```
L_total = L_LM + λ_c·L_contrast + λ_d·L_distill + λ_s·L_sort_reg

L_contrast = max(0, margin - ||f(I, q1) - f(I, q2)||²)
L_distill = MSE(E(g_raw(I, q)), E(resize(I)))
L_sort_reg = λ_ent·H(scores) + λ_div·Sim(Z_sorted)
```

#### 3.3 训练工具

**任务**:
- [ ] 创建 `PATOTrainer` 类
  - 继承 `transformers.Trainer`
  - 重写 `compute_loss` 方法
  - 实现budget采样逻辑

- [ ] 实现数据加载器
  - 支持VQA数据集
  - 实现query-swap数据增强
  - Batch collator

**代码位置**:
- 文件: `training/trainer.py`
- 文件: `training/data_loader.py`

---

### Phase 4: 训练与调试

**目标**: 端到端训练PATO-Qwen模型

#### 4.1 训练流程设计

**阶段1: 冻结预训练,只训练PATO组件** 
- 冻结: Vision Encoder, LLM
- 训练: g_raw, Token Sort, Simplified Projector
- 数据: VQA数据集 (10k-50k样本)
- Epochs: 3-5
- 目标: 验证PATO组件有效性

**阶段2: 端到端微调** 
- 解冻: 全部参数
- LoRA/QLoRA: LLM部分使用低秩适配
- 数据: 扩展到100k+样本
- Epochs: 5-10
- 目标: 达到或超越baseline性能

#### 4.2 超参数设置





### Phase 5: 评估与优化

**目标**: 全面评估PATO-Qwen性能

#### 5.1 评估基准

**任务**:
- [ ] VQA任务
  - VQAv2
  - TextVQA
  - DocVQA

- [ ] Image Captioning
  - COCO Caption
  - NoCaps

- [ ] OCR任务
  - 场景文本识别
  - 文档理解

**指标**:
- 准确率 (Accuracy)
- BLEU / CIDEr (Caption)
- 推理速度 (tokens/sec)
- 显存占用 (GB)

#### 5.2 Ablation Study

**任务**:
- [ ] g_raw消融
  - 无g_raw (baseline)
  - g_raw方案I/C/F/A/B对比
  - 不同target_size影响

- [ ] Token Sort消融
  - 无Token Sort
  - 方案A/B/C对比
  - 不同budget影响

- [ ] 损失函数消融
  - 去除各辅助损失
  - 调整损失权重

**输出**:
- Ablation结果表
- 各组件贡献分析

#### 5.3 性能优化

**任务**:
- [ ] 推理加速
  - 优化Token Sort计算
  - 使用torch.compile()
  - Flash Attention 2

- [ ] 显存优化
  - Gradient checkpointing
  - 量化 (INT8/FP8)

- [ ] 部署优化
  - ONNX导出
  - TensorRT转换

---

### Phase 6: 文档与开源 

**目标**: 整理文档,准备开源

#### 6.1 文档编写

**任务**:
- [ ] README.md
  - 项目介绍
  - 安装说明
  - 快速开始
  - 性能对比

- [ ] API文档
  - g_raw模块API
  - Token Sort模块API
  - 使用示例

- [ ] 训练指南
  - 数据准备
  - 训练命令
  - 超参数说明

- [ ] 模型卡片
  - 模型描述
  - 性能指标
  - 限制说明

#### 6.2 代码清理

**任务**:
- [ ] 代码规范检查
  - PEP8风格
  - Type hints
  - Docstrings

- [ ] 删除调试代码
- [ ] 统一命名规范
- [ ] 添加LICENSE

#### 6.3 模型发布

**任务**:
- [ ] 上传到Hugging Face Hub
  - 模型权重
  - 配置文件
  - Tokenizer

- [ ] 创建Demo
  - Gradio界面
  - Colab notebook

---

## 📊 关键指标与里程碑

### 技术指标

| 指标 | Baseline (Qwen原生) | 目标 (PATO-Qwen) |
|------|---------------------|------------------|
| **VQAv2 Accuracy** | 82.5% | ≥82.0% |
| **Visual Tokens** | 256 | 128-256 (可调) |
| **Inference Speed** | 1.0x | 1.2-1.5x |
| **GPU Memory** | 16GB | ≤16GB |
| **FLOPs** | 100% | 70-85% |


---

## 🚧 风险与挑战

### 技术风险

1. **g_raw可导性问题**
   - 风险: 某些压缩操作难以可导
   - 缓解: 使用可导近似(双线性插值,soft gating)

2. **Token Sort性能损失**
   - 风险: 排序可能丢失重要视觉信息
   - 缓解: 多轮实验调优budget,使用蒸馏损失

3. **Position IDs重建困难**
   - 风险: 排序后破坏空间结构
   - 缓解: 方案A简化投影,或方案B grid重建


### 资源风险

1. **计算资源不足**
   - 需要: 8×A100/H100 GPU
   - 缓解: 使用QLoRA,减少batch size

2. **数据集获取**
   - 需要: VQA等标注数据
   - 缓解: 使用公开数据集

---

## 🔄 迭代策略

### V1.0 (MVP - 最小可行产品)
- 只实现g_raw方案A和Token Sort方案A
- 使用简化Projector (方案A)
- 冻结预训练模型训练PATO组件
- 在单个VQA数据集上验证

### V2.0 (完整版)
- 实现所有g_raw和Token Sort方案
- 实现Grid重建的Patch Merger
- 端到端微调
- 多数据集评估

### V3.0 (优化版)
- 推理加速优化
- 量化支持
- 部署工具
- 更多下游任务支持
