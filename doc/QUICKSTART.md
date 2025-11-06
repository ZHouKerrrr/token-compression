# PATO-Qwen2.5-VL 快速开始指南

## 🎯 5分钟快速体验

### 1. 环境准备

```bash
# 激活qwen环境
conda activate qwen

# 进入项目目录
cd /home/baoyouneng/LLM_Compression/PATO
```

### 2. 运行测试

```bash
# 一键运行所有测试
./run_tests.sh
```

预期输出：
```
✓ ALL TESTS PASSED!
PATO V1.0 MVP is ready for training!
```

### 3. 理解结果

测试会展示：
- ✅ 组件独立验证
- ✅ 完整pipeline模拟
- ✅ 梯度流验证
- ✅ **78%效率提升**

---

## 📚 深入了解

### 查看核心组件

```bash
# G_Raw (像素压缩)
cat g_raw/weighted_downsample.py

# Token Sort (token选择)
cat token_sort/softsort.py

# PATO Loss
cat pato_integration/loss.py
```

### 阅读文档

- `README_V1_MVP.md` - 完整项目文档
- `SUMMARY.md` - 完成总结
- `PATO.md` - 方法论文档
- `PATO_Qwen_Integration_Plan.md` - 集成计划

---

## 🔬 自定义测试

### 修改配置

编辑 `tests/test_pato_demo.py`:

```python
# 修改压缩尺寸
pato_config.g_raw.target_size = (224, 224)  # 更小

# 修改token budget
pato_config.token_sort.budgets = [128]  # 更少tokens

# 修改batch size
batch_size = 4  # 更大batch
```

### 查看中间输出

在代码中添加：

```python
# 查看压缩后的图像
import matplotlib.pyplot as plt
plt.imshow(compressed_images[0].permute(1, 2, 0).cpu().numpy())

# 查看token分数
print("Token scores:", aux_outputs['scores'])

# 查看排序索引
print("Sort indices:", aux_outputs['sort_indices'])
```

---

## 🚀 下一步

### 方案A: 快速训练（推荐）

```bash
# 1. 准备小型VQA数据集 (~1k samples)
python scripts/prepare_mini_vqa.py  # [待实现]

# 2. 运行快速训练
python training/train.py \
    --model_path /data2/youneng/models/Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset mini_vqa \
    --epochs 3 \
    --batch_size 4 \
    --output_dir outputs/pato_v1_quick
```

### 方案B: 完整训练

```bash
# 1. 下载VQAv2数据集
python scripts/download_vqa.py  # [待实现]

# 2. 完整训练
python training/train.py \
    --model_path /data2/youneng/models/Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset vqav2 \
    --epochs 5 \
    --batch_size 8 \
    --output_dir outputs/pato_v1_full
```

---

## 🐛 问题排查

### 常见问题

**Q: ImportError: attempted relative import**
```bash
A: 使用提供的测试脚本，已处理导入问题
   ./run_tests.sh
```

**Q: 维度不匹配错误**
```bash
A: 检查配置文件中的维度设置
   - g_raw.text_dim = 3584
   - projector.vision_dim = 1152
   - projector.hidden_dim = 3584
```

**Q: CUDA out of memory**
```bash
A: 减小batch_size或使用gradient checkpointing
   pato_config.use_gradient_checkpointing = True
```

### 获取帮助

1. 查看测试输出
2. 阅读SUMMARY.md
3. 检查代码注释

---

## 📊 性能监控

### 训练时监控

```python
# 在训练循环中
print(f"Epoch {epoch}:")
print(f"  LM Loss: {lm_loss:.4f}")
print(f"  Distill Loss: {distill_loss:.4f}")
print(f"  Sort Loss: {sort_loss:.4f}")
print(f"  Total Loss: {total_loss:.4f}")
print(f"  Tokens used: {aux_outputs['num_tokens_after']}/{aux_outputs['num_tokens_before']}")
```

### 效率统计

```python
# 计算实际加速
baseline_time = time_without_pato
pato_time = time_with_pato
speedup = baseline_time / pato_time
print(f"Speedup: {speedup:.2f}×")
```

---

## 🎓 代码示例

### 最小使用示例

```python
import torch
from g_raw import WeightedDownsample
from token_sort import DifferentiableSortingTokenSorter

# 初始化
g_raw = WeightedDownsample(config, context)
token_sorter = DifferentiableSortingTokenSorter(config, context)

# 使用
images = torch.randn(1, 3, 1024, 1024)
text_query = torch.randn(1, 3584)

# 压缩
compressed = g_raw(images, text_query)

# 排序（假设已有vision tokens）
vision_tokens = torch.randn(1, 1024, 1152)
selected, indices, aux = token_sorter(
    vision_tokens, 
    query_embeddings=torch.randn(1, 1152),
    budget=256
)

print(f"Compressed: {images.shape} → {compressed.shape}")
print(f"Selected: {vision_tokens.shape[1]} → {selected.shape[1]} tokens")
```

---

## 📈 预期结果

### 测试输出

```
PATO V1.0 MVP Summary:
  • g_raw: ✓ Conditional pixel compression (5.4× reduction)
  • Token Sort: ✓ Query-based token selection (75% reduction)
  • Simplified Projector: ✓ Linear projection
  • Gradient Flow: ✓ End-to-end trainable
  • Efficiency: ~78% overall reduction
```

### 性能指标

- Pixel压缩: **80.9%** ↓
- Token压缩: **75.0%** ↓
- 总体效率: **~78%** ↑

---

## 🎯 检查清单

训练前确认：

- [ ] 所有测试通过 (`./run_tests.sh`)
- [ ] GPU可用且内存充足
- [ ] 数据集已准备
- [ ] 配置文件已检查
- [ ] 输出目录已创建

---

**就这么简单！现在你已经准备好使用PATO了！** 🚀

有问题？查看 `SUMMARY.md` 获取详细信息。
