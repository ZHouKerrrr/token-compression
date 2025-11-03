# PATO Large-Scale Training Guide

## 📊 可用数据集

已准备好的VQA数据集（位于 `/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/`）：

| 数据集 | 样本数 | 标注文件 |
|--------|--------|----------|
| TextVQA | 18,524 | `textvqa_cot_train.jsonl` |
| GQA | 98,149 | `gqa_cot_train_brief_alpaca.jsonl` |
| DocVQA | 33,453 | `docvqa_cot_train.jsonl` |
| Flickr30k | 135,735 | `flickr30k_cot_train.jsonl` |
| InfographicsVQA | 15,055 | `infographicsvqa_cot_train.jsonl` |
| OpenImages | 43,053 | `openimages_cot_train.jsonl` |
| Visual7W | 30,491 | `visual7w_cot_train.jsonl` |
| **总计** | **532,414** | - |

## 🚀 快速开始

### 1. 测试数据加载

```bash
conda run -n qwen python tests/test_data_loader.py
```

### 2. 小规模训练测试（100样本）

```bash
python training/train_large_scale.py \
    --image_dir /data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data \
    --annotation_file /data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/textvqa_cot_train.jsonl \
    --max_samples 100 \
    --batch_size 8 \
    --max_epochs 5 \
    --gpus "3" \
    --save_dir ./checkpoints_test
```

### 3. 大规模多GPU训练（10k样本）

```bash
# 使用预配置脚本
./training/run_large_scale_training.sh

# 或者手动指定参数
python training/train_large_scale.py \
    --image_dir /data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data \
    --annotation_file /data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/textvqa_cot_train.jsonl \
    --max_samples 10000 \
    --batch_size 8 \
    --max_epochs 20 \
    --gpus "3,1,6" \
    --num_workers 8 \
    --save_dir ./checkpoints_large_scale
```

### 4. 性能评估

```bash
# 使用预配置脚本
./training/run_evaluation.sh

# 或者手动指定
python training/evaluate_performance.py \
    --checkpoint ./checkpoints_large_scale/best_model.pt \
    --image_dir /data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data \
    --batch_size 4 \
    --num_samples 500 \
    --device cuda:3 \
    --output evaluation_results.json
```

## 📝 训练参数说明

### 数据参数
- `--image_dir`: 图像目录路径
- `--annotation_file`: JSONL标注文件路径
- `--max_samples`: 最大样本数（None=使用全部）

### 训练参数
- `--batch_size`: 批次大小（默认8）
- `--max_epochs`: 训练轮数（默认20）
- `--learning_rate`: 学习率（默认1e-4）
- `--weight_decay`: 权重衰减（默认0.01）

### 模型参数
- `--text_dim`: 文本维度（默认3584）
- `--vision_tokens`: Vision tokens数量（默认256）
- `--token_budget`: Token预算/压缩后数量（默认128，50%压缩率）

### GPU参数
- `--gpus`: GPU列表（如 "3" 或 "3,1,6"）
- `--num_workers`: DataLoader工作进程数（默认4）

### 输出参数
- `--save_dir`: checkpoint保存目录
- `--log_dir`: TensorBoard日志目录

## 📈 监控训练

### TensorBoard (如果已安装)

```bash
tensorboard --logdir=./logs --port=6006
```

然后访问 `http://localhost:6006`

### 查看日志

```bash
# 实时查看最新日志
tail -f logs/training_*.log

# 查看所有日志
ls -lt logs/
```

## 🎯 训练策略建议

### 快速验证（100-1k样本）
```bash
--max_samples 1000
--batch_size 8
--max_epochs 10
--gpus "3"
```

### 中等规模（10k样本）
```bash
--max_samples 10000
--batch_size 8
--max_epochs 20
--gpus "3,1,6"
```

### 大规模训练（50k-100k样本）
```bash
--max_samples 100000
--batch_size 16
--max_epochs 30
--gpus "3,1,6,7"
--num_workers 16
```

### 全量训练（532k样本）
```bash
--max_samples None  # 使用所有数据
--batch_size 32
--max_epochs 50
--gpus "3,1,6,7,8,9"
--num_workers 32
```

## 🔧 可用GPU

通过 `nvidia-smi` 查看可用GPU：

- cuda:0 ❌ (占用42GB)
- cuda:1 ✅ (可用)
- cuda:2 ❌ (占用38GB)
- cuda:3 ✅ (推荐，默认)
- cuda:6 ✅ (可用)
- cuda:7 ✅ (可用)
- cuda:8 ✅ (可用)
- cuda:9 ✅ (可用)

## 📊 性能评估指标

评估脚本将输出：

1. **重建质量**
   - MSE（均方误差）
   - PSNR（峰值信噪比）

2. **推理速度**
   - g_raw处理时间
   - Token sorting时间
   - 总延迟
   - 吞吐量（samples/sec）

3. **内存使用**
   - 不同batch size下的显存占用
   - Peak memory

4. **Token压缩**
   - 原始token数
   - 压缩后token数
   - 压缩比率

## 🐛 故障排除

### OOM (Out of Memory)
- 减小 `--batch_size`
- 使用更多GPU: `--gpus "3,1,6"`
- 减少 `--vision_tokens`

### 数据加载慢
- 增加 `--num_workers`
- 使用SSD存储
- 减少数据增强

### 训练不稳定
- 降低学习率: `--learning_rate 5e-5`
- 增加batch size
- 检查数据质量

## 📂 输出文件结构

```
PATO/
├── checkpoints_large_scale/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   ├── ...
│   └── best_model.pt
├── logs/
│   ├── training_20251103_*.log
│   └── events.out.tfevents.*
└── evaluation_results_*.json
```

## ✅ 检查清单

训练前确认：

- [ ] 数据集路径正确
- [ ] 标注文件存在
- [ ] GPU可用且空闲
- [ ] 有足够的磁盘空间（checkpoint）
- [ ] conda环境激活 (`conda activate qwen`)

## 🎓 下一步

1. ✅ 数据加载器测试通过
2. ⏳ 运行小规模训练验证（100样本）
3. ⏳ 运行中等规模训练（10k样本）
4. ⏳ 性能评估和对比
5. ⏳ 大规模训练（50k-532k样本）
6. ⏳ 论文结果和可视化

## 📧 联系

如有问题，请查看：
- 训练日志: `logs/training_*.log`
- Error traceback
- GPU使用: `nvidia-smi`
