#!/bin/bash
# PATO Training Script for Qwen2.5-VL
# 
# 这个脚本演示如何在小规模数据上训练PATO组件
#

set -e  # Exit on error

echo "========================================"
echo "PATO-Qwen2.5-VL Training Script"
echo "========================================"

# 配置
MODEL_PATH="/data2/youneng/models/Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="./demo_vqa_data.json"
IMAGE_DIR="./demo_images"  # 需要创建对应的图像
SAVE_DIR="./checkpoints"
MAX_SAMPLES=10  # 快速测试，只使用10个样本

# 训练参数
BATCH_SIZE=2
MAX_EPOCHS=2
LEARNING_RATE=1e-4
TARGET_SIZE="448 448"
TOKEN_BUDGET=256

# 检查环境
echo ""
echo "[1/3] Checking environment..."

if [ ! -f "$DATA_PATH" ]; then
    echo "  Creating demo dataset..."
    python -c "from training.data_loader import create_demo_dataset; create_demo_dataset('$DATA_PATH', 10)"
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "  Creating demo image directory..."
    mkdir -p "$IMAGE_DIR"
    # 创建一些演示图像（简单的彩色图）
    python -c "
from PIL import Image
import os
for i in range(10):
    img = Image.new('RGB', (448, 448), color=(100+i*10, 150, 200-i*10))
    img.save(os.path.join('$IMAGE_DIR', f'demo_image_{i:04d}.jpg'))
"
fi

echo "  ✓ Environment ready"

# 开始训练
echo ""
echo "[2/3] Starting training..."
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH"
echo "  Samples: $MAX_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $MAX_EPOCHS"
echo ""

conda run -n qwen python training/train.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --image_dir "$IMAGE_DIR" \
    --dataset_type custom \
    --max_samples $MAX_SAMPLES \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --target_size $TARGET_SIZE \
    --token_budget $TOKEN_BUDGET \
    --save_dir "$SAVE_DIR" \
    --num_workers 0 \
    --device cuda

# 完成
echo ""
echo "[3/3] Training completed!"
echo "  Checkpoints saved to: $SAVE_DIR"
echo ""
echo "========================================"
echo "✓ Done!"
echo "========================================"
