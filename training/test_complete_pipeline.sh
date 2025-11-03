#!/bin/bash
# 完整PATO训练Pipeline测试脚本
# 使用真实图像数据进行小规模训练验证

set -e

echo "========================================"
echo "PATO Complete Pipeline Training Test"
echo "========================================"

# 配置
MODEL_PATH="/data2/youneng/models/Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_DIR="/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/textvqa"
SAVE_DIR="./checkpoints_complete"
MAX_SAMPLES=50  # 小规模测试
BATCH_SIZE=2
MAX_EPOCHS=2
LEARNING_RATE=1e-4

# 检查环境
echo ""
echo "[1/3] Checking environment..."

if [ ! -d "$MODEL_PATH" ]; then
    echo "  ✗ Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "  ✗ Image directory not found: $IMAGE_DIR"
    exit 1
fi

echo "  ✓ Model path: $MODEL_PATH"
echo "  ✓ Image directory: $IMAGE_DIR"
echo "  ✓ Environment ready"

# 运行训练
echo ""
echo "[2/3] Starting complete pipeline training..."
echo "  Samples: $MAX_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $MAX_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo ""

conda run -n qwen python training/train_complete.py \
    --model_path "$MODEL_PATH" \
    --image_dir "$IMAGE_DIR" \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --max_samples $MAX_SAMPLES \
    --learning_rate $LEARNING_RATE \
    --save_dir "$SAVE_DIR" \
    --device cuda:3

# 完成
echo ""
echo "[3/3] Training test completed!"
echo "  Checkpoints: $SAVE_DIR"
echo ""

# 显示结果
if [ -d "$SAVE_DIR" ]; then
    echo "Saved checkpoints:"
    ls -lh "$SAVE_DIR"
fi

echo ""
echo "========================================"
echo "✓ Complete Pipeline Test Passed!"
echo "========================================"
