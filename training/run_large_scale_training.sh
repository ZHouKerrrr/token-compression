#!/bin/bash
# PATO大规模训练脚本
# 支持多GPU并行训练

set -e

echo "========================================"
echo "PATO Large-Scale Training"
echo "========================================"

# 配置
IMAGE_DIR="/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data"
ANNOTATION_FILE="/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/textvqa_cot_train.jsonl"
SAVE_DIR="./checkpoints_large_scale"
LOG_DIR="./logs"

# 训练参数
MAX_SAMPLES=10000  # 使用10000个样本进行大规模训练
BATCH_SIZE=8
MAX_EPOCHS=20
LEARNING_RATE=1e-4
NUM_WORKERS=8

# GPU设置 (使用多个空闲GPU)
GPUS="3,1,6"  # cuda:3, cuda:1, cuda:6

# Token设置
TEXT_DIM=3584
VISION_TOKENS=256
TOKEN_BUDGET=128

echo ""
echo "Configuration:"
echo "  • Image directory: $IMAGE_DIR"
echo "  • Annotation file: $ANNOTATION_FILE"
echo "  • Max samples: $MAX_SAMPLES"
echo "  • Batch size: $BATCH_SIZE"
echo "  • Epochs: $MAX_EPOCHS"
echo "  • GPUs: $GPUS"
echo "  • Token budget: $TOKEN_BUDGET (from $VISION_TOKENS)"
echo "  • Save directory: $SAVE_DIR"
echo ""

# 检查目录和文件
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    exit 1
fi

if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "Error: Annotation file not found: $ANNOTATION_FILE"
    exit 1
fi

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# 运行训练
echo "Starting training..."
echo ""

conda run -n qwen python training/train_large_scale.py \
    --image_dir "$IMAGE_DIR" \
    --annotation_file "$ANNOTATION_FILE" \
    --max_samples $MAX_SAMPLES \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --text_dim $TEXT_DIM \
    --vision_tokens $VISION_TOKENS \
    --token_budget $TOKEN_BUDGET \
    --gpus "$GPUS" \
    --num_workers $NUM_WORKERS \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "========================================"
echo "Training Completed!"
echo "========================================"
echo ""
echo "Checkpoints: $SAVE_DIR"
echo "Logs: $LOG_DIR"
echo ""
echo "View training progress:"
echo "  tensorboard --logdir=$LOG_DIR"
