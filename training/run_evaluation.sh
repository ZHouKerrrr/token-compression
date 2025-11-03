#!/bin/bash
# PATO性能评估脚本

set -e

echo "========================================"
echo "PATO Performance Evaluation"
echo "========================================"

# 配置
CHECKPOINT="./checkpoints_large_scale/best_model.pt"
IMAGE_DIR="/data2/youneng/datas/Visual-CoT/cot_images_tar_split/cot_image_data/textvqa"
BATCH_SIZE=4
NUM_SAMPLES=500
DEVICE="cuda:3"
OUTPUT="evaluation_results_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "Configuration:"
echo "  • Checkpoint: $CHECKPOINT"
echo "  • Image directory: $IMAGE_DIR"
echo "  • Samples: $NUM_SAMPLES"
echo "  • Batch size: $BATCH_SIZE"
echo "  • Device: $DEVICE"
echo "  • Output: $OUTPUT"
echo ""

# 检查checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Please specify a valid checkpoint path."
    exit 1
fi

# 运行评估
echo "Starting evaluation..."
echo ""

conda run -n qwen python training/evaluate_performance.py \
    --checkpoint "$CHECKPOINT" \
    --image_dir "$IMAGE_DIR" \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --device "$DEVICE" \
    --output "$OUTPUT"

echo ""
echo "========================================"
echo "Evaluation Completed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT"
echo ""
echo "View results:"
echo "  cat $OUTPUT | python -m json.tool"
