#!/bin/bash
# PATO V1.0 MVP 快速测试脚本

echo "============================================================"
echo "PATO-Qwen2.5-VL V1.0 MVP - Quick Test"
echo "============================================================"

# 设置环境
export PYTHONPATH="/home/baoyouneng/LLM_Compression/PATO:$PYTHONPATH"

echo -e "\n[1/2] Testing PATO Components..."
python tests/test_components.py

if [ $? -eq 0 ]; then
    echo -e "\n✓ Component tests passed!"
else
    echo -e "\n✗ Component tests failed!"
    exit 1
fi

echo -e "\n[2/2] Running Integration Demo..."
conda run -n qwen python tests/test_pato_demo.py

if [ $? -eq 0 ]; then
    echo -e "\n============================================================"
    echo "✓ ALL TESTS PASSED!"
    echo "============================================================"
    echo -e "\nPATO V1.0 MVP is ready for:"
    echo "  • Real model integration"
    echo "  • VQA dataset training"
    echo "  • Benchmark evaluation"
else
    echo -e "\n✗ Integration demo failed!"
    exit 1
fi
