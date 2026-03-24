ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

port=${PORT:-12345}


configs=( \
"train_configs/qwen2_5_3b_pato/qwen2_5_3b_pato.yaml" \
# "train_configs/qwen2_5_7b_pato/qwen2_5_7b_pato.yaml" \
)


output_dirs=( \
"output/qwen2_5_3b_pato" \
#"output/qwen2_5_7b_pato" \
)

base_models=( \
"Qwen/Qwen2.5-VL-3B-Instruct" \
#"Qwen/Qwen2.5-VL-7B-Instruct" \
)



for i in "${!configs[@]}"; do
    config=${configs[$i]}
    output_dir=${output_dirs[$i]}
    base_model=${base_models[$i]}
    echo "Running with config: $config"

    accelerate launch \
        --mixed_precision="bf16" \
        --num_processes $ngpus \
        --main_process_port $port \
        test_tokens_importance.py \
        --config "$config"

    train_ok=$?
    if [ $train_ok -ne 0 ]; then
        echo "Training failed for config: $config"
        continue
    fi

    # BASE_MODEL=$base_model bash scripts/infer_qwen_pato_cot.sh $output_dir
    # BASE_MODEL=$base_model DO_GLIMPSE=1 bash scripts/infer_qwen_pato_cot.sh $output_dir
    # BASE_MODEL=$base_model MAX_REMAIN_RATIO=0.111 bash scripts/infer_qwen_pato_cot.sh $output_dir
    # BASE_MODEL=$base_model DO_GLIMPSE=1 MAX_REMAIN_RATIO=0.111 bash scripts/infer_qwen_pato_cot.sh $output_dir
    # BASE_MODEL=$base_model bash scripts/eval_qwen_pato.sh $output_dir
done


