ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

export LMMS_EVAL_PLUGINS="my_lmms_eval"

base_model=${BASE_MODEL:-'Qwen/Qwen2.5-VL-7B-Instruct'}
base_model_suffix='spare'
base_output_path=${BASE_OUTPUT_PATH:-"result/${base_model_suffix}/lmms_eval"}
port=${PORT:-29500}
attn_implementation=${ATTN_IMPL:-"flash_attention_2"}

config="train_configs/qwen2_5_7b_spare/qwen2_5_7b_spare.yaml"
spare_state_dict_path="output/qwen2_5_7b_spare/spare_components.pt"

eval_list=( \
"vqav2_val_lite" \
)

for task in ${eval_list[@]}
do
    output_path=${base_output_path}/${task}

    # if [ -d "$output_path" ]; then
    #     echo "Output path $output_path already exists. Skipping evaluation for task: $task"
    #     continue
    # fi

    echo "Evaluating task: $task"
    accelerate launch --num_processes=$ngpus --main_process_port=$port -m lmms_eval \
        --model spare_qwen2_5_vl \
        --model_args=pretrained=${base_model},attn_implementation=${attn_implementation},spare_state_dict_path=${spare_state_dict_path} \
        --tasks $task \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples
done
