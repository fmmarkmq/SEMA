#!/bin/bash
set -e

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
else
    echo "ERROR: .env not found at repo root"; exit 1
fi

SERVER_LOGFILE="rl_grpo_server_qwen3b@llama8b_8gpu.log"
CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct > "$SERVER_LOGFILE" 2>&1 &
VLLM_PID=$!
echo "Starting VLLM server with PID: $VLLM_PID ..."
echo "Wait 60s ..."
sleep 60


accelerate launch --num_processes 7 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/cli/train_cli.py rl_grpo \
    --seed 42 \
    --data_seed 42 \
    --model_name_or_path files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct \
    --output_dir files/checkpoints/SEMA-Qwen2.5-3B-Instruct@Llama8B \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --steps_per_generation 12 \
    --num_generations 28 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 500 \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_base_url http://localhost:8000 \
    --save_strategy no \
    --logging_steps 1 \
    --log_completions \
    --report_to wandb \
    --run_name rl_grpo_qwen3b@llama8b_8gpu \
    --dataset_name advbench \
    --data_split_seed 42 \
    --test_size 0.2 \
    --max_turns 7 \
    --victim_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --gpu_memory_utilization 0.4 \
    --score_reward_name ida_reward \
    --score_reward_model_name openai/gpt-4.1-mini-2025-04-14 \
    --reward_log_save_dir files/logs