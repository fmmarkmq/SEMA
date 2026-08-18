#!/bin/bash
set -e

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
else
    echo "ERROR: .env not found at repo root"; exit 1
fi

SERVER_LOGFILE="rl_grpo_server_qwen3b@llama3b_4gpu.log"
CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct > "$SERVER_LOGFILE" 2>&1 &
VLLM_PID=$!
echo "Starting VLLM server with PID: $VLLM_PID ..."
echo "Wait 60s ..."
sleep 60


accelerate launch --num_processes 3 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/cli/train_cli.py rl_grpo \
    --seed 42 \
    --data_seed 42 \
    --model_name_or_path files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct \
    --output_dir files/checkpoints/SEMA-Qwen2.5-3B-Instruct@Llama3B \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 8 \
    --num_generations 8 \
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
    --run_name rl_grpo_qwen3b@llama3b_4gpu \
    --dataset_name advbench \
    --data_split_seed 42 \
    --test_size 0.2 \
    --max_turns 7 \
    --victim_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --gpu_memory_utilization 0.4 \
    --score_reward_name no_refusal_reward \
    --score_reward_model_name openai/gpt-4.1-mini-2025-04-14 \
    --reward_log_save_dir files/logs