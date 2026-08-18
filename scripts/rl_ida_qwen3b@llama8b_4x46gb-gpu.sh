#!/bin/bash
# Stage II — RL with Intent-drift-aware Reward, adapted for 4 x RTX A6000 (46 GiB each).
#
# Hardware mapping vs the published 8x80GB script:
#   GPU 0    : trl vllm-serve (attacker rollouts for GRPO)
#   GPU 1-3  : accelerate launch --num_processes 3 (trainer + co-located vLLM victim
#              via vllm distributed_executor_backend="external_launcher")
#
# Hyperparameter rebalancing (preserves paper math):
#   per_device_train_batch_size  : 4 -> 2     (lower fits 3 trainer ranks vs paper's 7)
#   gradient_accumulation_steps  : 6 -> 28    (preserves global batch    = 2 * 3 * 28 = 168)
#   steps_per_generation         : 12 -> 56   (preserves generation batch = 2 * 3 * 56 = 336)
#   unique prompts per gen cycle : 12         (= 336 / num_generations 28, unchanged from paper)
#
# Unchanged from paper (per user instruction):
#   num_generations=28, max_prompt_length=2048, max_completion_length=500,
#   learning_rate=1e-5, score_reward_name=ida_reward
#
# DeepSpeed:
#   Use deepspeed_zero3_cpu_offload.yaml (NOT the original deepspeed_zero3.yaml) —
#   moves Adam state (~9 GiB/rank) to CPU so the trainer's transient logits-fwd peak
#   (~3.74 GiB) fits alongside the co-located Llama-8B vLLM victim. Approved by the
#   user on 2026-05-06 after analytical OOM accounting; original Rule "don't change
#   DS setup" was waived for this hardware deviation. The arXiv:2604.23747 silent
#   grad-accum bug is the known risk; integrity must be checked via reward curve.
#
# Substituted because the SEMA OpenAI gateway uses undated deployment names:
#   score_reward_model_name : openai/gpt-4.1-mini-2025-04-14 -> openai/gpt-4.1-mini

set -e

# Source the OpenAI gateway secrets into the environment (used by the IDA reward).
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
else
    echo "ERROR: .env not found at repo root"; exit 1
fi

ATTACKER_CKPT="files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct"
SERVER_LOGFILE="files/logs/rl_grpo_server_qwen3b@llama8b_4x46gb.log"
mkdir -p files/logs

if [ ! -d "$ATTACKER_CKPT" ]; then
    echo "ERROR: Stage I checkpoint not found at $ATTACKER_CKPT — run Stage I first."
    exit 1
fi

# --- Start the attacker vLLM server on GPU 0 ---
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model "$ATTACKER_CKPT" > "$SERVER_LOGFILE" 2>&1 &
VLLM_PID=$!
echo "Starting vLLM attacker server on GPU 0 with PID: $VLLM_PID ..."

# Wait for the server to be ready by polling its health endpoint.
echo "Waiting for vLLM server to come up at http://localhost:8000 ..."
for i in $(seq 1 90); do
    if curl -sSf http://localhost:8000/health >/dev/null 2>&1; then
        echo "vLLM server is healthy after ${i}s."
        break
    fi
    sleep 1
done
if ! curl -sSf http://localhost:8000/health >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become healthy in 90s — see $SERVER_LOGFILE"
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

# --- Launch trainer on GPUs 1,2,3 (3 ranks) ---
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3_cpu_offload.yaml \
    sema/cli/train_cli.py rl_grpo \
    --seed 42 \
    --data_seed 42 \
    --model_name_or_path "$ATTACKER_CKPT" \
    --output_dir files/checkpoints/SEMA-Qwen2.5-3B-Instruct@Llama8B \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 28 \
    --steps_per_generation 56 \
    --num_generations 28 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 500 \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_base_url http://localhost:8000 \
    --save_strategy steps \
    --save_steps 10 \
    --logging_steps 1 \
    --log_completions \
    --report_to wandb \
    --run_name rl_grpo_qwen3b@llama8b_4x46gb \
    --dataset_name advbench \
    --data_split_seed 42 \
    --test_size 0.2 \
    --max_turns 7 \
    --victim_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --gpu_memory_utilization 0.55 \
    --score_reward_name ida_reward \
    --score_reward_model_name openai/gpt-4.1-mini \
    --reward_log_save_dir files/logs

# Tear down the vLLM server when training exits.
echo "Trainer finished — stopping vLLM server (PID $VLLM_PID)."
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
