#!/bin/bash
set -e

python -m sema.cli.train_cli rollout \
    --seed 42 \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --temperature 1.0 \
    --max_tokens 500 \
    --dataset_name advbench \
    --data_split_seed 42 \
    --test_size 0.2 \
    --max_turns 7 \
    --n_generations 10 \
    --system_prompt_path sema/prompts/sema_system.jinja \
    --query_prompt_path sema/prompts/sema_query.jinja \
    --batch_size 100 \
    --output_dir files/prefilling_rollouts \
    --output_file "qwen3b_prefilling_rollout.json"


accelerate launch --num_processes 4 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/cli/train_cli.py self_tuning \
    --torch_dtype bfloat16 \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --output_dir files/checkpoints/prefilling_self-tuned-Qwen2.5-3B-Instruct \
    --dataset_name json \
    --data_path "files/prefilling_rollouts/qwen3b_prefilling_rollout.json" \
    --gradient_checkpointing \
    --max_length 4096 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --logging_steps 1 \
    --save_strategy no \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --run_name prefilling_self-tuning_qwen3b \
    --seed 42