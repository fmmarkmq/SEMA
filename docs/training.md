# Training Pipeline

SEMA uses a two-stage training pipeline: **Prefilling Self-Tuning** followed by **Reinforcement Learning with Intent-drift-aware Reward**.

All stages are launched through the same entry point, `sema/cli/train_cli.py`, with the stage
name as the first argument (`rollout`, `self_tuning`, or `rl_grpo`).

## Stage I: Prefilling Self-Tuning

This stage has two sub-steps: rollout generation and self-tuning.

### Step 1: Prefilling Rollout Generation

Generates multi-turn attack rollouts using vLLM-accelerated inference.

```bash
python -m sema.cli.train_cli rollout \
    --seed 42 \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --temperature 1.0 \
    --max_tokens 500 \
    --dataset_name advbench \
    --max_turns 7 \
    --n_generations 10 \
    --batch_size 100 \
    --output_dir files/prefilling_rollouts \
    --output_file "llama8b_prefilling_rollout.json"
```

**What happens:**
1. Loads the AdvBench dataset (automatically downloaded on first run to `data/advbench.csv`)
2. Splits it 80/20 into train/test and keeps the train split
3. For each harmful request, renders system + query prompts via Jinja2 templates
4. Prefills the assistant turn with `"1. "` to guide numbered multi-turn generation
5. Generates `n_generations` samples per request
6. Parses the numbered list format → extracts `adv_prompt` turn steps

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name_or_path` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID or local path |
| `--n_generations` | 10 | Rollout samples per request |
| `--max_turns` | 7 | Maximum conversation turns |
| `--temperature` | 1.0 | Sampling temperature |
| `--max_tokens` | 500 | Maximum output tokens per rollout |
| `--batch_size` | 100 | Requests per vLLM batch |
| `--dataset_name` | `advbench` | Source dataset (only `advbench` is supported) |
| `--test_size` | 0.2 | Held-out fraction of the dataset |
| `--data_split_seed` | 42 | Seed for the train/test split |
| `--system_prompt_path` | `sema/prompts/sema_system.jinja` | System prompt template |
| `--query_prompt_path` | `sema/prompts/sema_query.jinja` | Query prompt template |
| `--output_dir` | `files/prefilling_rollouts` | Output directory |
| `--output_file` | `llama8b_prefilling_rollout.json` | Output filename |

### Step 2: Self-Tuning via SFT

Trains the attacker model on prefilling rollouts using supervised fine-tuning.

```bash
accelerate launch --num_processes 4 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/cli/train_cli.py self_tuning \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name json \
    --data_path "files/prefilling_rollouts/llama8b_prefilling_rollout.json" \
    --output_dir files/checkpoints/prefilling_self-tuned-Llama-3.1-8B-Instruct \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --max_length 4096 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 1e-5}'
```

`--dataset_name json` is required: the rollouts are loaded with the HuggingFace `json` dataset
builder, and `--data_path` is passed to it as `data_files`.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | — | Dataset builder; use `json` to read the rollout file |
| `--data_path` | — | Path to the prefilling rollout JSON |
| `--max_length` | 4096 | Maximum sequence length |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--num_train_epochs` | 1 | Training epochs |
| `--gradient_checkpointing` | false | Enable gradient checkpointing |

Beyond these, every argument of TRL's `SFTConfig` and `ModelConfig` is accepted.

## Stage II: Reinforcement Learning

Fine-tunes the self-tuned attacker using GRPO with intent-drift-aware reward against a training-time victim.

### Step 1: Start the attacker vLLM server

GRPO generates its rollouts from a standalone vLLM server hosting the **attacker** checkpoint.
It runs on its own GPU, separate from the trainer ranks:

```bash
CUDA_VISIBLE_DEVICES=7 trl vllm-serve \
    --model files/checkpoints/prefilling_self-tuned-Llama-3.1-8B-Instruct
```

The **victim** is *not* served here — it is loaded inside the trainer processes themselves
(see Step 2).

### Step 2: Run GRPO Training

```bash
accelerate launch --num_processes 7 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/cli/train_cli.py rl_grpo \
    --model_name_or_path files/checkpoints/prefilling_self-tuned-Llama-3.1-8B-Instruct \
    --victim_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir files/checkpoints/SEMA-Llama-3.1-8B-Instruct@Llama8B \
    --dataset_name advbench \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --steps_per_generation 12 \
    --num_generations 28 \
    --learning_rate 1e-5 \
    --score_reward_name ida_reward \
    --score_reward_model_name openai/gpt-4.1-mini-2025-04-14 \
    --use_vllm --vllm_mode server \
    --vllm_server_base_url http://localhost:8000 \
    --reward_log_save_dir files/logs
```

**What happens:**
1. Each trainer rank loads the victim model with vLLM using the `external_launcher` distributed
   backend, so the victim shares the trainer's GPUs (bounded by `--gpu_memory_utilization`)
2. The attacker generates multi-turn attack plans (completions) via the vLLM server from Step 1
3. The victim replays each attack turn by turn → full multi-turn conversations
4. Reward computation, on the victim's **final-turn** response:
   - **Format Reward**: validates the numbered list structure (binary: 0 or 1)
   - **Score Reward**: gpt-4.1-mini evaluates intent alignment, compliance risk, and detail level
   - **Final Reward** = `0.5 × Format Reward + 0.5 × Score Reward`
5. GRPO updates the policy using the reward signal

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--victim_model_name_or_path` | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace model for the victim |
| `--score_reward_name` | `ida_reward` | Reward function (`ida_reward`, `no_intent_alignment_reward`, `no_refusal_reward`) |
| `--score_reward_model_name` | `openai/gpt-4.1-mini-2025-04-14` | LLM API model for scoring |
| `--num_generations` | 28 | Completions sampled per prompt (GRPO group size) |
| `--steps_per_generation` | 12 | Optimizer steps taken per generation batch |
| `--gpu_memory_utilization` | 0.4 | vLLM memory fraction for the victim on each trainer GPU |
| `--max_turns` | 7 | Max conversation turns |
| `--max_tokens` | 500 | Max tokens per victim response |
| `--max_model_len` | 4000 | Max context length for the victim |
| `--dataset_name` | `advbench` | Source dataset (only `advbench` is supported) |
| `--test_size` | 0.2 | Held-out fraction of the dataset |
| `--data_split_seed` | 42 | Seed for the train/test split |
| `--reward_log_save_dir` | `files/logs` | Directory for reward logs |

Beyond these, every argument of TRL's `GRPOConfig` and `ModelConfig` is accepted.

## Notes

- **GPU allocation**: RL training reserves 1 GPU for the `trl vllm-serve` process that generates
  attacker rollouts, and uses the rest for GRPO training. For 8 GPUs: GPU 7 runs the server,
  GPUs 0–6 run training. The victim is co-located with the training ranks, which is why
  `--gpu_memory_utilization` must leave room for the trainer.
- **API key**: The IDA reward uses `gpt-4.1-mini`. Set your API key via the `OPENAI_API_KEY`
  environment variable or a `.env` file.
- **Experiment tracking**: All scripts log to [Weights & Biases](https://wandb.ai). Set
  `--report_to none` to disable.
- **DeepSpeed**: Distributed training uses ZeRO Stage 3
  (`scripts/deepspeed_configs/deepspeed_zero3.yaml`). On memory-constrained hardware, use
  `deepspeed_zero3_cpu_offload.yaml` instead to move optimizer state to CPU.
