# Configuration

## Hardware Requirements

| Configuration | GPUs | Use Case |
|---------------|------|----------|
| 4×H100 (80 GB) | 4 | Prefilling self-tuning (Llama-8B) |
| 8×H100 (80 GB) | 8 | Prefilling self-tuning (Qwen-14B) |
| 4×A6000 (48 GB) | 4 | Prefilling self-tuning (Qwen-3B) |
| 8×H100 (80 GB) | 8 | RL training (all model sizes) |
| 4×H100 (80 GB) | 4 | RL training with a smaller victim (ablations) |
| 4×A6000 (48 GB) | 4 | RL training, Qwen-3B attacker (needs ZeRO-3 CPU offload) |

RL training reserves 1 GPU for the `trl vllm-serve` process that generates attacker rollouts.
The remaining GPUs run GRPO training, and each of them also hosts a copy of the victim model
(bounded by `gpu_memory_utilization`).

## Key Hyperparameters

### Prefilling Rollout

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_generations` | 10 | Rollout samples per harmful request |
| `max_turns` | 7 | Maximum conversation turns |
| `temperature` | 1.0 | Sampling temperature for diversity |
| `max_tokens` | 500 | Maximum output tokens per rollout |
| `batch_size` | 100 | Requests per vLLM inference batch |

### Self-Tuning (SFT)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_length` | 4096 | Maximum sequence length |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `num_train_epochs` | 1 | Training epochs |
| `warmup_ratio` | 0.03 | Warmup proportion |
| `lr_scheduler_type` | `cosine_with_min_lr` | LR schedule |
| `torch_dtype` | `bfloat16` | Training precision |

### RL (GRPO)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-5 | GRPO learning rate |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 6 | Gradient accumulation |
| `num_generations` | 28 | Completions sampled per prompt (GRPO group size) |
| `steps_per_generation` | 12 | Optimizer steps per generation batch |
| `max_prompt_length` | 2048 | Maximum prompt length |
| `max_completion_length` | 500 | Maximum completion length |
| `gpu_memory_utilization` | 0.4 | vLLM memory fraction for the victim |

## DeepSpeed Configuration

Distributed training uses DeepSpeed ZeRO Stage 3. Two configs ship in
`scripts/deepspeed_configs/`:

| Config | When to use |
|--------|-------------|
| `deepspeed_zero3.yaml` | Default. Parameter, gradient, and optimizer partitioning across ranks. |
| `deepspeed_zero3_cpu_offload.yaml` | Memory-constrained GPUs. Additionally offloads optimizer state to CPU, at the cost of throughput. |

Both enable 16-bit model saving and are compatible with gradient checkpointing.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for IDA reward scoring via `gpt-4.1-mini` |
| `X_API_KEY` | Optional. If set, its value is forwarded as an `X-API-Key` header on judge API calls (for gateways that require it) |
| `WANDB_API_KEY` | Optional, for experiment tracking |
| `CUDA_VISIBLE_DEVICES` | Used to allocate specific GPUs (e.g. the vLLM server) |

You can set these in a `.env` file in the project root (loaded via `python-dotenv`).

## Dependencies

`requirements.txt` is pinned to match the published training image, `allenlao/sema:v0.1`
(Python 3.12.3, CUDA 12.6) — these are the versions the experiments actually ran on.

| Package | Version | Purpose |
|---------|---------|---------|
| `trl` | 0.20.0 | SFTTrainer, GRPOTrainer |
| `vllm` | 0.10.0 | Fast LLM inference |
| `litellm[caching]` | 1.74.8 | Unified LLM API with disk caching |
| `deepspeed` | 0.17.4 | ZeRO-3 distributed optimization |
| `wandb` | 0.28.1 | Experiment tracking |
| `kernels` | 0.9.0 | Optimized kernel operations |
| `python-dotenv` | 1.1.0 | Environment variable management |

The underlying framework versions are pinned too, so that bumping any package above fails at
install time instead of silently upgrading the training stack:

| Package | Version |
|---------|---------|
| `torch` | 2.7.1 |
| `transformers` | 4.57.6 |
| `accelerate` | 1.12.0 |
| `datasets` | 4.5.0 |

`peft==0.19.1` is listed but commented out — it is only needed for LoRA runs (`--use_peft`).
With the default `use_peft=False`, `trl.get_peft_config` returns `None` without importing it.

> **Upgrading:** verify any dependency bump against the image before merging it. Notably,
> `vllm >= 0.11` pulls in torch 2.11, transformers 5.x, and CUDA 13 wheels, none of which are
> compatible with the pinned `trl==0.20.0` or the image's CUDA 12.6 runtime.
