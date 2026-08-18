# Examples

Pre-built training scripts for different attacker/victim configurations. All scripts are in `scripts/`.

## Prefilling Self-Tuning (Stage I)

| Script | Attacker Model | GPUs |
|--------|---------------|------|
| [`prefill_selftuning_llama8b_4x80gb-gpu.sh`](../scripts/prefill_selftuning_llama8b_4x80gb-gpu.sh) | Llama-3.1-8B-Instruct | 4×H100 |
| [`prefill_selftuning_qwen14b_8x80gb-gpu.sh`](../scripts/prefill_selftuning_qwen14b_8x80gb-gpu.sh) | Qwen2.5-14B-Instruct | 8×H100 |
| [`prefill_selftuning_qwen3b_4x48gb-gpu.sh`](../scripts/prefill_selftuning_qwen3b_4x48gb-gpu.sh) | Qwen2.5-3B-Instruct | 4×A6000 |

**Usage:**

```bash
# Llama-3.1-8B-Instruct on 4 H100s
bash scripts/prefill_selftuning_llama8b_4x80gb-gpu.sh

# Qwen2.5-14B-Instruct on 8 H100s
bash scripts/prefill_selftuning_qwen14b_8x80gb-gpu.sh

# Qwen2.5-3B-Instruct on 4 A6000s
bash scripts/prefill_selftuning_qwen3b_4x48gb-gpu.sh
```

**Outputs:**
- Prefilling rollouts → `files/prefilling_rollouts/`
- Self-tuned checkpoint → `files/checkpoints/prefilling_self-tuned-<Model>/`

## RL with Intent-drift-aware Reward (Stage II)

| Script | Attacker (after Stage I) | Training-time Victim | GPUs |
|--------|--------------------------|---------------------|------|
| [`rl_ida_llama8b@llama8b_8x80gb-gpu.sh`](../scripts/rl_ida_llama8b@llama8b_8x80gb-gpu.sh) | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct | 8×H100 |
| [`rl_ida_qwen14b@llama8_8x80gb-gpu.sh`](../scripts/rl_ida_qwen14b@llama8_8x80gb-gpu.sh) | Qwen2.5-14B-Instruct | Llama-3.1-8B-Instruct | 8×H100 |
| [`rl_ida_qwen3b@llama8b_8x80gb-gpu.sh`](../scripts/rl_ida_qwen3b@llama8b_8x80gb-gpu.sh) | Qwen2.5-3B-Instruct | Llama-3.1-8B-Instruct | 8×H100 |
| [`rl_ida_qwen3b@llama8b_4x46gb-gpu.sh`](../scripts/rl_ida_qwen3b@llama8b_4x46gb-gpu.sh) | Qwen2.5-3B-Instruct | Llama-3.1-8B-Instruct | 4×A6000 |

**Usage:**

```bash
# Llama-8B attacker vs Llama-3.1-8B-Instruct victim
bash scripts/rl_ida_llama8b@llama8b_8x80gb-gpu.sh

# Qwen-14B attacker vs Llama-3.1-8B-Instruct victim
bash scripts/rl_ida_qwen14b@llama8_8x80gb-gpu.sh

# Qwen-3B attacker vs Llama-3.1-8B-Instruct victim
bash scripts/rl_ida_qwen3b@llama8b_8x80gb-gpu.sh
```

`rl_ida_qwen3b@llama8b_4x46gb-gpu.sh` is a smaller-hardware port of the 8×H100 script. It keeps
the paper's effective batch sizes by rebalancing `per_device_train_batch_size`,
`gradient_accumulation_steps`, and `steps_per_generation` across 3 trainer ranks instead of 7,
and it uses `deepspeed_zero3_cpu_offload.yaml` so the optimizer state moves to CPU. Unlike the
other scripts, it polls the vLLM server's `/health` endpoint instead of sleeping for a fixed 60s,
and it tears the server down when training exits.

**Outputs:**
- Final attacker → `files/checkpoints/SEMA-<Attacker>@<Victim>/`
- Reward logs → `files/logs/`

## Ablation Studies

Scripts in `scripts/ablations/` for component-level analysis:

| Script | Description | GPUs |
|--------|-------------|------|
| [`default_rl_ida_qwen3b@llama3b_4x80gb-gpu.sh`](../scripts/ablations/default_rl_ida_qwen3b@llama3b_4x80gb-gpu.sh) | Full SEMA (baseline) | 4×H100 |
| [`rl_no_intent_alignment_reward_qwen3b@llama3b_4x80gb-gpu.sh`](../scripts/ablations/rl_no_intent_alignment_reward_qwen3b@llama3b_4x80gb-gpu.sh) | Without intent alignment reward | 4×H100 |
| [`rl_no_prefilling_self-tuning_qwen3b@llama3b_4x80gb-gpu.sh`](../scripts/ablations/rl_no_prefilling_self-tuning_qwen3b@llama3b_4x80gb-gpu.sh) | Without prefilling self-tuning | 4×H100 |
| [`rl_no-refusal_as_reward_qwen3b@llama3b_4x80gb-gpu.sh`](../scripts/ablations/rl_no-refusal_as_reward_qwen3b@llama3b_4x80gb-gpu.sh) | Refusal detection as reward | 4×H100 |

**Usage:**

```bash
# Full SEMA baseline
bash scripts/ablations/default_rl_ida_qwen3b@llama3b_4x80gb-gpu.sh

# Ablation: no intent alignment in reward
bash scripts/ablations/rl_no_intent_alignment_reward_qwen3b@llama3b_4x80gb-gpu.sh

# Ablation: skip prefilling self-tuning stage
bash scripts/ablations/rl_no_prefilling_self-tuning_qwen3b@llama3b_4x80gb-gpu.sh

# Ablation: simple refusal detection instead of IDA reward
bash scripts/ablations/rl_no-refusal_as_reward_qwen3b@llama3b_4x80gb-gpu.sh
```

## Utilities

| Script | Description |
|--------|-------------|
| [`socket_watchdog.sh`](../scripts/socket_watchdog.sh) | Optional guard for long RL runs. Polls the in-use TCP socket count and sends `SIGINT` to the GRPO trainer ranks before ephemeral port exhaustion can starve the IDA judge calls, giving the trainer a chance to save first. |

```bash
# Run alongside an RL job; tail it via files/logs/socket_watchdog.log
bash scripts/socket_watchdog.sh &
```

Thresholds are configurable via `SOCKET_WATCHDOG_THRESHOLD` (default 25000) and
`SOCKET_WATCHDOG_POLL` (default 60 seconds).

## DeepSpeed Configs

| Config | Use |
|--------|-----|
| [`deepspeed_zero3.yaml`](../scripts/deepspeed_configs/deepspeed_zero3.yaml) | Default ZeRO-3 config used by all 80 GB scripts |
| [`deepspeed_zero3_cpu_offload.yaml`](../scripts/deepspeed_configs/deepspeed_zero3_cpu_offload.yaml) | ZeRO-3 with optimizer state offloaded to CPU, for memory-constrained GPUs |

## Custom Configuration

To train with a different model or configuration, see [docs/training.md](../docs/training.md) for the full argument reference and [docs/configuration.md](../docs/configuration.md) for hardware requirements and hyperparameters.
