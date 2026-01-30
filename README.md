# SEMA
The official repository for the paper: *SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks*.

**2025/1/25 Update:** Our paper has been accepted to ICLR 2026 main conference!

## Notes

**Our full codes are currently undergoing review by Microsoft Research. We will release them as soon as possible.**

# Overview

We train a open-loop, response-agnostic multi-turn jailbreak attacker with two stages: *Prefiiling Self-Tuning* and *Reinforcement Learning with Intent-drift-aware Reward*.

# Quick Start

## Installation

```bash
conda create -n sema python=3.11.13 -y
conda activate sema
pip install -r requirements.txt
```

## Stage I: Prefiiling Self-Tuning

Check `scripts` for all provided scripts. For example, the following script runs *Prefilling Self-Tuning* on *meta-llama/Llama-3.1-8B-Instruct* using 4xH100 gpus.

```bash
bash prefill_selftuning_llama8b_4gpu(80gb).sh
```

After that, a set of prefilling rollouts and a self-tuned model are supposed to be saved into `files/prefilling_rollouts` and `files/checkpoints`, respectively.

## Stage II: Reinforcement Learning with Intent-drift-aware Reward

Check `scripts` for all provided scripts. For example, the following script runs *Reinforcement Learning with Intent-drift-aware Reward* on the self-tuned *meta-llama/Llama-3.1-8B-Instruct* (attacker model) against *meta-llama/Llama-3.1-8B-Instruct* (training-time victim) using 8xH100 gpus.

```bash
bash rl_grpo_llama8b@llama8b_8gpu(80gb).sh
```

After that, a learned attacker model and reward logs are supposed to be saved into `files/checkpoints` and `files/logs`
