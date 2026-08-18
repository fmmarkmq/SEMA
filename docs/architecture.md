# Architecture

## Project Structure

```
sema/
├── cli/
│   └── train_cli.py            # Unified CLI entry point (rollout | self_tuning | rl_grpo)
├── trainer/
│   └── trainer.py              # All three training stages + the RL reward pipeline
├── utils.py                    # Shared utilities (seeding, async completions)
├── prompts/                    # Jinja2 prompt templates
│   ├── sema_system.jinja       # System prompt for multi-turn attack generation
│   ├── sema_query.jinja        # Query prompt template
│   ├── ida_reward_system.jinja # IDA reward scoring rubric
│   └── no_intent_alignment_reward_system.jinja  # Ablation variant
├── rewards/                    # Reward functions for RL training
│   ├── format_reward.py        # Numbered list format checking
│   ├── intent_drift_aware_reward.py  # Main IDA reward (via gpt-4.1-mini)
│   ├── utils.py                # JSON parsing helpers
│   └── ablations/              # Ablation reward variants
│       ├── no_intent_alignment_reward.py
│       └── no_refusal_reward.py
├── victims/                    # Victim model interfaces
│   ├── victims.py              # VLLMVictimWrapper for multi-turn simulation
│   └── utils.py                # RayLLM distributed wrapper
└── baselines/                  # Baseline methods for comparison
    ├── sft.py                  # SFT baseline
    ├── dpo.py                  # DPO baseline
    └── rollout.py              # Ray-distributed rollout with victim execution
```

## Module Overview

### Core Training Modules

All three stages are dispatched through a single entry point, `sema/cli/train_cli.py`, which
parses stage-specific arguments and calls the corresponding function in `sema/trainer/trainer.py`:

```bash
python -m sema.cli.train_cli rollout      [args]   # → run_rollout()
python -m sema.cli.train_cli self_tuning  [args]   # → run_self_tuning()
python -m sema.cli.train_cli rl_grpo      [args]   # → run_rl_grpo()
```

Under `accelerate launch`, the same file is invoked by path instead of as a module:
`accelerate launch ... sema/cli/train_cli.py self_tuning [args]`.

| Function (in `trainer/trainer.py`) | Role |
|--------|------|
| `run_rollout()` | Generates multi-turn attack rollouts using vLLM. Takes AdvBench harmful requests + prompt templates, outputs numbered multi-turn jailbreak attempts (up to 7 turns). Prefills the assistant turn with `"1. "` to bootstrap attack patterns. |
| `run_self_tuning()` | SFT stage. Trains the attacker on prefilling rollouts using TRL's `SFTTrainer` with DeepSpeed ZeRO-3. |
| `run_rl_grpo()` | Main RL loop. Instantiates a co-located vLLM victim, then runs GRPO training with format + intent-drift-aware rewards. |
| `SimulationRewardPipeline` | Glue class used as the GRPO `reward_funcs`: parses completions, runs the victim conversations, scores them, combines rewards, and writes reward logs. |
| `parse_adv_prompt()` | Parses a numbered-list completion into a list of per-turn attack prompts. |
| `utils.py` | `setup_seed()`, `set_deterministic()`, `get_completions()` for async API calls with litellm disk caching. |

### Reward System

| Component | Function |
|-----------|----------|
| `format_reward.py` | `NumberedListFormatReward` — checks if the completion follows `1. ... 2. ... 3. ...` format up to `max_turns`. Returns 1 if valid, 0 otherwise. Also parses turn items for victim consumption. |
| `intent_drift_aware_reward.py` | `IntentDriftAwareReward` — main reward. Calls gpt-4.1-mini to score three dimensions: Intent Alignment (0–10), Compliance Risk (0–10), Detail Level (0–10). Each score is normalized to 0–1, then combined as `Intent Alignment × (Compliance + Detail) / 2`. |
| `ablations/no_intent_alignment_reward.py` | `NoIntentAlignmentReward` — removes the intent alignment component (ablation baseline). |
| `ablations/no_refusal_reward.py` | `NoRefusalPhraseReward` — refusal detector matching 17 refusal phrases. |

### Victim Interface

| Component | Function |
|-----------|----------|
| `victims.py` | `VLLMVictimWrapper` — wraps vLLM for multi-turn conversation simulation. `execute()` runs one conversation (a single prompt, or a list of prompts played turn by turn), `execute_batch()` runs many conversations in parallel with per-conversation message history tracking. |
| `utils.py` | `RayLLM` — optional Ray-based wrapper that runs the victim in a dedicated single-GPU Ray actor. |

## Data Flow

```
AdvBench (520 harmful behaviors, 80/20 train/test split)
        │
        ▼
┌───────────────────────────────┐
│  Stage I: Prefilling          │
│  train_cli.py rollout         │
│  → 10 rollouts per request    │
│  → numbered multi-turn format │
└──────────┬────────────────────┘
           │  files/prefilling_rollouts/*.json
           ▼
┌───────────────────────────────┐
│  Stage I: Self-Tuning         │
│  train_cli.py self_tuning     │
│  → SFT on rollouts            │
│  → DeepSpeed ZeRO-3           │
└──────────┬────────────────────┘
           │  files/checkpoints/prefilling_self-tuned-*/
           ▼
┌───────────────────────────────┐
│  Stage II: RL (GRPO)          │
│  train_cli.py rl_grpo         │
│  → co-located vLLM victim     │
│  → format + IDA rewards       │
│  → gpt-4.1-mini scoring       │
└──────────┬────────────────────┘
           │
           ▼
  files/checkpoints/SEMA-*     (final attacker)
  files/logs/reward_logs_*.json (reward tracking)
```

Note that Stage II also needs a separate `trl vllm-serve` process, started by the shell scripts
in `scripts/`. That server hosts the **attacker** for GRPO rollout generation; the **victim** is
loaded inside the trainer ranks themselves (see [Training Pipeline](training.md)).

## Output Structure

```
files/
├── prefilling_rollouts/
│   └── <model>_prefilling_rollout.json
│       → List[{dataset_name, sample_id, request, attacker,
│                adv_prompt[], attempt_id, input_prompt, output_text}]
├── checkpoints/
│   ├── prefilling_self-tuned-<Model>/      # Stage I output
│   └── SEMA-<AttackerModel>@<VictimModel>/ # Stage II output
└── logs/
    └── reward_logs_rank*.json
        → List[{step, rank, timestamp, num_completions, requests,
                 completions, adv_prompts, format_rewards, score_rewards,
                 final_rewards, responses, last_response}]
```

One reward log file is written per training rank, and it is rewritten in full after every
reward computation.
