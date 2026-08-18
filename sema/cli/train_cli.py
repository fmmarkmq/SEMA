# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unified CLI entry point for SEMA training pipeline.

Usage:
    python -m sema.cli.train_cli rollout     [args]
    python -m sema.cli.train_cli self_tuning [args]
    python -m sema.cli.train_cli rl_grpo     [args]

Under `accelerate launch`, invoke this file by path instead:
    accelerate launch [...] sema/cli/train_cli.py self_tuning [args]
"""

import os
import sys
import argparse

from sema.utils import setup_seed


def cmd_rollout(argv):
    """Prefilling rollout stage."""
    from sema.trainer.trainer import run_rollout

    parser = argparse.ArgumentParser(prog="sema train rollout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--dataset_name", type=str, default="advbench")
    parser.add_argument("--data_split_seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_turns", type=int, default=7)
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--system_prompt_path", type=str, default="sema/prompts/sema_system.jinja")
    parser.add_argument("--query_prompt_path", type=str, default="sema/prompts/sema_query.jinja")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="files/prefilling_rollouts")
    parser.add_argument("--output_file", type=str, default="llama8b_prefilling_rollout.json")

    args = parser.parse_args(argv)
    setup_seed(args.seed)
    run_rollout(args)


def cmd_self_tuning(argv):
    """Self-tuning (SFT) stage."""
    from trl import ModelConfig, SFTConfig, TrlParser
    from sema.trainer.trainer import SelfTuningScriptArguments, run_self_tuning

    os.environ["WANDB_PROJECT"] = "SEMA"
    sys.argv = [sys.argv[0]] + argv
    parser = TrlParser((SelfTuningScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)

    setup_seed(training_args.seed)
    run_self_tuning(script_args, training_args, model_args)


def cmd_rl_grpo(argv):
    """RL-GRPO training stage."""
    from trl import GRPOConfig, ModelConfig, TrlParser
    from sema.trainer.trainer import DataConfig, RewardConfig, run_rl_grpo

    os.environ["WANDB_PROJECT"] = "SEMA"
    sys.argv = [sys.argv[0]] + argv
    parser = TrlParser((DataConfig, GRPOConfig, ModelConfig, RewardConfig))
    data_args, training_args, model_args, reward_args = parser.parse_args_and_config()

    setup_seed(training_args.seed)
    run_rl_grpo(data_args, training_args, model_args, reward_args)


COMMANDS = {
    "rollout": cmd_rollout,
    "self_tuning": cmd_self_tuning,
    "rl_grpo": cmd_rl_grpo,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sema train",
        description="SEMA training pipeline",
    )
    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
        help="Training stage to run: rollout | self_tuning | rl_grpo",
    )

    args, remaining = parser.parse_known_args()
    COMMANDS[args.command](remaining)
