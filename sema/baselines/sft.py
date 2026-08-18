# Copyright (c) Microsoft Corporation.
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "kernels",
# ]
# ///

"""
pip install --upgrade kernels

Example:

accelerate launch \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/baselines/sft.py \
    --torch_dtype bfloat16 \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --run_name sft-derefusal-llama3-8b \
    --dataset_name json \
    --data_path files/prefill_rollout/advbench - asdw_llama3_8b_original_adv_prompt.json \
    --gradient_checkpointing \
    --max_length 4096 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 1e-5}' \
    --output_dir models/sft-derefusal-llama3-8b \
    --report_to wandb \
    --seed 42
"""
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class ScriptArguments_(ScriptArguments):
    data_path: str = field(default=None)
    min_ida_reward: float = field(
        default=0.9,
        metadata={
            "help": "Keep only rollouts whose IDA reward (evaluation.reward, in [0, 1]) is at least this value."
        },
    )


def main(script_args, training_args, model_args):
    # Load model & tokenizer
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = load_dataset(
        script_args.dataset_name,
        data_files=script_args.data_path,
        split="train",
        num_proc=16,
    )

    def make_prompt_completion(example):
        prompt_tokens = tokenizer.apply_chat_template(example['input_prompt'], add_generation_prompt=True)
        prompt = tokenizer.decode(prompt_tokens)
        completion = example['output_text']  
        return {"prompt": prompt, "completion": completion}
    
    dataset = dataset.map(make_prompt_completion)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    dataset = dataset.filter(
        lambda x: float(x['evaluation']['reward']) >= script_args.min_ida_reward
    )

    # Train model
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(model_name=training_args.output_dir.split("/")[-1])


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "SEMA"
    parser = TrlParser((ScriptArguments_, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)