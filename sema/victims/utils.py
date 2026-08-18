# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ray
from vllm import LLM

@ray.remote(num_gpus=1)
class LLMWorker:
    def __init__(self, **kwargs):
        self.model = LLM(**kwargs)

    def generate(self, prompts, sampling_params):
        return self.model.generate(prompts, sampling_params)

    def chat(self, messages, sampling_params):
        return self.model.chat(messages, sampling_params)
    
    def get_tokenizer(self):
        return self.model.get_tokenizer()


class RayLLM:
    def __init__(self, **kwargs):
        self.model = LLMWorker.remote(**kwargs)
    
    async def generate(self, prompts, sampling_params):
        return await self.model.generate.remote(prompts, sampling_params)

    async def chat(self, messages, sampling_params):
        return await self.model.chat.remote(messages, sampling_params)
    
    def get_tokenizer(self):
        return ray.get(self.model.get_tokenizer.remote())
