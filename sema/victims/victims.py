# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
from vllm import LLM, SamplingParams

from .utils import RayLLM


class VLLMVictimWrapper:
    def __init__(
        self, 
        *,
        model_path: str,
        system_prompt: str = None,
        temperature: float = 1.0,
        max_tokens: int = 500,
        use_ray: bool = False,
        _name: str = None,
        **kwargs
        ):
        self._name = _name
        self.model_path = model_path
        self.use_ray = use_ray
        if use_ray:
            self.model = RayLLM(model=model_path, **kwargs)
        else:
            self.model = LLM(model=model_path, **kwargs)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.tokenizer = self.model.get_tokenizer()

    @property
    def victim_repr(self) -> str:
        if self._name is not None:
            return self._name
        else:
            return f"{self.model_path}"
    
    async def execute(
            self, 
            adv_prompt: str | List[str], 
            chat_history: List[dict] | None = None,
        ) -> str | List[str]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if chat_history:
            messages.extend(chat_history)
        
        sampling_params = SamplingParams(max_tokens=self.max_tokens, n=1, temperature=self.temperature)
        if isinstance(adv_prompt, str):
            messages.append({"role": "user", "content": adv_prompt})
            outputs = self.model.chat(messages, sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
            return response
        else:
            responses = []
            for adv_prompt_i in adv_prompt:
                messages.append({"role": "user", "content": adv_prompt_i})
                outputs = self.model.chat(messages, sampling_params=sampling_params)
                response = outputs[0].outputs[0].text
                messages.append({"role": "assistant", "content": response})
                responses.append(response)
            return responses
    
    async def execute_batch(
            self, 
            adv_prompts: List[str | List[str]],
            chats_history: List[List[dict]] | None = None,
        ) -> List[str | List[str]]:
        num_adv_prompts = len(adv_prompts)
        basic_messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        messages_batch = [basic_messages.copy() for _ in range(num_adv_prompts)]
        if chats_history:
            for i, chat_history in enumerate(chats_history):
                if chat_history:
                    messages_batch[i].extend(chat_history)
        
        assert all(isinstance(adv_prompt, str) for adv_prompt in adv_prompts) \
            or all(isinstance(adv_prompt, list) for adv_prompt in adv_prompts), \
            "adv_prompts must be all str or all List[str]"
        
        
        sampling_params = SamplingParams(max_tokens=self.max_tokens, n=1, temperature=self.temperature)
        if isinstance(adv_prompts[0], str):
            for i, adv_prompt in enumerate(adv_prompts):
                messages_batch[i].append({"role": "user", "content": adv_prompt})
            outputs = self.model.chat(messages_batch, sampling_params=sampling_params)
            response_batch = [output.outputs[0].text for output in outputs]
            return response_batch
        else:
            responses_batch = [[] for _ in range(num_adv_prompts)]
            
            max_turns = max(len(adv_prompt) for adv_prompt in adv_prompts)
            for turn in range(max_turns):
                active_chat_ids = []
                active_chat_messages = []
                for i, adv_prompt in enumerate(adv_prompts):
                    if turn < len(adv_prompt):
                        active_chat_ids.append(i)
                        messages_batch[i].append({"role": "user", "content": adv_prompt[turn]})
                        active_chat_messages.append(messages_batch[i])
                outputs = self.model.chat(active_chat_messages, sampling_params=sampling_params)
                for i, output in enumerate(outputs):
                    response = output.outputs[0].text
                    messages_batch[active_chat_ids[i]].append({"role": "assistant", "content": response})
                    responses_batch[active_chat_ids[i]].append(response)
                    
            return responses_batch

