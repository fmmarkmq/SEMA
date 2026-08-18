# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import asyncio
from typing import List, Dict
import random
import numpy as np
import torch
import litellm
from litellm import BadRequestError
from openai import PermissionDeniedError
from litellm.caching.caching import Cache
litellm.cache = Cache(type="disk")
from dotenv import load_dotenv
load_dotenv(".env")


def setup_seed(seed: int = 42, use_cuda: bool = True):
    """
    Set seed for random, numpy, and PyTorch.
    If use_cuda is True, set seed for PyTorch (CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic(use_cuda: bool = True):
    """
    Set deterministic mode for PyTorch.
    If use_cuda is True and CUDA is available, set deterministic mode for PyTorch (CUDA).
    """
    if use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


async def get_completions(
    text: str,
    model_id: str = "gpt-4o-mini",
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float = 1.0,
    n_samples: int = 1,
    system_prompt: str | None = None,
    init_chat_history: List[dict] | None = None,
    caching: bool = True,
    **kwargs
    ) -> List[Dict]:
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if init_chat_history:
        messages.extend(init_chat_history)
    messages.append({'role': 'user', 'content': text})
    

    if os.getenv("X_API_KEY"):
        if "extra_headers" not in kwargs:
            kwargs["extra_headers"] = {}
        kwargs["extra_headers"]["X-API-Key"] = os.getenv("X_API_KEY")


    tasks = []
    for _ in range(n_samples):
        task = litellm.acompletion(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    caching=caching,
                    n=1,
                    **kwargs
                    )
        tasks.append(task)
    try:
        responses = await asyncio.gather(*tasks)
    except (BadRequestError, PermissionDeniedError) as e:
        responses = [litellm.ModelResponse(
            model=model_id, 
            choices=[litellm.Choices(
                message=litellm.Message(
                    role='assistant', 
                    content=f"Error: {e}"
                )
            )]
        ) for _ in range(n_samples)]

    responses = [response.to_dict() for response in responses]

    return responses


if __name__ == "__main__":
    import asyncio

    async def main():
        responses = await get_completions(
            text="What is the capital of France?",
            model_id="gpt-4.1",
            max_tokens=10,
            temperature=0.5,
            n_samples=2,
            system_prompt="You are a helpful assistant."
        )
        for response in responses:
            print(response)

    asyncio.run(main())