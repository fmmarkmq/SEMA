# Base: published SEMA training image (Ubuntu 24.04 + CUDA + PyTorch + all SEMA Python deps)
FROM allenlao/sema:v0.1@sha256:c3c39d601404025a9e05ca891cc8a44c86a50d4c390fb15e493c4b7a48825271

# UTF-8 locale so tmux and Neovim draw box-drawing characters correctly
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CONSTRAINT=

# Build tools, common utilities, and agent dependencies
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  jq \
  less \
  openssh-client \
  ripgrep \
  tree \
  unzip \
  wget \
  && rm -rf /var/lib/apt/lists/*

# Node.js 22 via NodeSource (for agent CLIs)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get install -y -qq --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1000) for Claude's --dangerously-skip-permissions
# The NVIDIA PyTorch image (allenlao/sema:v0.1's base) is based on ubuntu:24.04,
# which ships with a 'ubuntu' user at uid 1000.
RUN if getent passwd 1000 >/dev/null; then \
      usermod -l sandbox -d /home/sandbox -m $(getent passwd 1000 | cut -d: -f1) 2>/dev/null || true; \
    else \
      groupadd -g 1000 sandbox && useradd -m -u 1000 -g sandbox sandbox; \
    fi

# Agent CLIs
RUN npm install -g @anthropic-ai/claude-code @google/gemini-cli @openai/codex @github/copilot \
  && npm cache clean --force

# Dev / test dependencies (not part of allenlao/sema:v0.1)
RUN pip install --no-cache-dir \
    pytest==9.0.3 \
    pytest-asyncio==1.3.0

# peft is optional for SEMA (only needed for LoRA runs via --use_peft; trl.get_peft_config
# returns None without it when use_peft is False). Installed here because it is absent from
# the allenlao/sema:v0.1 base image and this dev image should be able to run LoRA too.
RUN pip install --no-cache-dir peft==0.19.1

# Writable cache dirs for triton / torchinductor / vLLM / HF when running as non-root sandbox user
ENV TRITON_CACHE_DIR=/workspace/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    VLLM_CACHE_ROOT=/workspace/.cache/vllm

WORKDIR /workspace
CMD ["/bin/bash"]
