#!/bin/bash -l
# qsub job body invoked by submit.sh via `-v REPO_DIR=...,JOB_CMD=...`.
# Not meant to be run directly.
set -euo pipefail
cd "$REPO_DIR"
# qsub does not forward the login shell's environment. Machine-specific settings (HF_HOME, optionally
# BENCH_EXTRACT_CUDA_MODULE) live in the untracked .env, not in this file.
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
# Model weights must stay off $HOME.
: "${HF_HOME:?set HF_HOME in .env (a project/scratch path) -- weights must not download into \$HOME}"
# FlashInfer (vLLM's sampler) JIT-compiles kernels on first use and needs nvcc. Must share a major
# version with the CUDA build of vLLM's torch (cu130 for vllm 0.32.0); the cluster default is 12.8.
module load "${BENCH_EXTRACT_CUDA_MODULE:-cuda/13.2}"
# ...and ninja, which lives in vLLM's own (uv tool) env and is not on the job's PATH.
export PATH="$(dirname "$(readlink -f "$(command -v vllm)")"):$PATH"
eval "$JOB_CMD"
