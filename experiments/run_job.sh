#!/bin/bash -l
# qsub job body invoked by submit.sh via `-v REPO_DIR=...,JOB_CMD=...`.
# Not meant to be run directly.
set -euo pipefail
cd "$REPO_DIR"
# qsub does not forward the login shell's environment; keep model weights off $HOME.
export HF_HOME="${HF_HOME:-/projectnb/mcnet/kevin/cache-hf}"
# FlashInfer (vLLM's sampler) JIT-compiles kernels on first use and needs nvcc. Must share a major
# version with the CUDA build of vLLM's torch (cu130 for vllm 0.32.0); the cluster default is 12.8.
module load "${BENCH_EXTRACT_CUDA_MODULE:-cuda/13.2}"
eval "$JOB_CMD"
