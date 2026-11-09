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
# Triton (torch.compile / profile_run) gcc-compiles a small C helper that #includes <Python.h>; the
# nodes' system python3.12 ships no headers, so point gcc at the cluster module's. Needed whenever
# vLLM's torch-compile cache misses (it recompiles). Override the path in .env if the module moves.
export CPATH="${BENCH_EXTRACT_PYTHON_INCLUDE:-/share/pkg.8/python3/3.12.4/install/include/python3.12}${CPATH:+:$CPATH}"
[ -f "${CPATH%%:*}/Python.h" ] || { echo "run_job.sh: Python.h not found under ${CPATH%%:*}" >&2; exit 1; }
eval "$JOB_CMD"
