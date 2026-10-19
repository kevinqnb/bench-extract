# model-configs

One YAML file per LLM used by an experiment, keyed by filename (`<key>.yaml`,
`key:` inside must match). Loaded and validated into the
`HardwareRequirement`/`LLMModelConfig` dataclasses in `experiments/config.py`.
Per-model rationale that doesn't fit a comment lives in that model's own
`notes:` field.

## Reproducibility caveat

`seed` + `temperature=0` is best-effort determinism, not a guarantee. GPU batched
inference (vLLM) is not bit-reproducible across runs/hardware due to
kernel/batching nondeterminism. The seed and temperature are captured for
traceability, not promised as exact reproduction.

## Serving

`submit.sh serve` / `serve_model.py` accept any model with `hardware.device: cuda`
(whatever its `serving` value); a hosted API (`device: none`) is refused. Each serve job
picks a free port and registers it.

Recommended for any self-hosted model: **`serving: external` + `bash experiments/
submit.sh serve <model-key>`** (e.g. `gpt-oss-120b.yaml`). `serve_model.py` starts
that one model's vLLM server as its own standalone, long-running job -- typically
its own node/GPU allocation -- and registers its endpoint in
`experiments/.endpoints.json`, which `experiments/serving.py`'s `endpoint_for()`
reads automatically (falling back to the declared `base_url_env` only if that file
has no live entry, e.g. a non-shared filesystem). This is the only mode that
supports comparing more than one self-hosted model at once, which is the normal
case here -- a routing/cascade/scheduling benchmark needs every model_key in M
reachable simultaneously, not served one at a time inside the benchmark job itself.

`serving: external` with no self-hosted server at all covers a commercial/hosted
API: declare no `base_url_env`, and `endpoint_for()` resolves to `None`, i.e. the
OpenAI SDK's own default endpoint. Either way, `api_key_env` needs a real key (a
local vLLM server ignores the value, but the OpenAI SDK requires a non-empty
string regardless).

`serving: local_vllm` starts a self-contained vLLM server *inside* whatever job
runs `experiments/run_benchmark.py`/`run_training.py`, torn down on exit -- no
separate `serve` job. Only use this for a quick single-model run:
`experiments/submit.sh` refuses a benchmark/training config with more than one
`local_vllm` model_key (it can't size one qsub request for two different GPU
jobs), and even one ties up a whole GPU allocation for the client job's entire
runtime, not just the serving portion.

## Several instances of the same weights

`served_model_name` (optional, defaults to `model`) is the name the server advertises
and clients send as the request's `model`. Give each instance a distinct one (see
`qwen3-0.6b-a.yaml` / `-b.yaml`) when the same weights appear more than once in M;
some strategies (Abacus/Palimpzest) key models by that name.

## Hardware fields

`hardware.gpu_compute_capability` and `hardware.max_walltime` size the qsub request
`submit.sh` builds per model (`-l gpu_c=...`, `-l h_rt=...`). `hardware.device: none`
means a hosted API with no local hardware need.
