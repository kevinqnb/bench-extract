"""Starts one model's vLLM server as a standalone, long-running process --
the "server" half of bench-extract's serve/client split.

Comparing routing/cascade/scheduling strategies needs several models
reachable AT THE SAME TIME (M has more than one member), unlike a
single-model benchmark -- so each `serving: local_vllm` model in
experiments/model-configs/ gets its own server, typically its own qsub job
on its own node (see experiments/submit.sh's `serve` subcommand), separate
from the experiments/run_benchmark.py / run_training.py "client" job that
calls into all of them at once.

    uv run -m experiments.serve_model --model-key gpt-oss-120b

Registers its endpoint in experiments/.endpoints.json on startup (read by
experiments/serving.py's `endpoint_for`; requires a filesystem shared
between this job's node and the client job's node -- true by default on
this project's SCC allocation) and removes it on clean shutdown
(SIGTERM/SIGINT). A `kill -9` or OOM skips that cleanup, which is exactly
why `endpoint_for` always double-checks a cached entry with a live
`/health` request before trusting it, rather than trusting the file alone.
Blocks in the foreground until killed.
"""

from __future__ import annotations

import argparse
import signal
import sys

from dotenv import load_dotenv

from experiments.config import MODEL_REGISTRY, REPO_ROOT
from experiments.serving import LocalVLLMServer, clear_endpoint, resolve_vllm_command, write_endpoint


def serve_model(model_key: str, port: int = 8000) -> None:
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model_key {model_key!r}; choices: {sorted(MODEL_REGISTRY)}")
    model_config = MODEL_REGISTRY[model_key]
    if model_config.serving != "local_vllm":
        raise SystemExit(
            f"{model_key}: serving={model_config.serving!r}, not 'local_vllm'. "
            "serve_model.py only starts models configured for local_vllm serving -- "
            "an 'external' model is either a hosted API or already served by someone else."
        )

    server = LocalVLLMServer(
        model=model_config.model,
        port=port,
        extra_args=model_config.vllm_args,
        vllm_command=resolve_vllm_command(),
        inherit_stdio=True,
    )
    server.start()
    write_endpoint(model_key, server.base_url)
    print(f"[{model_key}] serving at {server.base_url} (registered in experiments/.endpoints.json)")

    def _shutdown(signum, frame):
        print(f"[{model_key}] received signal {signum}, shutting down...")
        clear_endpoint(model_key)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.wait()
    finally:
        clear_endpoint(model_key)
        server.stop()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve_model(args.model_key, port=args.port)


if __name__ == "__main__":
    main()
