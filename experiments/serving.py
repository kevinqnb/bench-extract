"""Optional local vLLM server management, ported from govscape-extract's
experiments/serving.py (same reasoning, renamed env vars).

A model-configs/*.yaml entry with `serving: local_vllm` gets a
self-contained `LocalVLLMServer` started and torn down around the run that
uses it -- no separately-submitted "server job" another job calls into over
the network. `serving: external` (a hosted API, or a vLLM server someone
else already started) is resolved from `base_url_env` instead.

vLLM itself is intentionally NOT a dependency of this project (see the note
in pyproject.toml) -- this module only ever shells out to a `vllm` CLI that
must already be on PATH (e.g. `uv tool install vllm` into its own
environment), or to $BENCH_EXTRACT_VLLM_COMMAND if set (e.g. a
Singularity/container wrapper).
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from experiments.config import LLMModelConfig

DEFAULT_STARTUP_TIMEOUT_S = 900.0  # a 100B+-class model's weight load can take minutes
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_VLLM_COMMAND = "vllm serve"
ENDPOINTS_FILE = Path(__file__).resolve().parent / ".endpoints.json"


def resolve_vllm_command(override: Optional[str] = None) -> list[str]:
    command_str = override or os.environ.get("BENCH_EXTRACT_VLLM_COMMAND") or DEFAULT_VLLM_COMMAND
    return shlex.split(command_str)


class ServerStartupError(RuntimeError):
    pass


def read_endpoints() -> dict:
    if not ENDPOINTS_FILE.exists():
        return {}
    try:
        return json.loads(ENDPOINTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def write_endpoint(model_key: str, base_url: str) -> None:
    endpoints = read_endpoints()
    endpoints[model_key] = {"base_url": base_url, "updated_at": datetime.now(timezone.utc).isoformat()}
    ENDPOINTS_FILE.write_text(json.dumps(endpoints, indent=2))


def clear_endpoint(model_key: str) -> None:
    endpoints = read_endpoints()
    if endpoints.pop(model_key, None) is not None:
        ENDPOINTS_FILE.write_text(json.dumps(endpoints, indent=2))


def _health_url_from_base(base_url: str) -> str:
    return base_url.rsplit("/v1", 1)[0].rstrip("/") + "/health"


def _is_reachable(base_url: str, timeout: float = 2.0) -> bool:
    try:
        return httpx.get(_health_url_from_base(base_url), timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


@dataclass
class LocalVLLMServer:
    model: str
    served_model_name: Optional[str] = None
    port: int = 8000
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    log_path: Optional[Path] = None
    inherit_stdio: bool = False
    vllm_command: list[str] = field(default_factory=lambda: ["vllm", "serve"])
    # "0.0.0.0" so another node on a multi-node cluster job can reach it;
    # still reachable at 127.0.0.1 from this same node for the health check.
    host: str = "0.0.0.0"
    public_host: Optional[str] = None

    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _log_fh: Any = field(default=None, init=False, repr=False)
    startup_seconds: Optional[float] = field(default=None, init=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.public_host or socket.gethostname()}:{self.port}/v1"

    @property
    def _health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def start(self) -> None:
        served_name = self.served_model_name or self.model
        argv = [
            *self.vllm_command,
            self.model,
            "--served-model-name",
            served_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        if self.log_path:
            self._log_fh = open(self.log_path, "w")
            stdout, stderr = self._log_fh, subprocess.STDOUT
        elif self.inherit_stdio:
            self._log_fh = None
            stdout, stderr = None, None
        else:
            self._log_fh = subprocess.DEVNULL
            stdout, stderr = subprocess.DEVNULL, subprocess.STDOUT
        t0 = time.monotonic()
        # start_new_session=True so teardown can kill vLLM's whole worker
        # process group, not just this parent process.
        self._proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            self._wait_healthy(t0)
        except Exception:
            self.stop()
            raise
        self.startup_seconds = time.monotonic() - t0

    def _output_hint(self) -> str:
        if self.log_path:
            return f"check {self.log_path}"
        if self.inherit_stdio:
            return "see vllm's output above"
        return "stdout/stderr were discarded -- pass log_path=... to capture them"

    def _wait_healthy(self, t0: float) -> None:
        deadline = t0 + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerStartupError(
                    f"vllm serve exited early with code {self._proc.returncode} "
                    f"(model={self.model!r}); {self._output_hint()}"
                )
            try:
                resp = httpx.get(self._health_url, timeout=5.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(self.poll_interval_s)
        raise ServerStartupError(f"vllm serve for {self.model!r} did not become healthy within {self.startup_timeout_s}s")

    def stop(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            if self._log_fh not in (None, subprocess.DEVNULL):
                self._log_fh.close()

    def wait(self) -> None:
        if self._proc is not None:
            self._proc.wait()

    def __enter__(self) -> "LocalVLLMServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


@contextmanager
def endpoint_for(
    model_config: LLMModelConfig, base_url_override: Optional[str] = None, port: int = 8000
) -> Iterator[tuple[Optional[str], Optional[float]]]:
    """Yield (base_url, startup_seconds).

    base_url_override always wins. Otherwise: `serving == "local_vllm"`
    launches+tears down a LocalVLLMServer for the duration of the `with`
    block; `serving == "external"` checks ENDPOINTS_FILE for a live entry,
    then falls back to os.environ[model_config.base_url_env]. A model with
    no `base_url_env` at all (a hosted API) yields base_url=None, i.e.
    "whatever the OpenAI SDK defaults to."

    `port` matters only for `local_vllm`: a runner that holds more than one
    model's endpoint_for() open at once (e.g. run_benchmark.py's ExitStack
    over every model_key in M) MUST pass a distinct port per model, or the
    second server either fails to bind or -- worse -- its health check can
    observe the first server's already-healthy port and silently return the
    wrong model's base_url.
    """
    if base_url_override:
        yield base_url_override, None
        return
    if model_config.serving == "local_vllm":
        with LocalVLLMServer(model=model_config.model, extra_args=model_config.vllm_args, vllm_command=resolve_vllm_command(), port=port) as server:
            yield server.base_url, server.startup_seconds
        return

    cached = read_endpoints().get(model_config.key)
    if cached and _is_reachable(cached["base_url"]):
        yield cached["base_url"], None
        return

    if not model_config.base_url_env:
        yield None, None
        return
    base_url = os.environ.get(model_config.base_url_env)
    if not base_url:
        raise ValueError(
            f"{model_config.key}: no reachable endpoint found. Set "
            f"${model_config.base_url_env} to the running server's base URL "
            "(e.g. http://host:8000/v1), or pass --base-url to override."
        )
    yield base_url, None
