import dataclasses

import pytest

from experiments import serve_model as serve_model_module
from experiments.config import MODEL_REGISTRY
from experiments.serve_model import serve_model


def test_serve_model_refuses_unknown_model_key():
    with pytest.raises(SystemExit):
        serve_model("not-a-real-model")


def test_serve_model_refuses_model_without_a_gpu(monkeypatch):
    # A hosted API (hardware.device: none) has nothing to serve; refuse
    # before touching vLLM at all.
    hosted = dataclasses.replace(
        MODEL_REGISTRY["gpt-oss-120b"], hardware=dataclasses.replace(MODEL_REGISTRY["gpt-oss-120b"].hardware, device="none")
    )
    monkeypatch.setitem(MODEL_REGISTRY, "hosted-fake", hosted)
    with pytest.raises(SystemExit, match="hardware.device"):
        serve_model("hosted-fake")


def test_serve_model_accepts_external_cuda_model_and_picks_free_port(monkeypatch):
    # serving: external + device: cuda is the documented way to serve a model;
    # stub the server so no vLLM starts, and check what it was asked to launch.
    started = {}

    class StubServer:
        base_url = "http://stub:1/v1"

        def __init__(self, **kwargs):
            started.update(kwargs)

        def start(self):
            raise KeyboardInterrupt  # stop right after construction

        def stop(self):
            pass

    monkeypatch.setattr(serve_model_module, "LocalVLLMServer", StubServer)
    monkeypatch.setattr(serve_model_module, "resolve_vllm_command", lambda: ["vllm", "serve"])
    with pytest.raises(KeyboardInterrupt):
        serve_model("qwen3-0.6b-a")
    assert started["model"] == "Qwen/Qwen3-0.6B"
    assert started["served_model_name"] == "qwen3-0.6b-a"
    assert isinstance(started["port"], int) and started["port"] > 0


def test_concurrent_endpoint_registration_loses_no_entries(tmp_path, monkeypatch):
    import multiprocessing

    from experiments import serving

    monkeypatch.setattr(serving, "ENDPOINTS_FILE", tmp_path / ".endpoints.json")
    keys = [f"model-{i}" for i in range(12)]
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=serving.write_endpoint, args=(k, f"http://h:{i}/v1")) for i, k in enumerate(keys)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert set(serving.read_endpoints()) == set(keys)
    serving.clear_endpoint("model-0")
    assert "model-0" not in serving.read_endpoints()
