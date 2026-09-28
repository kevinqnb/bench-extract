"""Loads experiment configuration: model-configs/*.yaml (which LLM, what
hardware, how it's served), EXTRACTOR_REGISTRY (which class each
params.method name in an experiment config resolves to), DATASET_REGISTRY
(which dataset-configs/<key>.py module a params.dataset name resolves to --
see bench_extract.datasets.base for the module contract), and the config
envelope every experiments/experiment-configs/{benchmark,training}/<id>.yaml
must have (id/project/description/seed/params -- see notes/hub/conventions.md).

Mirrors govscape-extract/experiments/config.py's split between *intent*
(this module) and *runtime facts* (experiments/runtime.py's RunManifest).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Literal, Optional

import yaml

from bench_extract.extractors.abacus import AbacusExtractor
from bench_extract.extractors.automix import AutomixExtractor
from bench_extract.extractors.bargain import BargainExtractor
from bench_extract.extractors.base import MultiModelExtractor
from bench_extract.extractors.cascade_extractor import CascadeExtractor
from bench_extract.extractors.cascade_routing import CascadeRoutingExtractor
from bench_extract.extractors.doctopus import DoctopusExtractor
from bench_extract.extractors.frugalgpt import FrugalGPTExtractor
from bench_extract.extractors.hybridllm import HybridLLMExtractor
from bench_extract.extractors.route_extractor import RouteExtractor
from bench_extract.extractors.routellm import RouteLLMExtractor
from bench_extract.extractors.schedule_extractor import ScheduleExtractor
from bench_extract.extractors.task_cascade import TaskCascadeExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIGS_DIR = Path(__file__).resolve().parent / "model-configs"
DATASET_CONFIGS_DIR = Path(__file__).resolve().parent / "dataset-configs"
EXPERIMENT_CONFIGS_DIR = Path(__file__).resolve().parent / "experiment-configs"


# --- Model registry ----------------------------------------------------------

@dataclass(frozen=True)
class HardwareRequirement:
    """Declarative hardware expectation -- documentation plus a guard
    submit.sh can check, not an auto-provisioner."""

    device: Literal["cpu", "cuda", "mps", "none"] = "cpu"
    min_vram_gb: Optional[float] = None  # None => no GPU needed
    gpu_count: int = 0
    gpu_compute_capability: Optional[str] = None  # submit.sh's `-l gpu_c=...`
    max_walltime: str = "24:00:00"  # submit.sh's `-l h_rt=...`
    notes: str = ""


@dataclass(frozen=True)
class LLMModelConfig:
    key: str
    model: str  # passed to OpenAIChatModel(model=...) / expected `vllm serve <model>` name
    role: Literal["candidate"] = "candidate"
    serving: Literal["local_vllm", "external"] = "external"
    base_url_env: Optional[str] = None
    api_key_env: str = "BENCH_EXTRACT_LLM_API_KEY"
    hardware: HardwareRequirement = field(default_factory=HardwareRequirement)
    vllm_args: list[str] = field(default_factory=list)
    temperature: Optional[float] = 0.0
    seed: Optional[int] = 0
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = None
    max_retries: Optional[int] = None
    extra_body: dict = field(default_factory=dict)
    response_format: Optional[dict] = field(default_factory=lambda: {"type": "json_object"})
    notes: str = ""


def _hardware_from_dict(data: Optional[dict]) -> HardwareRequirement:
    return HardwareRequirement(**(data or {}))


def _load_model_config(path: Path) -> LLMModelConfig:
    data = yaml.safe_load(path.read_text())
    kind = data.pop("kind")
    if kind != "llm":
        raise ValueError(f"{path}: unknown kind {kind!r} (only 'llm' is implemented so far)")
    hardware = _hardware_from_dict(data.pop("hardware", None))
    config = LLMModelConfig(hardware=hardware, **data)
    assert config.key == path.stem, f"{path}: key {config.key!r} does not match filename"
    return config


def load_model_registry(directory: Path = MODEL_CONFIGS_DIR) -> dict[str, LLMModelConfig]:
    return {path.stem: _load_model_config(path) for path in sorted(directory.glob("*.yaml"))}


MODEL_REGISTRY: dict[str, LLMModelConfig] = load_model_registry()


# --- Extractor strategy registry ---------------------------------------------

@dataclass(frozen=True)
class ExtractorSpec:
    extractor_cls: type[MultiModelExtractor]
    base_kind: Literal["route", "cascade", "schedule"]


EXTRACTOR_REGISTRY: dict[str, ExtractorSpec] = {
    # Working baselines -- see each base class's module docstring.
    "route_baseline": ExtractorSpec(RouteExtractor, "route"),
    "cascade_baseline": ExtractorSpec(CascadeExtractor, "cascade"),
    "schedule_baseline": ExtractorSpec(ScheduleExtractor, "schedule"),
    # Method stubs -- fit() raises NotImplementedError until wired in.
    "hybridllm": ExtractorSpec(HybridLLMExtractor, "route"),
    "routellm": ExtractorSpec(RouteLLMExtractor, "route"),
    "frugalgpt": ExtractorSpec(FrugalGPTExtractor, "cascade"),
    "bargain": ExtractorSpec(BargainExtractor, "cascade"),
    "task_cascade": ExtractorSpec(TaskCascadeExtractor, "cascade"),
    "automix": ExtractorSpec(AutomixExtractor, "cascade"),
    "abacus": ExtractorSpec(AbacusExtractor, "schedule"),
    "doctopus": ExtractorSpec(DoctopusExtractor, "schedule"),
    "cascade_routing": ExtractorSpec(CascadeRoutingExtractor, "cascade"),
}


# --- Dataset registry (experiments/dataset-configs/<key>.py) ----------------
# Loaded by path, not imported: "dataset-configs" has a hyphen, so it can't
# be a Python package. Each module is the single per-dataset customization
# point (corpus/windowing/prompt-template plus the load_run_context /
# load_split_doc_ids / describe_split functions run_benchmark.py and
# run_training.py dispatch through via params["dataset"]) -- see
# bench_extract.datasets.base for the required contract and
# dataset-configs/vrdu.py for the reference implementation.

_REQUIRED_DATASET_MODULE_ATTRS = (
    "EXTRACTION_PROMPT_TEMPLATE",
    "load_run_context",
    "load_split_doc_ids",
    "describe_split",
    "training_split_params",
)


def _load_dataset_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"bench_extract_dataset_config_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    # Must be registered in sys.modules *before* exec: the module's own
    # `@dataclass` fields use postponed annotations (`from __future__
    # import annotations`), and dataclasses resolves those string
    # annotations by looking the module back up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    missing = [attr for attr in _REQUIRED_DATASET_MODULE_ATTRS if not hasattr(module, attr)]
    assert not missing, f"{path}: dataset module missing required attrs: {missing}"
    return module


def load_dataset_registry(directory: Path = DATASET_CONFIGS_DIR) -> dict[str, ModuleType]:
    return {path.stem: _load_dataset_module(path) for path in sorted(directory.glob("*.py"))}


DATASET_REGISTRY: dict[str, ModuleType] = load_dataset_registry()


# --- Config envelope (notes/hub/conventions.md) ------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    project: str
    description: str
    seed: int
    params: dict


def load_experiment_config(path: Path) -> ExperimentConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    required = {"id", "project", "description", "seed", "params"}
    missing = required - set(data)
    assert not missing, f"{path}: missing required config-envelope keys: {sorted(missing)}"
    assert data["id"] == path.stem, f"{path}: id {data['id']!r} does not match filename {path.stem!r}"
    return ExperimentConfig(id=data["id"], project=data["project"], description=data["description"], seed=data["seed"], params=data["params"])
