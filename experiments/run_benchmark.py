"""Config-dispatched runner: accuracy vs. efficiency benchmark for one or
more extraction strategies (experiments.config.EXTRACTOR_REGISTRY) over a
dataset split (experiments.config.DATASET_REGISTRY). This is the "Accuracy
vs. Efficiency benchmarking" experiment -- see notes/hub/conventions.md for
the config envelope and run-output contract this follows.

    uv run -m experiments.run_benchmark experiments/experiment-configs/benchmark/<id>.yaml

Expected params:
    dataset: str              -- experiments.config.DATASET_REGISTRY key
                                  (experiments/dataset-configs/<dataset>.py)
    dataset_params: dict      -- shape owned by that dataset module; for "vrdu":
                                    corpus: str      -- dataset-configs/vrdu.py's CORPORA key
                                    split_file: str  -- data/vrdu/few_shot-splits/<corpus>/<split_file>.json;
                                                         its 'train' partition fits each method's profiling
                                                         table, its 'valid' partition is what gets scored
    model_keys: list[str]    -- experiments/model-configs/*.yaml keys forming M
    methods: list[str]       -- experiments.config.EXTRACTOR_REGISTRY keys to compare
    limit: Optional[int]     -- cap documents per partition (debugging)
    method_kwargs: Optional[dict[str, dict]]  -- extra constructor kwargs per method
                                name, e.g. {"schedule_baseline": {"min_accuracy": 0.5}}
                                or {"cascade_baseline": {"model_order": [...]}}
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import shutil
import socket
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from bench_extract.models.base import OpenAIChatModel
from bench_extract.profiling import build_profiling_table

from experiments.config import DATASET_REGISTRY, EXTRACTOR_REGISTRY, MODEL_REGISTRY, load_experiment_config
from experiments.runtime import RunManifest, Tee, git_sha, installed_versions
from experiments.serving import endpoint_for

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "results" / "benchmark"
RELEVANT_PACKAGES = ["openai", "pandas", "pyyaml", "httpx"]


def _profile_models(
    models: dict[str, OpenAIChatModel], documents: list, fields: list[str], windowed_text
) -> tuple[dict, dict]:
    """Calls every model on every document once; returns the
    (run_predictions, timing) dicts build_profiling_table expects.
    `windowed_text` is a DatasetRunContext.windowed_text: doc -> str."""
    run_predictions: dict[str, dict] = {}
    timing: dict[str, dict] = {}
    for model_key, model in models.items():
        run_predictions[model_key], timing[model_key] = {}, {}
        for doc in documents:
            text = windowed_text(doc)
            prediction = model.extract(text, fields)
            run_predictions[model_key][doc.document_id] = prediction.field_values
            timing[model_key][doc.document_id] = dataclasses.asdict(prediction.usage)
            print(f"  [{model_key}] {doc.document_id}: {'ok' if prediction.usage.error is None else 'ERROR ' + prediction.usage.error}")
    return run_predictions, timing


def _resolve_split_docs(doc_ids: list[str], all_docs: dict[str, object], limit: Optional[int], label: str) -> list:
    """Filters `doc_ids` (from a dataset split) to those present in
    `all_docs`, THEN applies `limit` -- filtering after slicing would let
    `--limit` silently return fewer documents than requested whenever the
    split references ids missing from the loaded dataset (e.g. only one
    corpus was downloaded), instead of surfacing that as a warning.
    """
    present = [i for i in doc_ids if i in all_docs]
    missing = len(doc_ids) - len(present)
    if missing:
        print(f"  warning: {missing} {label} document(s) from the split file are not in the loaded dataset (not fully downloaded?)")
    return [all_docs[i] for i in present[:limit]]


def _evaluate_extractor(extractor, documents: list, fields: list[str], score_field, windowed_text) -> dict:
    """`score_field` is a DatasetRunContext.score_field: (predicted, ground_truth, field) -> correct."""
    total_correct, total_fields, total_wall = 0, 0, 0.0
    for doc in documents:
        text = windowed_text(doc)
        result = extractor.extract(text, fields)
        total_wall += result.total_wall_seconds
        for f in fields:
            correct = score_field(result.field_values.get(f, []), doc.fields.get(f, []), f)
            total_correct += int(correct)
            total_fields += 1
    return {
        "accuracy": total_correct / total_fields if total_fields else 0.0,
        "mean_wall_seconds": total_wall / len(documents) if documents else 0.0,
        "n_documents": len(documents),
    }


def run_benchmark(
    config_path: Path,
    *,
    limit_override: Optional[int] = None,
    dry_run: bool = False,
    output_root_override: Optional[Path] = None,
    base_url_override: Optional[str] = None,
) -> Optional[Path]:
    config = load_experiment_config(config_path)
    params = config.params
    dataset = params["dataset"]
    if dataset not in DATASET_REGISTRY:
        raise SystemExit(f"Unknown dataset {dataset!r}; choices: {sorted(DATASET_REGISTRY)}")
    dataset_mod = DATASET_REGISTRY[dataset]
    dataset_params = params["dataset_params"]

    ctx = dataset_mod.load_run_context(dataset_params)
    prompt_template = dataset_mod.EXTRACTION_PROMPT_TEMPLATE

    limit = limit_override if limit_override is not None else params.get("limit")
    train_docs = _resolve_split_docs(dataset_mod.load_split_doc_ids(dataset_params, "train"), ctx.all_docs, limit, "train")
    valid_docs = _resolve_split_docs(dataset_mod.load_split_doc_ids(dataset_params, "valid"), ctx.all_docs, limit, "valid")

    output_root = output_root_override or DEFAULT_OUTPUT_ROOT
    run_dir = output_root / config.id

    manifest = RunManifest(id=config.id, config_path=str(config_path))
    manifest.git_sha, manifest.git_dirty = git_sha(REPO_ROOT)
    manifest.package_versions = installed_versions(RELEVANT_PACKAGES)
    manifest.started_at = datetime.now(timezone.utc).isoformat()
    manifest.host = socket.gethostname()
    manifest.job_id = os.environ.get("JOB_ID")  # set by SGE under qsub; None for an interactive run
    manifest.extra = {
        "dataset": dataset,
        "dataset_params": dataset_params,
        "n_train": len(train_docs),
        "n_valid": len(valid_docs),
        "methods": params["methods"],
        "model_keys": params["model_keys"],
    }

    print(f"[{config.id}] dataset={dataset} dataset_params={dataset_params} n_train={len(train_docs)} n_valid={len(valid_docs)}")
    if not train_docs or not valid_docs:
        raise SystemExit(f"No documents found for dataset={dataset!r} dataset_params={dataset_params!r} (check the dataset was downloaded).")

    if dry_run:
        print(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.snapshot.yaml")
    method_kwargs: dict[str, dict] = params.get("method_kwargs", {})

    metrics: dict[str, float] = {}
    with open(run_dir / "log.txt", "w") as log_fh, contextlib.redirect_stdout(Tee(sys.stdout, log_fh)):
        with ExitStack() as stack:
            models: dict[str, OpenAIChatModel] = {}
            # Distinct port per model_key: every model's endpoint_for() context
            # is held open simultaneously (below), and two local_vllm servers on
            # the same default port would collide -- see serving.endpoint_for's
            # docstring.
            for port_offset, model_key in enumerate(params["model_keys"]):
                model_config = MODEL_REGISTRY[model_key]
                base_url, _startup_seconds = stack.enter_context(
                    endpoint_for(model_config, base_url_override=base_url_override, port=8000 + port_offset)
                )
                api_key = os.environ.get(model_config.api_key_env, "EMPTY")
                models[model_key] = OpenAIChatModel(
                    key=model_key,
                    model=model_config.model,
                    prompt_template=prompt_template,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    extra_body=model_config.extra_body,
                    response_format=model_config.response_format,
                )

            print(f"[{config.id}] profiling {len(models)} model(s) on {len(train_docs)} training document(s)...")
            run_predictions, timing = _profile_models(models, train_docs, ctx.fields, ctx.windowed_text)
            profiling_table = build_profiling_table(run_predictions, ctx.all_docs, ctx.fields, ctx.score_field, timing)

            for method in params["methods"]:
                spec = EXTRACTOR_REGISTRY[method]
                extractor = spec.extractor_cls(list(models.values()), **method_kwargs.get(method, {}))
                extractor.fit(profiling_table)
                print(f"[{config.id}] evaluating {method!r} on {len(valid_docs)} validation document(s)...")
                result_metrics = _evaluate_extractor(extractor, valid_docs, ctx.fields, ctx.score_field, ctx.windowed_text)
                for k, v in result_metrics.items():
                    metrics[f"{method}_{k}"] = v

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.status = "success"
    (run_dir / "run.json").write_text(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))
    print(f"[{config.id}] done -> {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiments/experiment-configs/benchmark/<id>.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Cap documents per partition (overrides params.limit)")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print the manifest preview, call no model")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-url", default=None, help="Force this endpoint for every model_key, skipping model-configs serving/base_url_env entirely")
    return parser


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = build_arg_parser().parse_args()
    run_benchmark(
        args.config,
        limit_override=args.limit,
        dry_run=args.dry_run,
        output_root_override=args.output_root,
        base_url_override=args.base_url,
    )


if __name__ == "__main__":
    main()
