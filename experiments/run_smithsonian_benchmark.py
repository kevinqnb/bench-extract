"""Runs the Smithsonian image <-> record join through each method in an
experiment config and scores the returned pairs against gold_pairs.csv"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
from dotenv import load_dotenv

from bench_extract.linkers.base import (
    IMAGE_COLUMN,
    IMAGE_ID_COLUMN,
    RECORD_COLUMN,
    RECORD_ID_COLUMN,
    Linker,
    LinkModel,
)
from experiments.config import (
    MODEL_REGISTRY,
    ExperimentConfig,
    load_experiment_config,
)
from experiments.runtime import RunManifest, Tee, git_sha, installed_versions
from experiments.serving import endpoint_for

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "smithsonian"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "results" / "benchmark"
GOLD_PAIRS_NAME = "gold_pairs.csv"
PAIR_COLUMNS = ["method", "image_id", "record_id", "correct"]
# Versions recorded in run.json
RELEVANT_PACKAGES = [
    "pandas", "litellm", "sentence-transformers", "lotus-ai", "palimpzest",
]
# How a child process is started
RUNNER_MODULE = "experiments.run_smithsonian_benchmark"
RUNNER_EXTRA = "experiments"
UV_ENVIRONMENT_VARIABLE = "UV_PROJECT_ENVIRONMENT"
# Model endpoints
FIRST_LOCAL_PORT = 8000
NO_API_KEY = "EMPTY"

@dataclasses.dataclass(frozen=True)
class Method:
    linker: str  # "module:class", imported only when the method runs
    extra: str  # pyproject.toml extra that installs its library
    venv_dir: str  # its uv environment, relative to the repo root

LINKER_REGISTRY: dict[str, Method] = {
    "lotus": Method(
        linker="bench_extract.linkers.lotus:LotusLinker",
        extra="lotus",
        venv_dir=".venv-lotus",
    ),
    "abacus": Method(
        linker="bench_extract.linkers.abacus:AbacusLinker",
        extra="abacus",
        venv_dir=".venv",
    ),
}

@dataclasses.dataclass(frozen=True)
class LinkageTables:
    images: pd.DataFrame  # IMAGE_ID_COLUMN, IMAGE_COLUMN (file path)
    records: pd.DataFrame  # RECORD_ID_COLUMN, RECORD_COLUMN (text)
    gold_pairs: set[tuple[str, str]]  # (image_id, record_id)

def load_linkage_tables(
    sample_dir: Path, data_root: Path = DATA_ROOT
) -> LinkageTables:
    """Reads gold_pairs.csv; an image's id is its path under data_root"""
    gold = pd.read_csv(sample_dir / GOLD_PAIRS_NAME, dtype=str)
    if gold.empty:
        raise SystemExit(f"No gold pairs in {sample_dir / GOLD_PAIRS_NAME}")
    image_paths = [data_root / path for path in gold["jpg_path"]]
    record_paths = {
        record_id: data_root / path
        for record_id, path in zip(gold["record_id"], gold["record_path"])
    }
    missing = [
        str(path) for path in [*image_paths, *record_paths.values()]
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} file(s) named in {GOLD_PAIRS_NAME} are missing, "
            f"e.g. {missing[0]} (re-run data/download_smithsonian.py)"
        )
    record_ids = sorted(record_paths)
    record_texts = [
        record_paths[record_id].read_text() for record_id in record_ids
    ]
    return LinkageTables(
        images=pd.DataFrame({
            IMAGE_ID_COLUMN: gold["jpg_path"],
            IMAGE_COLUMN: [str(path) for path in image_paths],
        }),
        records=pd.DataFrame({
            RECORD_ID_COLUMN: record_ids,
            RECORD_COLUMN: record_texts,
        }),
        gold_pairs=set(zip(gold["jpg_path"], gold["record_id"])),
    )

def split_by_record(
    tables: LinkageTables, train_fraction: float, seed: int
) -> tuple[LinkageTables, LinkageTables]:
    """Seeded (train, test) split; an image goes where its record goes"""
    record_ids = list(tables.records[RECORD_ID_COLUMN])
    random.Random(seed).shuffle(record_ids)
    n_train_records = round(train_fraction * len(record_ids))
    train_ids = set(record_ids[:n_train_records])
    record_of_image = dict(tables.gold_pairs)
    image_records = tables.images[IMAGE_ID_COLUMN].map(record_of_image)
    image_in_train = image_records.isin(train_ids)
    record_in_train = tables.records[RECORD_ID_COLUMN].isin(train_ids)

    def subset(in_train: bool) -> LinkageTables:
        images = tables.images[image_in_train == in_train]
        records = tables.records[record_in_train == in_train]
        return LinkageTables(
            images=images.reset_index(drop=True),
            records=records.reset_index(drop=True),
            gold_pairs={
                (image_id, record_id)
                for image_id, record_id in tables.gold_pairs
                if (record_id in train_ids) == in_train
            },
        )

    return subset(True), subset(False)

def score_pairs(
    predicted: set[tuple[str, str]], gold: set[tuple[str, str]]
) -> dict[str, float]:
    n_correct = len(predicted & gold)
    precision = n_correct / len(predicted) if predicted else 0.0
    recall = n_correct / len(gold) if gold else 0.0
    f1 = 0.0
    if n_correct:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "n_predicted_pairs": len(predicted),
    }

def start_manifest(
    config: ExperimentConfig,
    config_path: Path,
    train: LinkageTables,
    test: LinkageTables,
) -> RunManifest:
    """run.json's contents at the start of a run"""
    manifest = RunManifest(id=config.id, config_path=str(config_path))
    manifest.git_sha, manifest.git_dirty = git_sha(REPO_ROOT)
    manifest.package_versions = installed_versions(RELEVANT_PACKAGES)
    manifest.started_at = datetime.now(timezone.utc).isoformat()
    manifest.host = socket.gethostname()
    manifest.job_id = os.environ.get("JOB_ID")  # set by SGE under qsub
    manifest.extra = {
        "sample_dir": config.params["sample_dir"],
        "n_train_images": len(train.images),
        "n_train_records": len(train.records),
        "n_test_images": len(test.images),
        "n_test_records": len(test.records),
        "methods": config.params["methods"],
        "model_keys": config.params["model_keys"],
        "embedding_model": config.params["embedding_model"],
    }
    return manifest

@contextlib.contextmanager
def printing_to_log(run_dir: Path) -> Iterator[None]:
    """Copies everything printed into run_dir/log.txt"""
    with open(run_dir / "log.txt", "w") as log_file, \
            contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
        yield

def write_run_outputs(
    run_dir: Path,
    metrics: dict[str, float],
    pair_rows: list[dict],
    manifest: RunManifest,
    status: str,
) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    with open(artifacts_dir / "pairs.csv", "w", newline="") as pairs_file:
        writer = csv.DictWriter(pairs_file, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(pair_rows)
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.status = status
    (run_dir / "run.json").write_text(
        json.dumps(dataclasses.asdict(manifest), indent=2, default=str)
    )

def load_linker_class(method: str) -> type[Linker]:
    module_name, class_name = LINKER_REGISTRY[method].linker.split(":")
    return getattr(importlib.import_module(module_name), class_name)

def connect_models(
    model_keys: list[str],
    base_url_override: str | None,
    endpoints: contextlib.ExitStack,
) -> list[LinkModel]:
    """Finds each model's endpoint; `endpoints` keeps them open"""
    models = []
    for port_offset, model_key in enumerate(model_keys):
        model_config = MODEL_REGISTRY[model_key]
        # A distinct port per model, as in run_benchmark.py
        base_url, _ = endpoints.enter_context(endpoint_for(
            model_config,
            base_url_override=base_url_override,
            port=FIRST_LOCAL_PORT + port_offset,
        ))
        served_name = model_config.served_model_name or model_config.model
        api_key = os.environ.get(model_config.api_key_env, NO_API_KEY)
        models.append(LinkModel(
            key=model_key,
            served_name=served_name,
            base_url=base_url,
            api_key=api_key,
            max_tokens=model_config.max_tokens,
            extra_body=model_config.extra_body,
        ))
    return models

def fit_link_and_score(
    method: str,
    linker: Linker,
    train: LinkageTables,
    test: LinkageTables,
) -> tuple[dict[str, float], list[dict]]:
    """Returns the method's metrics.json entries and its pairs.csv rows"""
    print(f"fitting {method!r}...")
    linker.fit(train.images, train.records, train.gold_pairs)
    print(f"joining with {method!r}...")
    started = time.perf_counter()
    result = linker.link(test.images, test.records)
    elapsed_seconds = time.perf_counter() - started
    if result.plan:
        print(result.plan)
    predicted = set(result.pairs)
    scores = {
        **score_pairs(predicted, test.gold_pairs),
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    print(f"{method}: {scores}")
    metrics = {f"{method}_{name}": value for name, value in scores.items()}
    for name, value in linker.fit_metrics.items():
        metrics[f"{method}_fit_{name}"] = value
    pair_rows = [
        {
            "method": method,
            "image_id": image_id,
            "record_id": record_id,
            "correct": (image_id, record_id) in test.gold_pairs,
        }
        for image_id, record_id in sorted(predicted)
    ]
    return metrics, pair_rows

def run_method(
    method: str,
    config: ExperimentConfig,
    config_path: Path,
    run_dir: Path,
    manifest: RunManifest,
    train: LinkageTables,
    test: LinkageTables,
    base_url_override: str | None,
) -> Path:
    """Runs one method in the current environment (what a child does)"""
    params = config.params
    manifest.extra["methods"] = [method]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.snapshot.yaml")
    with printing_to_log(run_dir), contextlib.ExitStack() as endpoints:
        models = connect_models(
            params["model_keys"], base_url_override, endpoints
        )
        linker = load_linker_class(method)(
            models,
            params["embedding_model"],
            params["join_condition"],
            **params["method_kwargs"][method],
        )
        metrics, pair_rows = fit_link_and_score(method, linker, train, test)
    write_run_outputs(run_dir, metrics, pair_rows, manifest, "success")
    print(f"[{config.id}] {method} done -> {run_dir}")
    return run_dir

def child_command(
    config_path: Path,
    method: str,
    output_root: Path,
    base_url_override: str | None,
) -> list[str]:
    """This runner again, for one method, with that method's library"""
    command = [
        "uv", "run",
        "--extra", LINKER_REGISTRY[method].extra,
        "--extra", RUNNER_EXTRA,
        "-m", RUNNER_MODULE, str(config_path),
        "--method", method,
        "--output-root", str(output_root),
    ]
    if base_url_override is not None:
        command += ["--base-url", base_url_override]
    return command

def child_environment(method: str) -> dict[str, str]:
    """This process's environment variables, pointed at the method's own
    uv environment in place of the current one"""
    environment = dict(os.environ)
    environment.pop("VIRTUAL_ENV", None)
    environment[UV_ENVIRONMENT_VARIABLE] = LINKER_REGISTRY[method].venv_dir
    return environment

def run_methods_in_environments(
    methods: list[str],
    config_path: Path,
    run_dir: Path,
    manifest: RunManifest,
    base_url_override: str | None,
) -> Path:
    """Runs each method as a child process, one after the other, and merges
    what they wrote. A failed method does not stop the rest; the run then
    exits non-zero"""
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.snapshot.yaml")
    metrics: dict[str, float] = {}
    pair_rows: list[dict] = []
    package_versions: dict[str, dict] = {}
    failed_methods: list[str] = []
    with printing_to_log(run_dir):
        for method in methods:
            command = child_command(
                config_path, method, run_dir.parent, base_url_override
            )
            print(f"[{manifest.id}] starting {method!r}: {' '.join(command)}")
            sys.stdout.flush()
            child = subprocess.run(
                command, cwd=REPO_ROOT, env=child_environment(method)
            )
            method_dir = run_dir / method
            if child.returncode != 0:
                failed_methods.append(method)
                print(
                    f"[{manifest.id}] {method!r} failed (exit "
                    f"{child.returncode}); see {method_dir / 'log.txt'}"
                )
                continue
            metrics.update(
                json.loads((method_dir / "metrics.json").read_text())
            )
            pairs_path = method_dir / "artifacts" / "pairs.csv"
            with open(pairs_path, newline="") as pairs_file:
                pair_rows.extend(csv.DictReader(pairs_file))
            child_manifest = json.loads((method_dir / "run.json").read_text())
            package_versions[method] = child_manifest["package_versions"]
    manifest.extra["package_versions_by_method"] = package_versions
    manifest.extra["failed_methods"] = failed_methods
    status = "failed" if failed_methods else "success"
    write_run_outputs(run_dir, metrics, pair_rows, manifest, status)
    print(f"[{manifest.id}] done -> {run_dir}")
    if failed_methods:
        raise SystemExit(f"Method(s) {failed_methods} failed; see {run_dir}")
    return run_dir

def run_smithsonian_benchmark(
    config_path: Path,
    *,
    method: str | None = None,
    dry_run: bool = False,
    output_root_override: Path | None = None,
    base_url_override: str | None = None,
) -> Path | None:
    config = load_experiment_config(config_path)
    params = config.params
    unknown_methods = set(params["methods"]) - set(LINKER_REGISTRY)
    if unknown_methods:
        raise SystemExit(
            f"Unknown method(s) {sorted(unknown_methods)}; "
            f"choices: {sorted(LINKER_REGISTRY)}"
        )
    if method is not None and method not in params["methods"]:
        raise SystemExit(f"{method!r} is not in the config's methods")

    tables = load_linkage_tables(REPO_ROOT / params["sample_dir"])
    train, test = split_by_record(
        tables, params["train_fraction"], config.seed
    )
    print(
        f"[{config.id}] train: {len(train.images)} images x "
        f"{len(train.records)} records; test: {len(test.images)} images x "
        f"{len(test.records)} records"
    )
    if test.records.empty:
        raise SystemExit("train_fraction leaves an empty test split")

    manifest = start_manifest(config, config_path, train, test)
    if dry_run:
        print(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))
        return None
    run_dir = (output_root_override or DEFAULT_OUTPUT_ROOT) / config.id
    if method is None:
        return run_methods_in_environments(
            params["methods"], config_path, run_dir, manifest,
            base_url_override,
        )
    return run_method(
        method, config, config_path, run_dir / method, manifest,
        train, test, base_url_override,
    )

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config", type=Path,
        help="experiments/experiment-configs/benchmark/smithsonian/<id>.yaml",
    )
    parser.add_argument(
        "--method", default=None, choices=sorted(LINKER_REGISTRY),
        help="Run only this method, in the current environment. Default: "
             "every method in the config, each in its own environment",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load the config and data, print the manifest, call no model",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--base-url", default=None,
        help="Use this endpoint for every model_key",
    )
    return parser

def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = build_arg_parser().parse_args()
    run_smithsonian_benchmark(
        args.config,
        method=args.method,
        dry_run=args.dry_run,
        output_root_override=args.output_root,
        base_url_override=args.base_url,
    )

if __name__ == "__main__":
    main()
