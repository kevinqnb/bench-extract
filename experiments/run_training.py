"""Config-dispatched runner: training-dataset-size sweep for a given
extraction strategy across VRDU's official few_shot-splits train sizes
(10/50/100/200) and seeds -- the project's "Training Dataset Size
experiments." Reuses run_benchmark's model-profiling/evaluation plumbing
per split file rather than reimplementing it.

    uv run -m experiments.run_training experiments/experiment-configs/training/<id>.yaml

Expected params:
    corpus: str              -- experiments/dataset-configs/vrdu.py's CORPORA key
    split_files: list[str]   -- data/vrdu/few_shot-splits/<corpus>/<name>.json,
                                 each decoded for its train_size/seed via VRDU's own
                                 naming convention (...-train_N-...-SD_S)
    model_keys: list[str]
    methods: list[str]
    limit: Optional[int]     -- cap documents per partition (debugging)
    method_kwargs: Optional[dict[str, dict]]  -- extra constructor kwargs per method
                                name, e.g. {"schedule_baseline": {"min_accuracy": 0.5}}
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import shutil
import socket
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from bench_extract.datasets import vrdu
from bench_extract.models.base import OpenAIChatModel
from bench_extract.profiling import build_profiling_table

from experiments.config import EXTRACTOR_REGISTRY, MODEL_REGISTRY, load_dataset_config, load_experiment_config, load_extraction_prompt_template
from experiments.run_benchmark import RELEVANT_PACKAGES, _evaluate_extractor, _profile_models, _resolve_split_docs
from experiments.runtime import RunManifest, Tee, git_sha, installed_versions
from experiments.serving import endpoint_for

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "results" / "training"

# e.g. "DeepForm-unk_template-train_100-test_215-valid_100-SD_0" -> train_size=100, seed=0
_SPLIT_NAME_RE = re.compile(r"train_(\d+)-test_\d+-valid_\d+-SD_(\d+)")


def _decode_split_file(name: str) -> tuple[int, int]:
    match = _SPLIT_NAME_RE.search(name)
    if not match:
        raise ValueError(f"Cannot decode train_size/seed from split file name {name!r}")
    return int(match.group(1)), int(match.group(2))


def run_training(
    config_path: Path,
    *,
    limit_override: Optional[int] = None,
    dry_run: bool = False,
    output_root_override: Optional[Path] = None,
    base_url_override: Optional[str] = None,
) -> Optional[Path]:
    config = load_experiment_config(config_path)
    params = config.params
    corpus = params["corpus"]
    dataset_cfg = load_dataset_config(corpus)
    schema = vrdu.load_corpus_schema(corpus)
    fields = sorted(schema.entity_name_to_match_func)
    prompt_template = load_extraction_prompt_template()
    limit = limit_override if limit_override is not None else params.get("limit")
    all_docs = {d.document_id: d for d in vrdu.load_documents(corpus)}

    output_root = output_root_override or DEFAULT_OUTPUT_ROOT
    run_dir = output_root / config.id

    manifest = RunManifest(id=config.id, config_path=str(config_path))
    manifest.git_sha, manifest.git_dirty = git_sha(REPO_ROOT)
    manifest.package_versions = installed_versions(RELEVANT_PACKAGES)
    manifest.started_at = datetime.now(timezone.utc).isoformat()
    manifest.host = socket.gethostname()
    manifest.job_id = os.environ.get("JOB_ID")  # set by SGE under qsub; None for an interactive run
    manifest.extra = {"corpus": corpus, "split_files": params["split_files"], "methods": params["methods"], "model_keys": params["model_keys"]}

    print(f"[{config.id}] corpus={corpus} split_files={params['split_files']}")

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
            # Distinct port per model_key -- see serving.endpoint_for's docstring.
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

            for split_file in params["split_files"]:
                train_size, seed = _decode_split_file(split_file)
                splits = vrdu.load_split(corpus, split_file)
                train_docs = _resolve_split_docs(splits["train"], all_docs, limit, f"{split_file} train")
                valid_docs = _resolve_split_docs(splits["valid"], all_docs, limit, f"{split_file} valid")
                if not train_docs or not valid_docs:
                    raise SystemExit(f"No documents found for split_file={split_file!r} (check data/download_vrdu.py was run).")

                print(f"[{config.id}] {split_file}: train_size={train_size} seed={seed}, profiling {len(models)} model(s) on {len(train_docs)} doc(s)...")
                run_predictions, timing = _profile_models(models, train_docs, fields, dataset_cfg.max_pages, dataset_cfg.max_chars)
                profiling_table = build_profiling_table(run_predictions, all_docs, schema, timing)

                for method in params["methods"]:
                    spec = EXTRACTOR_REGISTRY[method]
                    extractor = spec.extractor_cls(list(models.values()), **method_kwargs.get(method, {}))
                    extractor.fit(profiling_table)
                    result_metrics = _evaluate_extractor(extractor, valid_docs, fields, schema, dataset_cfg.max_pages, dataset_cfg.max_chars)
                    prefix = f"{method}__trainsize_{train_size}__seed_{seed}"
                    for k, v in result_metrics.items():
                        metrics[f"{prefix}_{k}"] = v

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.status = "success"
    (run_dir / "run.json").write_text(json.dumps(dataclasses.asdict(manifest), indent=2, default=str))
    print(f"[{config.id}] done -> {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiments/experiment-configs/training/<id>.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Cap documents per partition (overrides params.limit)")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print the manifest preview, call no model")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-url", default=None, help="Force this endpoint for every model_key, skipping model-configs serving/base_url_env entirely")
    return parser


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = build_arg_parser().parse_args()
    run_training(
        args.config,
        limit_override=args.limit,
        dry_run=args.dry_run,
        output_root_override=args.output_root,
        base_url_override=args.base_url,
    )


if __name__ == "__main__":
    main()
