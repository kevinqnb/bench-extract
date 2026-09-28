"""VRDU dataset adapter: implements the experiments.config dataset-registry
contract (bench_extract.datasets.base.DatasetRunContext + load_run_context /
load_split_doc_ids / describe_split) on top of bench_extract.datasets.vrdu's
loader, plus per-corpus windowing defaults and the extraction prompt
template. Python (not YAML), since this holds a prompt template, not just
scalars.

Loaded by path, not imported as a package (experiment-configs' sibling
`dataset-configs/` directory name has a hyphen, so it can't be a Python
package) -- see experiments.config.load_dataset_registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bench_extract.datasets import matching, vrdu as vrdu_loader
from bench_extract.datasets.base import DatasetRunContext


@dataclass(frozen=True)
class VRDUCorpusConfig:
    corpus: str
    # Every extractable field for this corpus and the VRDU match function
    # (bench_extract.datasets.matching.MATCH_FUNCS) that scores it -- the
    # fields to extract are this config's business, never inferred from
    # whatever data/vrdu/data.json happens to contain (load_run_context
    # cross-checks against it as a drift guard, not a source).
    match_func_by_field: dict[str, str]
    max_pages: int = 3
    max_chars: int = 9000


CORPORA: dict[str, VRDUCorpusConfig] = {
    "ad-buy-form": VRDUCorpusConfig(
        corpus="ad-buy-form",
        match_func_by_field={
            "advertiser": "GeneralStringMatch",
            "agency": "GeneralStringMatch",
            "contract_num": "NumericalStringMatch",
            "flight_from": "DateMatch",
            "flight_to": "DateMatch",
            "gross_amount": "PriceMatch",
            "product": "GeneralStringMatch",
            "tv_address": "AddressMatch",
            "property": "GeneralStringMatch",
            "channel": "GeneralStringMatch",
            "program_desc": "GeneralStringMatch",
            "program_start_date": "DateMatch",
            "program_end_date": "DateMatch",
            "sub_amount": "PriceMatch",
        },
        max_pages=3,
        max_chars=9000,
    ),
    "registration-form": VRDUCorpusConfig(
        corpus="registration-form",
        # TODO: not yet downloaded in this checkout (`uv run data/download_vrdu.py
        # --corpus registration-form`) -- populate from the printed
        # bench_extract.datasets.vrdu.load_corpus_schema("registration-form")
        # .entity_name_to_match_func once it has been. load_run_context refuses
        # to run this corpus until this is filled in.
        match_func_by_field={},
        max_pages=2,
        max_chars=6000,
    ),
}


EXTRACTION_PROMPT_TEMPLATE = """You are extracting structured fields from a scanned government form.

FIELDS TO EXTRACT: {fields}

DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"

Return a single JSON object mapping each field name to a list of extracted text \
value(s) as they literally appear in the document. Use an empty list for a field \
that is not present. Do not include any other keys."""


# e.g. "DeepForm-unk_template-train_100-test_215-valid_100-SD_0" -> (100, 0)
_SPLIT_NAME_RE = re.compile(r"train_(\d+)-test_\d+-valid_\d+-SD_(\d+)")


def load_run_context(dataset_params: dict, data_root: Path = vrdu_loader.DEFAULT_DATA_ROOT) -> DatasetRunContext:
    """dataset_params: {"corpus": "ad-buy-form" | "registration-form", ...}."""
    corpus = dataset_params["corpus"]
    if corpus not in CORPORA:
        raise KeyError(f"Unknown corpus {corpus!r}; choices: {sorted(CORPORA)}")
    cfg = CORPORA[corpus]
    if not cfg.match_func_by_field:
        raise NotImplementedError(
            f"CORPORA[{corpus!r}].match_func_by_field is empty in experiments/dataset-configs/vrdu.py -- "
            f"run `uv run data/download_vrdu.py --corpus {corpus}`, then populate it from "
            f"bench_extract.datasets.vrdu.load_corpus_schema({corpus!r}).entity_name_to_match_func."
        )
    match_func_by_field = cfg.match_func_by_field

    # Drift guard, not the source: the config above is authoritative for
    # which fields get extracted, but data/vrdu/data.json (regenerable, not
    # committed) is where VRDU's upstream meta.json actually ends up, so a
    # future re-download disagreeing with the committed config should fail
    # loud rather than silently score against whichever one a caller reads.
    schema = vrdu_loader.load_corpus_schema(corpus, data_root=data_root)
    if schema.entity_name_to_match_func != match_func_by_field:
        only_in_data = sorted(set(schema.entity_name_to_match_func) - set(match_func_by_field))
        only_in_config = sorted(set(match_func_by_field) - set(schema.entity_name_to_match_func))
        mismatched = sorted(
            f
            for f in set(schema.entity_name_to_match_func) & set(match_func_by_field)
            if schema.entity_name_to_match_func[f] != match_func_by_field[f]
        )
        raise ValueError(
            f"CORPORA[{corpus!r}].match_func_by_field (dataset-configs/vrdu.py) disagrees with "
            f"{data_root}/data.json's schema -- only_in_data={only_in_data} only_in_config={only_in_config} "
            f"mismatched_match_func={mismatched}"
        )

    all_docs = {d.document_id: d for d in vrdu_loader.load_documents(corpus, data_root=data_root)}

    def score_field(predicted: list[str], ground_truth: list[str], field: str) -> bool:
        return matching.score_prediction(predicted, ground_truth, match_func_by_field[field])

    return DatasetRunContext(
        fields=sorted(match_func_by_field),
        all_docs=all_docs,
        windowed_text=lambda doc: vrdu_loader.windowed_text(doc, max_pages=cfg.max_pages, max_chars=cfg.max_chars),
        score_field=score_field,
    )


def load_split_doc_ids(dataset_params: dict, part: str, data_root: Path = vrdu_loader.DEFAULT_DATA_ROOT) -> list[str]:
    """dataset_params: {"corpus": str, "split_file": str}. part: "train" | "valid"."""
    splits = vrdu_loader.load_split(dataset_params["corpus"], dataset_params["split_file"], data_root=data_root)
    return splits[part]


def describe_split(dataset_params: dict) -> dict:
    """Decodes VRDU's official few_shot-splits naming convention
    (...-train_N-...-SD_S) for metrics namespacing."""
    split_file = dataset_params["split_file"]
    match = _SPLIT_NAME_RE.search(split_file)
    if not match:
        raise ValueError(f"Cannot decode train_size/seed from split file name {split_file!r}")
    return {"train_size": int(match.group(1)), "seed": int(match.group(2))}


def training_split_params(dataset_params: dict) -> list[dict]:
    """dataset_params: {"corpus": str, "split_files": list[str]}. Expands
    VRDU's list-of-split-files sweep into one dataset_params dict per
    split_file, each valid input to load_run_context / describe_split /
    load_split_doc_ids."""
    return [{**dataset_params, "split_file": split_file} for split_file in dataset_params["split_files"]]
