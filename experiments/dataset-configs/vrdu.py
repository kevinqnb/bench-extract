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
from dataclasses import dataclass, field
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
    # field -> description, handed to strategies that prompt per field
    # (Abacus). Keys must equal match_func_by_field's (enforced in
    # __post_init__). Written for a model that has never seen the form: what the
    # field is, where it sits, and what its value looks like.
    field_descriptions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.field_descriptions and set(self.field_descriptions) != set(self.match_func_by_field):
            raise ValueError(
                f"{self.corpus}: field_descriptions and match_func_by_field disagree -- "
                f"missing={sorted(set(self.match_func_by_field) - set(self.field_descriptions))} "
                f"extra={sorted(set(self.field_descriptions) - set(self.match_func_by_field))}"
            )


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
        field_descriptions={
            "advertiser": (
                "The political committee, candidate campaign, or organization the ad time is bought for "
                "(the buyer named on the order, e.g. 'Michael Bloomberg 2020, Inc' or 'Committee to Elect Mike Carr'). "
                "Not the station and not the media-buying agency."
            ),
            "agency": (
                "The media-buying agency or consultancy that placed the order on the advertiser's behalf "
                "(e.g. 'Beacon Media -- EDI (5138)'). Often absent when the campaign buys directly; "
                "return an empty list if no separate agency is named."
            ),
            "contract_num": (
                "The contract / order number that identifies this ad-buy contract, usually labelled "
                "'Contract', 'Contract #', 'Order No.' or similar near the top of the form. Digits only, e.g. '950658'."
            ),
            "flight_from": (
                "Start date of the whole contract's flight (the overall date range the buy runs), "
                "usually labelled 'Flight', 'Flight Dates' or 'Start Date'. A date such as '03/27/20'. "
                "Not the start date of an individual line item."
            ),
            "flight_to": (
                "End date of the whole contract's flight (the overall date range the buy runs), "
                "usually labelled 'Flight', 'Flight Dates' or 'End Date'. A date such as '03/31/20'. "
                "Not the end date of an individual line item."
            ),
            "gross_amount": (
                "The total gross dollar amount of the contract, usually labelled 'Total', 'Gross Amount', "
                "'Total Gross' or 'Gross Total' near the bottom or in a summary box. A price such as '$5,625.00'. "
                "Not a single line item's amount."
            ),
            "product": (
                "What the ad promotes: the candidate or race, or a ballot issue (e.g. 'Mike Carr for Jackson Co "
                "States Atty', 'issue'). Often labelled 'Product', 'Estimate' or 'Description' in the header."
            ),
            "tv_address": (
                "The television station's or its billing office's mailing address as printed on the form, "
                "typically a remit-to / payment address (street or PO box, city, state, ZIP), "
                "e.g. '4614 Collection Center Drive Chicago, IL 60693'. Include the full multi-line address."
            ),
            "property": (
                "The TV station's call sign or name that the ad runs on (e.g. 'KMSP', 'WXXA', 'WSIL-TV'), "
                "usually in the form's letterhead or labelled 'Station' / 'Property'."
            ),
            "channel": (
                "The channel identifier of the station for a line item: a channel number or call letters "
                "(e.g. '9', '3', 'WXXA'), in a 'Channel' / 'Ch.' column of the order lines."
            ),
            "program_desc": (
                "The program, daypart or spot description for a line item: the show or time slot the ad is bought in "
                "(e.g. 'Fox 9 AM News at 6am', 'Tuesday Prime'). From the description column of the order lines."
            ),
            "program_start_date": (
                "Start date of a single line item / program airing, from the line-item table's start-date or "
                "'From' column (e.g. '02/17/20'). Distinct from the whole-contract flight dates."
            ),
            "program_end_date": (
                "End date of a single line item / program airing, from the line-item table's end-date or "
                "'To' column (e.g. '02/23/20'). Distinct from the whole-contract flight dates."
            ),
            "sub_amount": (
                "The dollar amount charged for a single line item (a row of the order table), "
                "such as '$600.00'. Not the contract's total gross amount."
            ),
        },
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
        field_descriptions=dict(cfg.field_descriptions),
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
