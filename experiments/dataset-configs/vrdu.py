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
from typing import Optional

from bench_extract.datasets import matching, vrdu as vrdu_loader
from bench_extract.datasets.base import DatasetRunContext
from bench_extract.scoring.unified import NATIVE, ScoringConfig, coerce_records, is_empty, unified_context_kwargs


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
    # field is, where it sits, and what its value looks like. Line-item
    # sub-field descriptions are folded into the one `line_item_group`
    # description (see line_items_description).
    field_descriptions: dict[str, str] = field(default_factory=dict)
    # Unified measure (bench_extract.scoring.unified): each field's type, keys
    # equal to match_func_by_field's. Line-item sub-fields keep their own type
    # inside the record.
    field_type_by_field: dict[str, str] = field(default_factory=dict)
    # Fields VRDU annotates as "line_item" rows (meta.json's
    # entity_appearance_pattern; load_run_context cross-checks it). They are
    # not extracted one by one: models return one extraction field,
    # `line_item_group`, as a list of row objects with these keys, so which
    # values share a row is explicit. The unified measure scores it as records;
    # the native one flattens it back to per-sub-field lists.
    line_item_fields: list[str] = field(default_factory=list)
    line_item_group: str = "line_items"

    def __post_init__(self):
        for name, mapping in (("field_descriptions", self.field_descriptions), ("field_type_by_field", self.field_type_by_field)):
            if mapping and set(mapping) != set(self.match_func_by_field):
                raise ValueError(
                    f"{self.corpus}: {name} and match_func_by_field disagree -- "
                    f"missing={sorted(set(self.match_func_by_field) - set(mapping))} "
                    f"extra={sorted(set(mapping) - set(self.match_func_by_field))}"
                )
        if not set(self.line_item_fields) <= set(self.match_func_by_field):
            raise ValueError(f"{self.corpus}: line_item_fields not in match_func_by_field: {sorted(set(self.line_item_fields) - set(self.match_func_by_field))}")
        if self.line_item_group in self.match_func_by_field:
            raise ValueError(f"{self.corpus}: line_item_group {self.line_item_group!r} collides with a field name")

    @property
    def header_fields(self) -> list[str]:
        """Fields extracted on their own (everything but the line-item sub-fields)."""
        return sorted(f for f in self.match_func_by_field if f not in self.line_item_fields)

    @property
    def extraction_fields(self) -> list[str]:
        return self.header_fields + ([self.line_item_group] if self.line_item_fields else [])

    def line_items_description(self) -> str:
        keys = " ".join(f"[{f}] {self.field_descriptions.get(f, f.replace('_', ' '))}" for f in self.line_item_fields)
        return (
            "The rows of the form's line-item (order) table, in table order: a list with one JSON object per row, "
            f'each with exactly the keys {", ".join(self.line_item_fields)} and "" for a cell that is blank in that row. '
            f"Keys: {keys}"
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
        field_type_by_field={
            "advertiser": "short_text",
            "agency": "short_text",
            "contract_num": "identifier",
            "flight_from": "date",
            "flight_to": "date",
            "gross_amount": "quantity",
            "product": "short_text",
            "tv_address": "short_text",
            "property": "identifier",  # call sign, e.g. 'KMSP' -- near-misses are different stations
            "channel": "identifier",  # '9' vs '3' must not partially match
            "program_desc": "short_text",
            "program_start_date": "date",
            "program_end_date": "date",
            "sub_amount": "quantity",
        },
        line_item_fields=["channel", "program_desc", "program_start_date", "program_end_date", "sub_amount"],
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


# The line-item keys the prompt names. One template serves every corpus, so
# load_run_context refuses a corpus whose line_item_fields differ.
PROMPT_LINE_ITEM_FIELDS: list[str] = list(CORPORA["ad-buy-form"].line_item_fields)
_PROMPT_LINE_ITEM_GROUP = CORPORA["ad-buy-form"].line_item_group

EXTRACTION_PROMPT_TEMPLATE = (
    """You are extracting structured fields from a scanned government form.

FIELDS TO EXTRACT: {fields}

DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"

Return a single JSON object mapping each field name to a list of extracted text \
value(s) as they literally appear in the document. Use an empty list for a field \
that is not present. Do not include any other keys."""
    + f"""

If {_PROMPT_LINE_ITEM_GROUP} is among the fields, its value is instead a list of JSON \
objects, one per row of the form's line-item (order) table, in table order. Each \
object has exactly the keys {", ".join(PROMPT_LINE_ITEM_FIELDS)}, with the text as it \
literally appears in that row, or "" for a cell that is blank in that row."""
)


# e.g. "DeepForm-unk_template-train_100-test_215-valid_100-SD_0" -> (100, 0)
_SPLIT_NAME_RE = re.compile(r"train_(\d+)-test_\d+-valid_\d+-SD_(\d+)")


def load_run_context(
    dataset_params: dict, scoring: Optional[dict] = None, data_root: Path = vrdu_loader.DEFAULT_DATA_ROOT
) -> DatasetRunContext:
    """dataset_params: {"corpus": "ad-buy-form" | "registration-form", ...}.
    scoring: the experiment's params.scoring (None => native VRDU matching)."""
    scoring_config = ScoringConfig.from_params(scoring)
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

    _check_line_items(cfg, schema, data_root)

    all_docs = {d.document_id: d for d in vrdu_loader.load_documents(corpus, data_root=data_root)}
    group, sub_fields = cfg.line_item_group, list(cfg.line_item_fields)
    record_fields = {group: sub_fields} if sub_fields else {}
    descriptions = {f: d for f, d in cfg.field_descriptions.items() if f not in sub_fields}
    if sub_fields:
        descriptions[group] = cfg.line_items_description()
    common = dict(
        fields=cfg.extraction_fields,
        all_docs=all_docs,
        windowed_text=lambda doc: vrdu_loader.windowed_text(doc, max_pages=cfg.max_pages, max_chars=cfg.max_chars),
        field_descriptions=descriptions,
        record_fields=record_fields,
    )

    if scoring_config.measure == NATIVE:
        # The 14 VRDU entities are scored one by one, as before: each line-item
        # sub-field's predicted list is that key's non-empty values across the
        # predicted rows, matched against doc.fields like any other field.
        def prediction_for(field_values: dict[str, list], scored_field: str) -> list[str]:
            if scored_field in sub_fields:
                rows = coerce_records(field_values.get(group), sub_fields)
                return [r[scored_field] for r in rows if not is_empty(r[scored_field])]
            return [str(v) for v in (field_values.get(scored_field) or []) if v is not None]

        def score_field(predicted: list[str], ground_truth: list[str], field: str) -> float:
            return float(matching.score_prediction(predicted, ground_truth, match_func_by_field[field]))

        return DatasetRunContext(
            score_field=score_field,
            scored_fields=sorted(match_func_by_field),
            extraction_field_of={f: (group if f in sub_fields else f) for f in match_func_by_field},
            prediction_for=prediction_for,
            **common,
        )

    return DatasetRunContext(**common, **_unified_kwargs(cfg, all_docs, scoring_config, data_root))


def _check_line_items(cfg: VRDUCorpusConfig, schema, data_root: Path) -> None:
    """The line-item shape models are asked for must match both the prompt
    and data.json's entity_appearance_pattern."""
    rerun = f"re-run `uv run data/download_vrdu.py --corpus {cfg.corpus}` to regenerate {data_root}/data.json"
    if cfg.line_item_fields and (
        list(cfg.line_item_fields) != PROMPT_LINE_ITEM_FIELDS or cfg.line_item_group != _PROMPT_LINE_ITEM_GROUP
    ):
        raise ValueError(
            f"CORPORA[{cfg.corpus!r}] line items {cfg.line_item_group}={cfg.line_item_fields} differ from the keys "
            f"EXTRACTION_PROMPT_TEMPLATE names ({_PROMPT_LINE_ITEM_GROUP}={PROMPT_LINE_ITEM_FIELDS}); "
            "make the template per-corpus first."
        )
    if schema.entity_appearance_pattern is None:
        raise ValueError(f"{data_root}/data.json has no entity_appearance_pattern for {cfg.corpus!r}; {rerun}")
    data_line_items = sorted(f for f, p in schema.entity_appearance_pattern.items() if p == "line_item")
    if data_line_items != sorted(cfg.line_item_fields):
        raise ValueError(
            f"CORPORA[{cfg.corpus!r}].line_item_fields={sorted(cfg.line_item_fields)} disagrees with "
            f"data.json's entity_appearance_pattern line_item fields={data_line_items}"
        )


def _unified_kwargs(cfg: VRDUCorpusConfig, all_docs: dict, scoring_config: ScoringConfig, data_root: Path) -> dict:
    """Unified-measure context pieces; fails loud on anything the grouped
    ground truth needs but this config / data.json doesn't have."""
    rerun = f"re-run `uv run data/download_vrdu.py --corpus {cfg.corpus}` to regenerate {data_root}/data.json"
    if not cfg.field_type_by_field:
        raise NotImplementedError(f"CORPORA[{cfg.corpus!r}].field_type_by_field is empty -- the unified measure needs every field typed.")
    stale = [d for d in all_docs.values() if d.field_entities is None or d.line_items is None]
    if stale:
        raise ValueError(f"{len(stale)} document(s) lack grouped ground truth (field_entities/line_items); {rerun}")

    record_fields = {cfg.line_item_group: list(cfg.line_item_fields)} if cfg.line_item_fields else {}
    return unified_context_kwargs(
        fields=cfg.extraction_fields,
        field_types=cfg.field_type_by_field,
        record_fields=record_fields,
        config=scoring_config,
        entity_ground_truth=lambda doc, field: doc.field_entities.get(field, []),
        record_ground_truth=lambda doc, _group: doc.line_items,
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
