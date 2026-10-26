"""CUAD dataset adapter: implements the experiments.config dataset-registry
contract (bench_extract.datasets.base.DatasetRunContext + load_run_context /
load_split_doc_ids / describe_split / training_split_params) on top of
bench_extract.datasets.cuad's loader and cuad_matching's ported scoring rule,
plus the field schema (the 10 non-yes/no CUAD categories) and the extraction prompt template. Python (not
YAML), since this holds a prompt template, not just scalars.

CUAD (https://arxiv.org/abs/2103.06268): 510 contracts, 41 categories,
each labelled with verbatim clause spans (zero, one or many per category). Only the
non-yes/no categories are extracted -- see FIELDS / EXCLUDED_YES_NO_FIELDS.

Loaded by path, not imported as a package -- see
experiments.config.load_dataset_registry. Nothing here touches data/ at import
time (the registry imports every dataset module on startup).

dataset_params (all required, none defaulted here -- they live in the
experiment config):
    valid_source: "train" | "test"  where the 'valid' partition comes from
                                     (see bench_extract.datasets.cuad.resolve_split;
                                     "train" holds out valid_size official-train
                                     contracts and leaves the official test set untouched)
    valid_size:   int | null         only read when valid_source == "train"
    train_size:   int | null         subsample of train (null = all)
    seed:         int                seeds the valid holdout and train subsample
    max_chars:    int | null         extraction window (null = whole contract).
                                     Share of the 10 kept fields' ground-truth spans lying
                                     entirely in the first N characters: 9k 73%, 20k 81%,
                                     30k 86%, 50k 92%, 100k 96%, 150k 99% (front-loaded:
                                     parties/dates/title; median contract is ~33k
                                     characters, max ~338k).
(run_training.py instead takes `train_sizes: list[int|null]` in place of train_size.)
"""

from __future__ import annotations

from pathlib import Path

from bench_extract.datasets import cuad as cuad_loader, cuad_matching
from bench_extract.datasets.base import DatasetRunContext

# field slug -> (CUAD category name, description). Authoritative for which
# fields get extracted; load_run_context cross-checks the slug -> category map
# against data/cuad/data.json as a drift guard, not a source. Descriptions are
# CUAD's own (category_descriptions.csv "Description" column, whitespace
# normalized) -- the questions annotators were given. Category names follow
# evaluate.py's .title() casing, which is what the qa ids carry.
FIELDS: dict[str, tuple[str, str]] = {
    'document_name': ('Document Name', 'The name of the contract'),
    'parties': ('Parties', 'The two or more parties who signed the contract'),
    'agreement_date': ('Agreement Date', 'The date of the contract'),
    'effective_date': ('Effective Date', 'The date when the contract is effective'),
    'expiration_date': ('Expiration Date', "On what date will the contract's initial term expire?"),
    'renewal_term': ('Renewal Term', 'What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.'),
    'notice_period_to_terminate_renewal': ('Notice Period To Terminate Renewal', 'What is the notice period required to terminate renewal?'),
    'governing_law': ('Governing Law', "Which state/country's law governs the interpretation of the contract?"),
    'competitive_restriction_exception': ('Competitive Restriction Exception', 'This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.'),
    'warranty_duration': ('Warranty Duration', 'What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?'),
}

# The other 31 CUAD categories are yes/no review questions ("Is there a ...", "Does the
# contract ...") about whether a clause exists, and are deliberately not extracted. Listed
# (slug -> CUAD category name) so the drift guard in load_run_context still fails loud if
# data.json gains or loses a category: data.json's categories must equal FIELDS | EXCLUDED_YES_NO_FIELDS.
EXCLUDED_YES_NO_FIELDS: dict[str, str] = {
    'most_favored_nation': 'Most Favored Nation',
    'non_compete': 'Non-Compete',
    'exclusivity': 'Exclusivity',
    'no_solicit_of_customers': 'No-Solicit Of Customers',
    'no_solicit_of_employees': 'No-Solicit Of Employees',
    'non_disparagement': 'Non-Disparagement',
    'termination_for_convenience': 'Termination For Convenience',
    'rofr_rofo_rofn': 'Rofr/Rofo/Rofn',
    'change_of_control': 'Change Of Control',
    'anti_assignment': 'Anti-Assignment',
    'revenue_profit_sharing': 'Revenue/Profit Sharing',
    'price_restrictions': 'Price Restrictions',
    'minimum_commitment': 'Minimum Commitment',
    'volume_restriction': 'Volume Restriction',
    'ip_ownership_assignment': 'Ip Ownership Assignment',
    'joint_ip_ownership': 'Joint Ip Ownership',
    'license_grant': 'License Grant',
    'non_transferable_license': 'Non-Transferable License',
    'affiliate_license_licensor': 'Affiliate License-Licensor',
    'affiliate_license_licensee': 'Affiliate License-Licensee',
    'unlimited_all_you_can_eat_license': 'Unlimited/All-You-Can-Eat-License',
    'irrevocable_or_perpetual_license': 'Irrevocable Or Perpetual License',
    'source_code_escrow': 'Source Code Escrow',
    'post_termination_services': 'Post-Termination Services',
    'audit_rights': 'Audit Rights',
    'uncapped_liability': 'Uncapped Liability',
    'cap_on_liability': 'Cap On Liability',
    'liquidated_damages': 'Liquidated Damages',
    'insurance': 'Insurance',
    'covenant_not_to_sue': 'Covenant Not To Sue',
    'third_party_beneficiary': 'Third Party Beneficiary',
}

CATEGORY_BY_FIELD: dict[str, str] = {slug: category for slug, (category, _) in FIELDS.items()}

# CUAD's descriptions are yes/no review questions ("Is there a ..."), but the
# ground truth is the verbatim clause text (e.g. '7th day of September, 1999.'),
# not a yes/no or a normalized value -- so tell the model what to return.
_SPAN_INSTRUCTION = (
    " Return the exact text of the clause(s) or phrase(s) in the contract that answer this, copied "
    "verbatim; return an empty list if the contract has none."
)
FIELD_DESCRIPTIONS: dict[str, str] = {
    slug: (description if description.endswith(("?", ".")) else description + ".") + _SPAN_INSTRUCTION
    for slug, (_, description) in FIELDS.items()
}


EXTRACTION_PROMPT_TEMPLATE = """You are reviewing a legal contract and extracting clauses for a lawyer.

FIELDS TO EXTRACT: {fields}

CONTRACT TEXT:
\"\"\"
{document_text}
\"\"\"

Return a single JSON object mapping each field name to a list of text spans copied \
verbatim from the contract (the relevant clause or phrase, not a paraphrase or a \
normalized value). Use an empty list for a field the contract does not address. \
Do not include any other keys."""


def load_run_context(dataset_params: dict, data_root: Path = cuad_loader.DEFAULT_DATA_ROOT) -> DatasetRunContext:
    """dataset_params: see the module docstring; only max_chars is read here."""
    max_chars = dataset_params["max_chars"]

    # Drift guard, not the source (see FIELDS above).
    data_categories = cuad_loader.load_categories(data_root)
    declared = {**CATEGORY_BY_FIELD, **EXCLUDED_YES_NO_FIELDS}
    if data_categories != declared:
        only_in_data = sorted(set(data_categories) - set(declared))
        only_in_config = sorted(set(declared) - set(data_categories))
        mismatched = sorted(f for f in set(data_categories) & set(declared) if data_categories[f] != declared[f])
        raise ValueError(
            f"FIELDS | EXCLUDED_YES_NO_FIELDS (dataset-configs/cuad.py) disagrees with {data_root}/data.json's categories -- "
            f"only_in_data={only_in_data} only_in_config={only_in_config} mismatched_category={mismatched}"
        )

    all_docs = {d.document_id: d for d in cuad_loader.load_documents(data_root=data_root)}

    def score_field(predicted: list[str], ground_truth: list[str], field: str) -> bool:
        return cuad_matching.score_prediction(predicted, ground_truth, CATEGORY_BY_FIELD[field])

    return DatasetRunContext(
        fields=sorted(FIELDS),
        all_docs=all_docs,
        windowed_text=lambda doc: cuad_loader.windowed_text(doc, max_chars),
        score_field=score_field,
        field_descriptions=dict(FIELD_DESCRIPTIONS),
    )


def load_split_doc_ids(dataset_params: dict, part: str, data_root: Path = cuad_loader.DEFAULT_DATA_ROOT) -> list[str]:
    """part: "train" | "valid". See cuad_loader.resolve_split."""
    if part not in ("train", "valid"):
        raise ValueError(f"part must be 'train' or 'valid', got {part!r}")
    split = cuad_loader.resolve_split(
        cuad_loader.load_official_split(data_root),
        valid_source=dataset_params["valid_source"],
        train_size=dataset_params["train_size"],
        seed=dataset_params["seed"],
        valid_size=dataset_params["valid_size"],
    )
    return split[part]


def describe_split(dataset_params: dict) -> dict:
    """Scalar labels for metrics.json namespacing (train_size is None for
    "all of train"; run_training sweeps should give explicit sizes)."""
    return {"train_size": dataset_params["train_size"], "seed": dataset_params["seed"]}


def training_split_params(dataset_params: dict) -> list[dict]:
    """dataset_params: the usual keys with `train_sizes: list[int|null]` in place of
    `train_size`. One dataset_params dict per train size."""
    base = {k: v for k, v in dataset_params.items() if k != "train_sizes"}
    return [{**base, "train_size": size} for size in dataset_params["train_sizes"]]
