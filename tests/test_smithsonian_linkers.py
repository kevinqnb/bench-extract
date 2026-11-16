"""experiments/run_smithsonian_benchmark.py and bench_extract.linkers, with
stubbed LLM and embedding models -- no network. The LOTUS and Abacus tests
each need their own extra (`--extra lotus` / `--extra abacus`)"""

import base64
import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from bench_extract.linkers.base import (
    IMAGE_ID_COLUMN,
    RECORD_COLUMN,
    RECORD_ID_COLUMN,
    LinkModel,
)
from experiments.config import (
    EXPERIMENT_CONFIGS_DIR,
    MODEL_REGISTRY,
    load_experiment_config,
)
from experiments import run_smithsonian_benchmark as runner

REPO_ROOT = Path(__file__).resolve().parent.parent
LOAD_LINKAGE_TABLES = runner.load_linkage_tables
SMITHSONIAN_CONFIGS_DIR = EXPERIMENT_CONFIGS_DIR / "benchmark" / "smithsonian"
JOIN_CONDITION = "{image} is a photograph of the object in {record}"
N_RECORDS = 4
COLOR_STEP = 50
USAGE = {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102}

def record_text(number: int) -> str:
    return json.dumps({"title": f"object number {number}"})

@pytest.fixture
def sample_dir(tmp_path):
    """Record i is linked to one solid-color image whose red value encodes
    i, so a stub can tell from pixels alone which record an image matches"""
    Image = pytest.importorskip("PIL.Image")
    (tmp_path / "records").mkdir()
    (tmp_path / "media").mkdir()
    rows = []
    for number in range(N_RECORDS):
        record_path = f"records/rec_{number}.json"
        jpg_path = f"media/img_{number}.png"
        (tmp_path / record_path).write_text(record_text(number))
        color = (number * COLOR_STEP, 0, 0)
        Image.new("RGB", (8, 8), color).save(tmp_path / jpg_path)
        rows.append({
            "unit_code": "AAA", "record_id": f"rec_{number}",
            "record_path": record_path, "jpg_path": jpg_path,
        })
    with open(tmp_path / runner.GOLD_PAIRS_NAME, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path

@pytest.fixture
def tables(sample_dir):
    return runner.load_linkage_tables(sample_dir, data_root=sample_dir)

def tables_at(sample_dir):
    """load_linkage_tables for a sample whose files sit beside its csv"""
    return LOAD_LINKAGE_TABLES(sample_dir, data_root=sample_dir)

def image_number(data_url: str) -> int:
    from PIL import Image

    encoded = data_url.split("base64,", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return round(image.getpixel((0, 0))[0] / COLOR_STEP)

def is_match(messages: list[dict]) -> bool:
    """Whether a join prompt's image and record text are a gold pair"""
    image_numbers, text = [], ""
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            text += content
            continue
        for part in content:
            if part["type"] == "image_url":
                image_numbers.append(image_number(part["image_url"]["url"]))
            else:
                text += part["text"]
    record_numbers = re.findall(r"object number (\d+)", text)
    assert len(image_numbers) == 1 and len(record_numbers) == 1
    return image_numbers[0] == int(record_numbers[0])

def link_model() -> LinkModel:
    return LinkModel(
        key="vision", served_name="stub-vision",
        base_url="http://localhost:8000/v1", api_key="EMPTY", max_tokens=64,
    )

def test_load_and_split_by_record(tables):
    assert len(tables.images) == len(tables.records) == N_RECORDS
    assert tables.records[RECORD_COLUMN][0] == record_text(0)
    train, test = runner.split_by_record(tables, 0.5, seed=0)
    assert len(train.records) == len(test.records) == 2
    train_ids = set(train.records[RECORD_ID_COLUMN])
    test_ids = set(test.records[RECORD_ID_COLUMN])
    assert not train_ids & test_ids
    assert {pair[1] for pair in train.gold_pairs} == train_ids
    assert {pair[0] for pair in test.gold_pairs} == set(
        test.images[IMAGE_ID_COLUMN]
    )
    again, _ = runner.split_by_record(tables, 0.5, seed=0)
    assert again.records.equals(train.records)
    nothing, everything = runner.split_by_record(tables, 0.0, seed=0)
    assert nothing.records.empty and nothing.images.empty
    assert everything.gold_pairs == tables.gold_pairs

def test_load_fails_loud_on_missing_file(sample_dir):
    (sample_dir / "media" / "img_0.png").unlink()
    with pytest.raises(SystemExit, match="missing"):
        runner.load_linkage_tables(sample_dir, data_root=sample_dir)

@pytest.mark.parametrize(
    "path",
    sorted(SMITHSONIAN_CONFIGS_DIR.glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_every_smithsonian_config_references_known_models_and_methods(path):
    params = load_experiment_config(path).params
    assert set(params["model_keys"]) <= set(MODEL_REGISTRY)
    assert set(params["methods"]) <= set(runner.LINKER_REGISTRY)
    assert set(params["method_kwargs"]) <= set(params["methods"])
    for placeholder in ("{image}", "{record}"):
        assert placeholder in params["join_condition"]

def test_runner_runs_each_method_in_its_environment(sample_dir, tmp_path):
    config_path = tmp_path / "2026-01-01-stub-01.yaml"
    config_path.write_text(
        "id: 2026-01-01-stub-01\nproject: bench-extract\n"
        "description: stub\nseed: 0\nparams:\n"
        f"  sample_dir: {sample_dir}\n  train_fraction: 0.0\n"
        "  model_keys: []\n  embedding_model: null\n"
        f"  join_condition: \"{JOIN_CONDITION}\"\n"
        "  methods: [lotus, abacus]\n"
    )
    output_root = tmp_path / "results"
    environments = {}
    real_run = subprocess.run

    def child(command, **kwargs):
        if "--method" not in command:  # the manifest's own git calls
            return real_run(command, **kwargs)
        env = kwargs["env"]
        method = command[command.index("--method") + 1]
        environments[method] = (
            command[command.index("--extra") + 1],
            env[runner.UV_ENVIRONMENT_VARIABLE],
        )
        if method == "abacus":
            return mock.Mock(returncode=1)
        method_dir = output_root / "2026-01-01-stub-01" / method
        (method_dir / "artifacts").mkdir(parents=True)
        (method_dir / "metrics.json").write_text('{"lotus_pair_f1": 1.0}')
        (method_dir / "run.json").write_text(
            '{"package_versions": {"lotus-ai": "1.2.4"}}'
        )
        (method_dir / "artifacts" / "pairs.csv").write_text(
            "method,image_id,record_id,correct\nlotus,a,1,True\n"
        )
        return mock.Mock(returncode=0)

    with mock.patch.object(runner, "load_linkage_tables", tables_at), \
            mock.patch.object(runner.subprocess, "run", child), \
            pytest.raises(SystemExit, match="abacus"):
        runner.run_smithsonian_benchmark(
            config_path, output_root_override=output_root
        )
    # the failed method did not stop the other, and each got its own env
    assert environments == {
        "lotus": ("lotus", ".venv-lotus"), "abacus": ("abacus", ".venv"),
    }
    run_dir = output_root / "2026-01-01-stub-01"
    assert json.loads((run_dir / "metrics.json").read_text()) == {
        "lotus_pair_f1": 1.0
    }
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["extra"]["failed_methods"] == ["abacus"]
    assert manifest["extra"]["package_versions_by_method"] == {
        "lotus": {"lotus-ai": "1.2.4"}
    }
    pairs = (run_dir / "artifacts" / "pairs.csv").read_text().splitlines()
    assert pairs == ["method,image_id,record_id,correct", "lotus,a,1,True"]

def test_score_pairs():
    gold = {("a", "1"), ("b", "2")}
    scores = runner.score_pairs({("a", "1"), ("b", "1"), ("c", "2")}, gold)
    assert scores["pair_precision"] == pytest.approx(1 / 3)
    assert scores["pair_recall"] == pytest.approx(1 / 2)
    assert scores["pair_f1"] == pytest.approx(0.4)
    assert scores["n_predicted_pairs"] == 3
    assert runner.score_pairs(set(), gold)["pair_f1"] == 0.0

def test_runner_imports_without_either_library():
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "import experiments.run_smithsonian_benchmark; "
        "assert not {'lotus', 'palimpzest'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)

def test_join_condition_must_name_both_sides():
    pytest.importorskip("lotus")
    from bench_extract.linkers.lotus import LotusLinker

    with pytest.raises(ValueError, match="{record}"):
        LotusLinker(
            [link_model()], None, "{image} matches",
            cascade_args=None, lm_kwargs={},
        )

def test_lotus_cascade_needs_an_embedding_model():
    pytest.importorskip("lotus")
    from bench_extract.linkers.lotus import LotusLinker

    with pytest.raises(ValueError, match="embedding_model"):
        LotusLinker(
            [link_model()], None, JOIN_CONDITION,
            cascade_args={"recall_target": 0.8}, lm_kwargs={},
        )

def test_lotus_linker_joins_every_pair_through_the_llm(tables):
    pytest.importorskip("lotus")
    import litellm

    from bench_extract.linkers.lotus import LotusLinker

    calls = []

    def batch_completion(model, messages, **kwargs):
        calls.append((model, kwargs["api_base"], len(messages)))
        return [
            litellm.ModelResponse(
                choices=[{"message": {
                    "role": "assistant",
                    "content": f"Answer: {is_match(prompt)}",
                }}],
                usage=USAGE,
            )
            for prompt in messages
        ]

    linker = LotusLinker(
        [link_model()], None, JOIN_CONDITION,
        cascade_args=None, lm_kwargs={},
    )
    with mock.patch("lotus.models.lm.batch_completion", batch_completion):
        linker.fit(tables.images, tables.records, tables.gold_pairs)
        result = linker.link(tables.images, tables.records)
    assert set(result.pairs) == tables.gold_pairs
    assert sum(n_prompts for _, _, n_prompts in calls) == N_RECORDS ** 2
    assert {call[:2] for call in calls} == {
        ("hosted_vllm/stub-vision", "http://localhost:8000/v1")
    }
    assert result.prompt_tokens == N_RECORDS ** 2 * USAGE["prompt_tokens"]
    assert linker.fit_metrics == {}

def test_gold_pair_validator_scores_agreement_with_gold():
    pytest.importorskip("palimpzest")
    from bench_extract.linkers.utils import pz_join_support as pjs

    validator = pjs.GoldPairValidator({("img_0", "rec_0")})
    image = {pjs.IMAGE_ID_COLUMN: "img_0", pjs.IMAGE_COLUMN: "x.jpg"}
    match = {pjs.RECORD_ID_COLUMN: "rec_0", pjs.RECORD_COLUMN: "..."}
    other = {pjs.RECORD_ID_COLUMN: "rec_1", pjs.RECORD_COLUMN: "..."}
    assert validator.join_score_fn("c", image, match, True) == 1.0
    assert validator.join_score_fn("c", image, match, False) == 0.0
    assert validator.join_score_fn("c", image, other, False) == 1.0
    assert validator.join_score_fn("c", image, other, True) == 0.0

class StubClip:
    """Stands in for the sentence-transformers CLIP model inside pz: embeds
    an image or a record text as a one-hot vector of its number"""
    seen_types: set = set()

    def __init__(self, model_name):
        pass

    def encode(self, inputs, convert_to_numpy=True):
        import numpy as np

        embeddings = np.zeros((len(inputs), 512))
        for row, value in enumerate(inputs):
            StubClip.seen_types.add(type(value).__name__)
            if isinstance(value, str):
                numbers = re.findall(r"object number (\d+)", value)
                number = int(numbers[0]) if numbers else 500
            else:
                red = value.convert("RGB").getpixel((0, 0))[0]
                number = round(red / COLOR_STEP)
            embeddings[row, number] = 1.0
        return embeddings

def stub_completion(calls: list):
    def completion(model, messages, **kwargs):
        calls.append((model, kwargs.get("api_base")))
        response = mock.MagicMock()
        response.choices[0].message.content = str(is_match(messages)).upper()
        response.usage.model_dump.return_value = dict(USAGE)
        return response

    return completion

def test_abacus_linker_static_join_checks_every_pair(tables):
    pytest.importorskip("palimpzest")
    from bench_extract.linkers.abacus import AbacusLinker

    calls = []
    linker = AbacusLinker(
        [link_model()], None, JOIN_CONDITION,
        optimize=False, policy={"name": "MaxQuality"}, pz_config={"seed": 0},
    )
    with mock.patch("litellm.completion", stub_completion(calls)):
        linker.fit(tables.images[:0], tables.records[:0], set())
        assert not calls and linker.fit_metrics == {}
        result = linker.link(tables.images, tables.records)
    assert set(result.pairs) == tables.gold_pairs
    assert len(calls) == N_RECORDS ** 2
    assert set(calls) == {
        ("hosted_vllm/stub-vision", "http://localhost:8000/v1")
    }
    assert result.prompt_tokens == N_RECORDS ** 2 * USAGE["prompt_tokens"]
    assert "NestedLoopsJoin" in result.plan

def test_abacus_linker_fits_then_joins(tables):
    pytest.importorskip("palimpzest")
    from palimpzest.query.operators import join as pz_join

    from bench_extract.linkers.abacus import AbacusLinker

    calls = []
    completion = stub_completion(calls)
    train, test = runner.split_by_record(tables, 0.5, seed=0)
    linker = AbacusLinker(
        [link_model()], "clip-ViT-B-32", JOIN_CONDITION,
        optimize=True, policy={"name": "MaxQuality"},
        pz_config={"seed": 0, "sample_budget": 4},
    )
    with pytest.raises(RuntimeError, match="fit"):
        linker.link(test.images, test.records)
    linker.pz_config["sample_budget"] = 5
    with pytest.raises(ValueError, match="sample_budget"):
        linker.fit(train.images, train.records, train.gold_pairs)
    linker.pz_config["sample_budget"] = 4
    StubClip.seen_types = set()
    with mock.patch("litellm.completion", completion), \
            mock.patch.object(pz_join, "SentenceTransformer", StubClip), \
            mock.patch.object(pz_join.Locks, "model", None):
        linker.fit(train.images, train.records, train.gold_pairs)
        n_fit_calls = len(calls)
        result = linker.link(test.images, test.records)
    assert linker.fit_metrics["n_train_records"] == 2
    assert linker.fit_metrics["optimization_input_tokens"] > 0
    assert n_fit_calls > 0  # optimization sampled the candidate joins
    assert set(result.pairs) == test.gold_pairs
    assert result.plan
    assert set(calls) == {
        ("hosted_vllm/stub-vision", "http://localhost:8000/v1")
    }
    # an embedding join was among the sampled candidates. pz 1.5.3 embeds an
    # image's path string, never its pixels, so a file name that contains
    # the record id leaks the answer. A pz upgrade that fixes this should
    # fail here
    assert StubClip.seen_types == {"str"}
