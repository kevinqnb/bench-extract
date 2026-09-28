"""Unit tests for OpenAIChatModel against a stubbed OpenAI client -- no
network call, no API key needed. Covers prompt formatting, lenient JSON
parsing (fenced/prefixed output), usage extraction, and per-call error
handling, all of which the profiling table and every extractor strategy
depend on being correct.
"""

from types import SimpleNamespace

from bench_extract.models.base import OpenAIChatModel, _loads_lenient

PROMPT_TEMPLATE = "FIELDS: {fields}\nTEXT: {document_text}"


class FakeChatCompletions:
    def __init__(self, content, usage=None, raise_error=None):
        self._content = content
        self._usage = usage
        self._raise_error = raise_error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raise_error:
            raise self._raise_error
        message = SimpleNamespace(content=self._content)
        usage = SimpleNamespace(**self._usage) if self._usage else None
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _model_with_fake_client(content, usage=None, raise_error=None):
    model = OpenAIChatModel(key="fake", model="fake-model", prompt_template=PROMPT_TEMPLATE)
    fake_completions = FakeChatCompletions(content, usage=usage, raise_error=raise_error)
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    return model, fake_completions


def test_loads_lenient_plain_json():
    assert _loads_lenient('{"a": 1}') == {"a": 1}


def test_loads_lenient_strips_json_fence():
    assert _loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_loads_lenient_extracts_object_from_preamble():
    assert _loads_lenient('Sure, here is the JSON:\n{"a": 1}\nHope that helps!') == {"a": 1}


def test_extract_formats_prompt_and_parses_fields():
    model, fake = _model_with_fake_client(
        '```json\n{"advertiser": ["Acme Corp"], "gross_amount": ["$100.00"]}\n```',
        usage={"prompt_tokens": 42, "completion_tokens": 8, "total_tokens": 50},
    )
    prediction = model.extract("some document text", ["advertiser", "gross_amount"])

    assert fake.last_kwargs["messages"][0]["content"] == "FIELDS: advertiser, gross_amount\nTEXT: some document text"
    assert prediction.field_values == {"advertiser": ["Acme Corp"], "gross_amount": ["$100.00"]}
    assert prediction.usage.prompt_tokens == 42
    assert prediction.usage.completion_tokens == 8
    assert prediction.usage.total_tokens == 50
    assert prediction.usage.error is None
    assert prediction.usage.wall_seconds >= 0


def test_extract_missing_field_in_response_becomes_empty_list():
    model, _ = _model_with_fake_client('{"advertiser": ["Acme Corp"]}')
    prediction = model.extract("text", ["advertiser", "gross_amount"])
    assert prediction.field_values == {"advertiser": ["Acme Corp"], "gross_amount": []}


def test_extract_coerces_non_list_field_value_to_list():
    model, _ = _model_with_fake_client('{"advertiser": "Acme Corp"}')
    prediction = model.extract("text", ["advertiser"])
    assert prediction.field_values == {"advertiser": ["Acme Corp"]}


def test_extract_records_error_without_raising():
    model, _ = _model_with_fake_client(None, raise_error=RuntimeError("endpoint unreachable"))
    prediction = model.extract("text", ["advertiser"])
    assert prediction.field_values == {}
    assert prediction.usage.error == "endpoint unreachable"
