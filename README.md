# bench-extract

[![attested by humans](https://github.com/kevinqnb/bench-extract/actions/workflows/git-signoff.yml/badge.svg)](https://github.com/kevinqnb/bench-extract/actions/workflows/git-signoff.yml)

Benchmarking routing, cascade, and scheduling strategies for information
extraction -- methods that take a collection of extraction models (LLMs or
smaller NLP models) and decide, per document or per field, which model(s) to
call, trading accuracy against efficiency.

## Data

[VRDU](https://arxiv.org/abs/2211.15421) ("A Benchmark for Structured
Extractions from Complex Documents", Wang et al.): FCC political ad-buy forms
and FARA foreign-agent registration forms, with OCR text and human-annotated
field spans.

- Original dataset: https://github.com/google-research-datasets/vrdu
- Original evaluator (match functions ported into
  `src/bench_extract/datasets/matching.py`, cited there):
  https://github.com/google-research/google-research/tree/master/vrdu

Run `uv run data/download_vrdu.py` to download and reshape it into
`data/vrdu/` (gitignored; see `data/README.md`).

## Methods

Benchmarks strategies from the following papers, each via its own adapter in
`src/bench_extract/extractors/`:

| Type | Method | Paper | Code |
|---|---|---|---|
| Routing | HybridLLM | https://openreview.net/pdf?id=02f3mUtqnM | https://github.com/ulab-uiuc/LLMRouter |
| Routing | RouteLLM | https://arxiv.org/pdf/2406.18665 | https://github.com/ulab-uiuc/LLMRouter |
| Cascade | FrugalGPT | https://arxiv.org/abs/2305.05176 | https://github.com/stanford-futuredata/FrugalGPT |
| Cascade | BARGAIN | https://arxiv.org/abs/2509.02896 | https://github.com/ucbepic/BARGAIN |
| Cascade | Task Cascades | https://arxiv.org/pdf/2601.05536 | https://github.com/ucbepic/task-cascades |
| Cascade | Automix | https://arxiv.org/abs/2310.12963 | https://github.com/ulab-uiuc/LLMRouter |
| Scheduling | Abacus (Palimpzest) | https://arxiv.org/abs/2505.14661 | https://github.com/mitdbg/palimpzest |
| Scheduling | Doctopus | https://dl.acm.org/doi/10.1145/3722212.3725103 | https://github.com/mutong184/Doctopus |
| Unified | Cascade-routing | https://proceedings.mlr.press/v267/dekoninck25a.html | https://github.com/eth-sri/cascade-routing |

Every method fits against the same profiling table -- see `CLAUDE.md` for the
design and current implementation status.

## Skills

Project skills in `.claude/skills/`, available to anyone who clones this repo:

- **`/developlog [id]`** -- run after a build/development session, to leave a durable
  record of what was asked for and what was built: the prompts that drove the
  session and a pointer-length summary of the implementation, paired with the
  commit(s) that carry it. Writes `devlog/develop/<id>.md`.
- **`/experimentlog [id]`** -- run after an experiment session, or once a submitted
  job finishes, to record what was asked for and what was run: the experiment
  prompts, the hypothesis behind the run (kept verbatim), and the config/entry point
  used, paired with the commit that carries the config. Writes
  `devlog/experiment/<id>.md`.
- **`/git-signoff`** -- run before opening a PR, to maintain human ownership over the
  code: a Socratic interview verifying the reviewer actually understands the diff
  and explicitly accepts its risks and trade-offs, recorded as a signed attestation
  commit.

See `devlog/README.md` for the devlog/experimentlog file format and
`.claude/skills/git-signoff/SKILL.md` for the signoff protocol.
