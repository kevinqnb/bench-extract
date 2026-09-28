# devlog

A per-session record of **what was asked for** and **what was built or run**, for
AI-assisted development in this repo. Public and curated: anyone who clones the repo
sees this. Deeper rationale, hypotheses, and result assessments (when they exist) are
kept outside this repo, in whichever private notes system a given contributor uses --
this directory only ever holds the shareable subset.

## Two kinds of session, two subfolders

- `devlog/develop/<id>.md` -- a build/development session: code was written or
  changed.
- `devlog/experiment/<id>.md` -- an experiment session: a run was configured and
  launched.

`<id>` is the contract id (`YYYY-MM-DD-slug-NN`, see `CLAUDE.md`) -- the same string
used by `experiments/experiment-configs/{benchmark,training}/<id>.yaml` and the run
directory `experiments/results/{benchmark,training}/<id>/`. A file is only created
for an `<id>` that corresponds to an actual registered build or experiment; a one-off
refactor, dependency bump, or bug fix is covered by its commit message alone and gets
no devlog file.

## What goes in a `develop/<id>.md` file

Frontmatter (`id`, `config` -- the config path if this build has one, omitted
otherwise), then one `## Session <date>` block per working session on that id (never
rewrite earlier blocks):

- **Prompts** -- the instructions the human gave, verbatim or lightly trimmed.
  Substantive ones only; drop typo-fixes and bare approvals.
- **Implemented** -- 3-5 sentences: what changed, which entry points, which configs,
  which tests/checks passed. Pointer-length, not exhaustive.
- **Commits** -- the short hashes and subjects for that session's commits.

## What goes in an `experiment/<id>.md` file

Same frontmatter and per-session shape, adapted to a run instead of a code change:

- **Prompts** -- the experiment requests the human gave that session, verbatim or
  lightly trimmed.
- **Hypothesis** -- the human's general-form hypothesis for the run, kept verbatim
  rather than sharpened or reworded.
- **What was run** -- the config(s) used, entry points invoked, and where the run
  landed (`experiments/results/.../<id>/`, repo-relative).
- **Commits** -- the short hashes and subjects for that session's commits (typically
  the experiment config).

Actual results and any assessment of them are deliberately out of scope for this
file -- it records that the run happened and why, not how it turned out.

## Workflow

Two project skills, checked into `.claude/skills/`, write these files:

- `/developlog <id>` -- run at the end of a build session. Writes or appends
  `devlog/develop/<id>.md`, commits the session's implementation code, records the
  commit hash(es) here, then commits this file on its own
  (`devlog: <id> session <date>`).
- `/experimentlog <id>` -- run at the end of an experiment session, or once a
  submitted job has finished. Commits the experiment config if it isn't already
  committed, writes or appends `devlog/experiment/<id>.md`, then commits this file
  on its own (`explog: <id> session <date>`).

Both are trailing commits: every hash a devlog/explog commit lists already exists,
and it never lists itself.

## This is a public file

Same review bar as a commit message. No secrets or API keys, no absolute cluster
paths (repo-relative only), no unpublished-result specifics, no session identifiers
from any contributor's local tooling. If a prompt contained something that shouldn't
be public, paraphrase it and say so.
