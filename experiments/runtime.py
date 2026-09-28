"""Runtime facts captured fresh at the start of a run: git state and
installed package versions. Companion to experiments/config.py's *intent*
(what was asked to run) -- this module answers "what actually happened,"
snapshotted into run.json (see notes/hub/conventions.md's run-output
contract).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import IO, Optional


def git_sha(repo_root: Path) -> tuple[Optional[str], Optional[bool]]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def installed_versions(packages: list[str]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


@dataclass
class RunManifest:
    id: str
    project: str = "bench-extract"
    config_path: Optional[str] = None
    git_sha: Optional[str] = None
    git_dirty: Optional[bool] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    status: str = "running"
    host: Optional[str] = None
    job_id: Optional[str] = None  # SGE's $JOB_ID when submitted via qsub; None for an interactive run
    package_versions: dict = field(default_factory=dict)
    # Free-form per-runner details (corpus, split, methods, model_keys, ...) --
    # kept out of metrics.json, which must stay flat-scalar per the run-output
    # contract.
    extra: dict = field(default_factory=dict)


class Tee:
    """Writes to every given stream -- used to mirror a run's stdout into
    both the terminal and the run directory's log.txt (see
    notes/hub/conventions.md's run-output contract)."""

    def __init__(self, *streams: IO[str]):
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()
