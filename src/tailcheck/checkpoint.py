"""Resumable unit store: one file per finished unit, atomic writes, skip-if-done.

Every long computation (Monte Carlo cells, study asset x model x alpha units) writes each
finished unit through this module and appends a line to a progress log. A rerun with the
same configuration skips completed units; a changed configuration hash refuses to mix.
Milestone M1.
"""

from __future__ import annotations

from pathlib import Path


class UnitStore:
    """Directory of completed units keyed by a string id."""

    def __init__(self, root: Path, config_hash: str) -> None:
        self.root = root
        self.config_hash = config_hash

    def done(self, unit_id: str) -> bool:
        raise NotImplementedError("M1")

    def write(self, unit_id: str, payload: dict[str, object]) -> Path:
        raise NotImplementedError("M1")

    def read(self, unit_id: str) -> dict[str, object]:
        raise NotImplementedError("M1")
