"""Summarise unit timings of a Monte Carlo run from its progress.log.

Usage: uv run python scripts/mc_timing.py runs/mc-full results/mc-full/timing.json

Each progress.log line is `<iso time> | <unit key> | <seconds> | <done>/<total>`; the seconds
are the worker's time for that unit. CPU-hours sum them per block (first key field); wall
time runs from the first unit's start (its finish time minus its seconds) to the last finish.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


def summarise(log: Path, command: str) -> dict:
    blocks: dict[str, float] = {}
    starts: list[datetime] = []
    ends: list[datetime] = []
    units: set[str] = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        stamp, key, secs = parts[0], "|".join(parts[1:-2]), float(parts[-2])
        end = datetime.fromisoformat(stamp)
        units.add(key)
        blocks[key.split("|")[0]] = blocks.get(key.split("|")[0], 0.0) + secs
        ends.append(end)
        starts.append(end - timedelta(seconds=secs))
    total = sum(blocks.values())
    return {
        "source": f"{log.parent.name}/{log.name}",
        "command": command,
        "units": len(units),
        "cpu_hours": round(total / 3600, 2),
        "cpu_hours_by_block": {k: round(v / 3600, 2) for k, v in sorted(blocks.items())},
        "wall_hours": round((max(ends) - min(starts)).total_seconds() / 3600, 2),
        "first_start": min(starts).isoformat(),
        "last_end": max(ends).isoformat(),
    }


def main() -> None:
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    command = f"uv run python scripts/mc_timing.py {run_dir.as_posix()} {out.as_posix()}"
    summary = summarise(run_dir / "progress.log", command)
    out.write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
