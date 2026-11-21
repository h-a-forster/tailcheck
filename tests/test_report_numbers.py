"""REPORT.md figures that must trace to committed results."""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROWS = {
    "strict ESR (v1)": "esr_v1",
    "strict ESR, esback G2''": "esr_v1_esback_g2pp",
    "ESR v3 (one-sided)": "esr_v3",
}


def test_full_profile_esr_size_table_matches_results() -> None:
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    section = report.split("### 3.0 Full profile", 1)[1].split("###", 1)[0]
    df = pd.read_csv(ROOT / "results" / "mc-full" / "size_power.csv")
    size = df[(df["kind"] == "size") & (df["dgp"] == "garch-t5")]
    checked = 0
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in ROWS:
            continue
        alpha = float(cells[1].rstrip("%")) / 100
        for n, cell in zip((500, 1000, 2000, 5000), cells[2:], strict=True):
            rate, lo, hi = (float(x) for x in re.findall(r"[\d.]+", cell))
            row = size[(size["test"] == ROWS[cells[0]]) & (size["n"] == n)]
            row = row[(row["alpha"] - alpha).abs() < 1e-12].iloc[0]
            assert abs(row["rate"] - rate) < 5e-4, (cells[0], alpha, n)
            assert abs(row["lo"] - lo) < 5e-4 or (lo == 0 and row["lo"] < 5e-4), (cells[0], n)
            assert abs(row["hi"] - hi) < 5e-4, (cells[0], alpha, n)
            checked += 1
    assert checked == 24


def test_full_profile_cost_matches_timing() -> None:
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    section = " ".join(report.split("### 3.0 Full profile", 1)[1].split("###", 1)[0].split())
    timing = json.loads((ROOT / "results" / "mc-full" / "timing.json").read_text())
    assert timing["units"] == 12680
    assert f"{timing['wall_hours']:.1f} hours of wall time" in section
    assert f"{timing['cpu_hours']:.1f} CPU-hours" in section
    assert f"ESR {timing['cpu_hours_by_block']['esr']:.1f}" in section


def test_full_profile_christoffersen_ind_oversized_at_n5000_matches_results() -> None:
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    section = " ".join(report.split("### 3.0 Full profile", 1)[1].split("###", 1)[0].split())
    df = pd.read_csv(ROOT / "results" / "mc-full" / "size_power.csv")
    cells = df[
        (df["test"] == "christoffersen_ind")
        & (df["kind"] == "size")
        & (df["n"] == 5000)
        & ((df["alpha"] - 0.025).abs() < 1e-12)
    ]
    assert len(cells) == 3 and (cells["lo"] > 0.05).all()
    lo, hi = 100 * cells["rate"].min(), 100 * cells["rate"].max()
    assert f"oversized ({lo:.1f}-{hi:.1f}%;" in section


def test_full_profile_var_es_prose_matches_results() -> None:
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    section = " ".join(report.split("### 3.0 Full profile", 1)[1].split("###", 1)[0].split())
    df = pd.read_csv(ROOT / "results" / "mc-full" / "size_power.csv")
    size = df[df["kind"] == "size"]
    covers = (size["lo"] <= 0.05) & (size["hi"] >= 0.05)

    def summary(test: str) -> tuple[int, int, str]:
        cells = size[size["test"] == test]
        rng = f"{100 * cells['rate'].min():.1f}-{100 * cells['rate'].max():.1f}%"
        return len(cells), int(covers[cells.index].sum()), rng

    n, cov, rng = summary("as_z2")
    assert f"Z2 covers 5% in all {n} ({rng})" in section and cov == n == 45
    n, cov, rng = summary("as_z1")
    assert f"Z1 in {cov} ({rng})" in section
    n, cov, rng = summary("mcneil_frey")
    assert f"({rng}; {n - cov} of {n} intervals exclude 5%)" in section
    total = len(size)
    assert f"{int(size['distorted'].sum())} of {total} size cells are distorted" in section
    assert f"{int((~covers).sum())} of {total} exclude 5%" in section
    kupiec = size[size["test"] == "kupiec"]
    assert f"Kupiec is {summary('kupiec')[2]}" in section
    big = kupiec[kupiec["n"] >= 1000]
    big_excl = int((~covers[big.index]).sum())
    assert f"excluding 5% in {big_excl} of {len(big)} cells with n >= 1000" in section
    dq = size[size["test"] == "dq"]
    dq5000 = dq[(dq["n"] == 5000) & ((dq["alpha"] - 0.01).abs() < 1e-12)]["rate"]
    assert f"n = 5000 ({100 * dq5000.min():.1f}-{100 * dq5000.max():.1f}%)" in section
    assert f"up to {100 * dq['rate'].max():.1f}%" in section
