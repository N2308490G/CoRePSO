"""Recompute Table III statistics from completed records; no optimizer runs."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from learned_statistics import learned_significance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "author_cec_correct_segments_ordered_float64_v1"
LEARNED = ["DEDQN", "SYMBOL", "GLEET", "RLHPSDE", "RLEPSO"]


def read_json(name):
    with gzip.open(ROOT / "results" / name, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def table_iii():
    records = read_json("learned_endpoints.json.gz")
    expected = {(m, f, r) for m in LEARNED for f in range(1, 13)
                for r in range(970701, 970712)}
    keys = [(r["method"], r["fid"], r["replicate"]) for r in records]
    if len(keys) != 660 or set(keys) != expected:
        raise ValueError("Learned record grid must contain exactly 660 unique entries")
    values = defaultdict(list)
    for row in records:
        if not (
            row["verified"] and row["dim"] == 20 and row["budget"] == 24800000
            and 0 < row["actual_fes"] <= row["budget"]
            and row["unused_fes"] == row["budget"] - row["actual_fes"]
            and row["objective_protocol"] == PROTOCOL
            and np.isfinite(row["verified_error"])
        ):
            raise ValueError("Invalid learned record")
        values[row["fid"], row["method"]].append(row["verified_error"])
    endpoints = read_json("fe_endpoints.json.gz")
    selected = [r for r in endpoints if r["group"] == "cec2022" and r["dim"] == 20
                and r["method"] in ("original", "cvbr_hybrid")]
    expected_fe = {(m, f"F{f}", r) for m in ("original", "cvbr_hybrid")
                   for f in range(1, 13) for r in range(960701, 960712)}
    if len(selected) != 264 or {(r["method"], r["name"], r["replicate"])
                               for r in selected} != expected_fe:
        raise ValueError("Reused FE endpoint grid is incomplete or duplicated")
    for row in sorted(selected, key=lambda x: x["replicate"]):
        if row["budget"] != 24800000 or not np.isfinite(row["verified_error"]):
            raise ValueError("Invalid FE endpoint")
        method = "CoRePSO" if row["method"] == "cvbr_hybrid" else "AutoPSO"
        values[int(row["name"][1:]), method].append(row["verified_error"])
    tests = learned_significance(values)
    with (ROOT / "results/learned_significance.csv").open(newline="", encoding="utf-8") as f:
        recorded = list(csv.DictReader(f))
    if len(recorded) != len(tests):
        raise ValueError("Recorded table has a different number of comparisons")
    for actual, saved in zip(tests, recorded):
        for key in ("fid", "method", "test", "direction", "significance"):
            if str(actual[key]) != saved[key]:
                raise ValueError(f"Recorded {key} differs: {actual}")
        for key in ("p_raw", "p_holm"):
            if not np.isclose(actual[key], float(saved[key]), atol=1e-13, rtol=1e-11):
                raise ValueError(f"Recorded probability differs: {actual}")
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new CSV path")
    args = parser.parse_args()
    tests = table_iii()
    print("Table III: 660 learned runs + 264 reused FE endpoints; 72 joint Holm tests.")
    for method in ["AutoPSO"] + LEARNED:
        count = Counter(r["significance"] for r in tests if r["method"] == method)
        print(f"{method}: W/N/L = {count['win']}/{count['not_rejected']}/{count['loss']}")
    print("All directions, decisions and probabilities match the manuscript CSV.")
    if args.output:
        if args.output.exists():
            raise FileExistsError("Choose a new output file; recorded results are preserved")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tests[0]))
            writer.writeheader()
            writer.writerows(tests)


if __name__ == "__main__":
    main()
