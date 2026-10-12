"""Bounded real-Storm high-dimensional regression, not a timing benchmark.

Run from the repository: PYTHONPATH=src python tests/section3_storm_smoke.py
    --output /new/output/path
Each of 12 runs is limited to four iterations and 60 wall seconds. UNKNOWN
at the iteration cap is permitted, but failed direction checks are not.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from mdp_safety.experiments.execution import execute_command

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    # Only this short-lived test process and its children see these settings.
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    rows = []
    for family, variant in [("network-routing", "safe"), ("robot-mission", "unsafe")]:
        for dimension in [6, 10]:
            model = f"{family}-d{dimension}"
            for rule in ["l2", "linf", "gauge"]:
                output = args.output / f"{model}-{variant}-{rule}"
                command = [sys.executable, "-m", "mdp_safety",
                           str(ROOT / "benchmarks/models" / f"{model}.prism"),
                           str(ROOT / "benchmarks/specs" / f"{model}-{variant}.json"),
                           "--projection", rule, "--max-iterations", "4", "--timeout", "30",
                           "--omit-scheduler-actions", "--output-dir", str(output)]
                try:
                    process, timed_out = execute_command(command, 60)
                    if timed_out:
                        raise RuntimeError("whole-run 60-second limit reached (process group terminated)")
                    if process.returncode not in (0, 2):
                        raise RuntimeError(process.stderr)
                    data = json.loads((output / "result.json").read_text())
                    if "direction selection could not" in data["reason"]:
                        raise RuntimeError(data["reason"])
                    expected = "SAFE" if variant == "safe" else "UNSAFE"
                    if data["status"] not in (expected, "UNKNOWN"):
                        raise RuntimeError(f"unexpected verdict {data['status']}")
                    for selection in data["selections"]:
                        if selection["direction_score"] is None:
                            raise RuntimeError("missing unified score")
                    row = dict(model=model, variant=variant, rule=rule, status=data["status"],
                               reason=data["reason"], queries=data["oracle_call_count"],
                               audited_selections=len(data["selections"]))
                except RuntimeError as error:
                    row = dict(model=model, variant=variant, rule=rule, status="FAILED", reason=str(error))
                rows.append(row)
                print(json.dumps(row), flush=True)
                (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return int(any(row["status"] == "FAILED" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
