"""Plan or explicitly launch the paper study; dry-run is the default."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys

from mdp_safety.benchmarking import BenchmarkSettings, load_manifest, run_benchmark
from mdp_safety.benchmark_report import SuiteArtifacts, render_report
from mdp_safety.experiments.checkpoint import _validate_resume
from mdp_safety.experiments.manifest import manifest_digest
from mdp_safety.experiments.provenance import configuration_digest, python_source_hashes
from mdp_safety.timeouts import format_timeout, validate_timeout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "experiments/paper.json"


@dataclass(frozen=True)
class PlannedSuite:
    key: str
    label: str
    settings: BenchmarkSettings
    instances: int

    @property
    def jobs(self) -> int:
        return self.instances * len(self.settings.projection_override or ()) * self.settings.repetitions


def plan(root: Path, output: Path, profile: dict, storm: str, resume: bool) -> list[PlannedSuite]:
    """Read the inputs and validate settings, without writing files or running Storm."""
    if profile.get("schema_version") != 2:
        raise ValueError("unsupported study profile schema_version; use a version 2 fixed-order profile")
    repetitions = profile["repetitions"]
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    for field in ("run_timeout_seconds", "storm_timeout_seconds"):
        validate_timeout(profile[field], field)
    suites = []
    for entry in profile["suites"]:
        key = entry["key"]
        if not key or Path(key).name != key or key in (".", "..") or key in {s.key for s in suites}:
            raise ValueError("suite keys must be unique directory names")
        settings = BenchmarkSettings(
            manifest=(root / entry["manifest"]).resolve(), output_directory=output / key,
            projection_override=tuple(profile["methods"]), repetitions=repetitions,
            run_timeout_seconds=profile["run_timeout_seconds"],
            storm_timeout_seconds=profile["storm_timeout_seconds"], storm=storm,
            precision=profile["precision"], minimum_precision=entry["minimum_precision"],
            geometry_tolerance=profile["geometry_tolerance"], max_iterations=profile["max_iterations"],
            sandwich_precision=profile["sandwich_precision"],
            resume=resume,
        )
        settings.validate()
        instances = load_manifest(settings.manifest)
        if not instances:
            raise ValueError(f"empty suite: {key}")
        for instance in instances:
            for path in (instance.model, instance.specification):
                if not path.is_file():
                    raise ValueError(f"missing benchmark input: {path}")
        suites.append(PlannedSuite(key, entry["label"], settings, len(instances)))
    if not suites:
        raise ValueError("the study must contain at least one suite")
    return suites


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def provenance(root: Path, profile: dict, suites: list[PlannedSuite]) -> dict:
    """Pin code, inputs and the Storm binary; do not run its date-dependent banner."""
    inputs = {}
    for suite in suites:
        for instance in load_manifest(suite.settings.manifest):
            for path in (instance.model, instance.specification):
                inputs[str(path)] = digest(path)
    storm = shutil.which(suites[0].settings.storm)
    if storm is None:
        raise ValueError(f"Storm executable not found: {suites[0].settings.storm}")
    return {
        "profile": profile,
        "profile_sha256": configuration_digest(profile),
        "manifest_sha256": {s.key: manifest_digest(s.settings.manifest) for s in suites},
        "input_sha256": inputs,
        "code_sha256": python_source_hashes(root / "src/mdp_safety", relative_to=root),
        "runner_sha256": digest(Path(__file__)),
        "storm": str(Path(storm).resolve()), "storm_sha256": digest(Path(storm)),
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "stormpy")},
        "execution": "sequential jobs in manifest, method, repetition order",
    }


def prepare(root: Path, output: Path, profile: dict, suites: list[PlannedSuite], resume: bool) -> None:
    path = output / "study_provenance.json"
    if resume:
        if not path.is_file():
            raise ValueError("no matching study provenance; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output is not empty; use --resume or a new output directory")
    current = provenance(root, profile, suites)
    if resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(key for key in current.keys() | previous.keys()
                         if current.get(key) != previous.get(key))
        if changed:
            raise ValueError("cannot resume with changed study: " + ", ".join(changed))
        # Validate all suites before spending time on any of them. Unstarted suites are OK.
        for suite in suites:
            directory = suite.settings.output_directory
            if directory.exists() and any(directory.iterdir()) and not (directory / "benchmark_config.json").is_file():
                raise ValueError(f"missing checkpoint in nonempty suite: {suite.key}")
            _validate_resume(suite.settings)
    else:
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / ".study_provenance.json.tmp"
        temporary.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def report(output: Path, suites: list[PlannedSuite], profile: dict) -> None:
    artifacts = [SuiteArtifacts.load(s.label, s.settings.output_directory) for s in suites
                 if (s.settings.output_directory / "aggregate.csv").exists()]
    title = (f"{profile.get('title', 'Paper study')} (max {profile['max_iterations']} refinements, "
             f"run timeout: {format_timeout(profile['run_timeout_seconds'])}, "
             f"Storm-call timeout: {format_timeout(profile['storm_timeout_seconds'])}, "
             f"{profile['repetitions']} repetitions)")
    (output / "full_report.md").write_text(render_report(artifacts, title), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--profile", type=Path, help="custom study profile (default: paper.json)")
    parser.add_argument("--output", type=Path, help="result directory (relative paths use --root)")
    parser.add_argument("--storm", default="storm")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="actually execute experiments")
    mode.add_argument("--dry-run", action="store_true", help="show plan only (the default)")
    parser.add_argument("--resume", action="store_true", help="with --run, skip completed jobs")
    args = parser.parse_args(argv)
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    try:
        root = args.root.resolve()
        profile_path = args.profile or (DEFAULT_PROFILE if root == ROOT else root / "experiments/paper.json")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        output = (root / (args.output or profile.get(
            "output_directory", "results/paper"))).resolve()
        suites = plan(root, output, profile, args.storm, args.resume)
        print(f"Methods: {', '.join(profile['methods'])}")
        print(f"Repetitions: {profile['repetitions']}; max refinements: {profile['max_iterations']}; "
              f"run timeout: {format_timeout(profile['run_timeout_seconds'])}; "
              f"Storm-call timeout: {format_timeout(profile['storm_timeout_seconds'])}")
        print("Order: benchmark manifest -> method list -> repetition")
        for suite in suites:
            print(f"  {suite.key}: {suite.instances} instances, {suite.jobs} runs")
        print(f"Total: {sum(s.jobs for s in suites)} runs; output: {output}", flush=True)
        if not args.run:
            print("Plan only: no experiments started and no output files created. Add --run to start.")
            return 0
        prepare(root, output, profile, suites, args.resume)
        failures = False
        for suite in suites:
            print(f"Starting suite: {suite.key}", flush=True)
            # A TIMEOUT/ERROR is a recorded outcome, not a reason to omit later suites.
            failures = bool(run_benchmark(suite.settings)) or failures
            report(output, suites, profile)
        print(f"All suites finished: {output}", flush=True)
        return int(failures)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
