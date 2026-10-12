"""Read/copy the three fixed domain families; never regenerate them."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPOSITORY_ROOT / "benchmarks"
METADATA_ROOT = REPOSITORY_ROOT / "experiments/metadata"
MANIFEST_ROOT = REPOSITORY_ROOT / "experiments/manifests"
FIXED_DOMAINS = ("gridworld", "robot-mission", "network-routing")
FIXED_DIMENSIONS = (4, 6, 8, 10)


def fixed_metadata(domain: str | None = None) -> list[dict]:
    if domain is not None and domain not in FIXED_DOMAINS:
        raise ValueError(f"not a fixed-input domain: {domain}")
    name = "model_metadata.json"
    return [item for item in json.loads((METADATA_ROOT / name).read_text(encoding="utf-8"))
            if item["domain"] in FIXED_DOMAINS and (domain is None or item["domain"] == domain)]


def copy_fixed_inputs(output: Path, *sections: str,
                      domains: tuple[str, ...] = FIXED_DOMAINS) -> None:
    """Copy exact existing bytes when assembling another directory.

    In-place use is read-only. A missing source fails rather than invoking a
    generator or silently substituting another model. This performs no model
    construction, specification generation or recalibration.
    """
    sources = []
    if any(domain not in FIXED_DOMAINS for domain in domains):
        raise ValueError(f"not fixed-input domains: {domains}")
    for section in sections:
        if section not in {"models", "specs"}:
            raise ValueError(f"unknown fixed input section: {section}")
        for domain in domains:
            for dimension in FIXED_DIMENSIONS:
                names = ([f"{domain}-d{dimension}.prism"] if section == "models" else
                         [f"{domain}-d{dimension}-{status}.json" for status in ("safe", "unsafe")])
                sources.extend(INPUT_ROOT / section / name for name in names)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"missing fixed benchmark input: {source}")
    for source in sources:
        destination = output / source.relative_to(INPUT_ROOT)
        if destination.resolve() == source.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
