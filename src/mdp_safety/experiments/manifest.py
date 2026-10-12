"""Manifest composition, reproducibility hashing and job ordering."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import BenchmarkInstance, BenchmarkJob, _resolve

JOB_ORDER = "manifest-method-repetition-v1"

def _load_manifest(path: Path, suite_override: str | None, stack: tuple[Path, ...]) -> list[BenchmarkInstance]:
    manifest_path = path.resolve()
    if manifest_path in stack:
        raise ValueError(f"cyclic manifest include: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [
            BenchmarkInstance.from_mapping(
                raw,
                manifest_directory=manifest_path.parent,
                default_suite=suite_override or manifest_path.parent.name,
            )
            for raw in data
        ]
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON array or suite object")
    suite = str(data.get("suite", suite_override or manifest_path.parent.name))
    instances: list[BenchmarkInstance] = []
    inline = data.get("instances", [])
    if not isinstance(inline, list):
        raise ValueError("suite manifest instances must be an array")
    instances.extend(
        BenchmarkInstance.from_mapping(
            raw,
            manifest_directory=manifest_path.parent,
            default_suite=suite,
        )
        for raw in inline
    )
    includes = data.get("include", [])
    if not isinstance(includes, list):
        raise ValueError("suite manifest include must be an array")
    for included in includes:
        included_path = _resolve(manifest_path.parent, str(included))
        instances.extend(
            _load_manifest(included_path, suite, (*stack, manifest_path))
        )
    return instances


def load_manifest(path: Path) -> list[BenchmarkInstance]:
    instances = _load_manifest(path, None, ())
    names = [instance.name for instance in instances]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate benchmark names: {', '.join(duplicates)}")
    return instances


def manifest_digest(path: Path) -> str:
    """Hash a suite manifest and all recursively included manifests."""

    digest = hashlib.sha256()
    visited: set[Path] = set()

    def update(current: Path) -> None:
        resolved = current.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        content = resolved.read_bytes()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        data = json.loads(content)
        if isinstance(data, dict):
            for included in data.get("include", []):
                update(_resolve(resolved.parent, str(included)))

    update(path)
    return digest.hexdigest()


def build_jobs(
    instances: Iterable[BenchmarkInstance],
    repetitions: int,
) -> list[BenchmarkJob]:
    """Use manifest order, then each instance's method order, then repetition."""
    return [
        BenchmarkJob(instance, projection, repetition)
        for instance in instances
        for projection in instance.projections
        for repetition in range(1, repetitions + 1)
    ]
