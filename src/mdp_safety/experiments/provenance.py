"""Stable configuration fingerprints and recursive source identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def configuration_digest(value: object) -> str:
    """Fingerprint JSON settings independently of mapping insertion order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def python_source_hashes(directory: Path, relative_to: Path | None = None) -> dict[str, str]:
    """Hash Python sources recursively, excluding bytecode and generated results."""
    base = directory if relative_to is None else relative_to
    return {
        str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.py"))
    }
