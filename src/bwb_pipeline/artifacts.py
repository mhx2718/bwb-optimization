"""Strict artifact compatibility and atomic persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


ARTIFACT_SCHEMA_VERSION = 2


class ArtifactError(RuntimeError):
    """Base artifact error."""


class IncompatibleArtifactError(ArtifactError):
    """Raised instead of silently loading or overwriting an incompatible model."""


class PartialArtifactError(ArtifactError):
    """Raised when only part of an artifact bundle exists."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when a persisted payload is missing or has the wrong digest."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Non-finite values are forbidden in artifact manifests.")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(paths: tuple[str | Path, ...]) -> str:
    """Hash ordered source files for fail-closed model compatibility."""

    digest = hashlib.sha256()
    resolved = sorted((Path(path).resolve() for path in paths), key=lambda p: p.name)
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot fingerprint missing source file: {path}")
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def installed_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_type: str
    dataset_fingerprint: str
    split_fingerprint: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    master_seed: int
    model_config: Mapping[str, Any]
    unit_transforms: Mapping[str, Any]
    package_versions: Mapping[str, str | None]
    code_version: str = "0.1.0"
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION
    artifact_id: str = ""
    # Relative payload paths are deliberately excluded from artifact identity:
    # identity answers "is this the requested model?", while these digests
    # answer "are the stored bytes exactly the bytes that were trained?".
    payload_hashes: Mapping[str, str] = field(default_factory=dict)

    def compatibility_payload(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_type": self.artifact_type,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "master_seed": self.master_seed,
            "model_config": self.model_config,
            "unit_transforms": self.unit_transforms,
            "package_versions": self.package_versions,
            "code_version": self.code_version,
        }

    def with_artifact_id(self) -> "ArtifactManifest":
        payload = self.compatibility_payload()
        return ArtifactManifest(
            **payload,
            artifact_id=sha256_json(payload)[:16],
            payload_hashes=dict(self.payload_hashes),
        )

    def with_payload_hashes(
        self,
        base_directory: str | Path,
        relative_paths: tuple[str, ...],
    ) -> "ArtifactManifest":
        base = Path(base_directory)
        hashes = {
            str(relative): sha256_file(base / relative)
            for relative in sorted(relative_paths)
        }
        manifest = self if self.artifact_id else self.with_artifact_id()
        values = manifest.compatibility_payload()
        return ArtifactManifest(
            **values,
            artifact_id=manifest.artifact_id,
            payload_hashes=hashes,
        )

    def to_dict(self) -> dict[str, Any]:
        manifest = self if self.artifact_id else self.with_artifact_id()
        return _jsonable(asdict(manifest))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ArtifactManifest":
        converted = dict(values)
        converted["feature_names"] = tuple(converted["feature_names"])
        converted["target_names"] = tuple(converted["target_names"])
        return cls(**converted)


def assert_artifact_compatible(
    actual: ArtifactManifest,
    expected: ArtifactManifest,
) -> None:
    actual_payload = actual.compatibility_payload()
    expected_payload = expected.compatibility_payload()
    if canonical_json_bytes(actual_payload) != canonical_json_bytes(expected_payload):
        differing = sorted(
            key
            for key in set(actual_payload) | set(expected_payload)
            if canonical_json_bytes(actual_payload.get(key))
            != canonical_json_bytes(expected_payload.get(key))
        )
        raise IncompatibleArtifactError(
            "Existing artifact is incompatible with this run. "
            f"Differing fields: {differing}. Use force_retrain=True or a new "
            "versioned output directory; the artifact was not overwritten."
        )
    expected_id = expected.with_artifact_id().artifact_id
    if actual.artifact_id and actual.artifact_id != expected_id:
        raise IncompatibleArtifactError(
            f"Artifact ID mismatch: stored={actual.artifact_id}, expected={expected_id}."
        )


def assert_artifact_payloads(
    manifest: ArtifactManifest,
    base_directory: str | Path,
) -> None:
    """Verify every payload byte before deserializing model code or weights."""

    if not manifest.payload_hashes:
        raise ArtifactIntegrityError(
            "Artifact manifest has no payload hashes. Retrain into a new versioned "
            "directory; legacy unverified weights are never loaded silently."
        )
    base = Path(base_directory).resolve()
    for relative, expected in sorted(manifest.payload_hashes.items()):
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"Artifact payload escapes its directory: {relative!r}."
            ) from exc
        if not candidate.is_file():
            raise ArtifactIntegrityError(
                f"Artifact payload is missing: {candidate}."
            )
        actual = sha256_file(candidate)
        if actual != expected:
            raise ArtifactIntegrityError(
                f"Artifact payload digest mismatch for {relative!r}: "
                f"stored={expected}, actual={actual}."
            )


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace the target on success."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    with atomic_output_path(path) as temporary:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_artifact_manifest(path: str | Path, manifest: ArtifactManifest) -> None:
    atomic_write_json(path, manifest.to_dict())


def load_artifact_manifest(path: str | Path) -> ArtifactManifest:
    return ArtifactManifest.from_dict(load_json(path))


def require_complete_artifact(paths: tuple[str | Path, ...]) -> bool:
    """Return False if all files are absent; raise if only some are present."""

    existence = [Path(path).exists() for path in paths]
    if all(existence):
        return True
    if any(existence):
        missing = [str(path) for path, exists in zip(paths, existence) if not exists]
        raise PartialArtifactError(
            f"Artifact bundle is incomplete; missing files: {missing}."
        )
    return False
