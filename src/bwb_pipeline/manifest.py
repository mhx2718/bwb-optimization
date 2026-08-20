"""Strict, machine-readable provenance for a complete public pipeline run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _strict_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strict_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set)):
        values = list(value)
        if isinstance(value, set):
            values = sorted(values, key=str)
        return [_strict_jsonable(item) for item in values]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _strict_jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _strict_jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(strict_json_bytes(value)).hexdigest()


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "status_sha256": None
        if status is None
        else hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def environment_report() -> dict[str, Any]:
    all_distributions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            all_distributions[str(name)] = str(distribution.version)
    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(
            (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "torch",
                "catboost",
                "cma",
                "joblib",
                "PyYAML",
                "matplotlib",
                "seaborn",
            )
        ),
        "all_installed_distributions": dict(sorted(all_distributions.items())),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }
    try:
        import torch
    except ImportError:
        report["torch_runtime"] = None
    else:
        report["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    return report


def hash_existing_files(
    paths: Iterable[str | Path],
    *,
    base_directory: str | Path | None = None,
) -> dict[str, str]:
    """Hash files using portable project-relative keys whenever possible."""

    base = None if base_directory is None else Path(base_directory).resolve()
    result: dict[str, str] = {}
    resolved = sorted(
        {Path(path).resolve() for path in paths}, key=lambda path: str(path)
    )
    for item in resolved:
        if item.is_file():
            digest = sha256_file(item)
            if base is not None:
                try:
                    key = item.relative_to(base).as_posix()
                except ValueError:
                    key = f"external/{item.name}__{digest[:12]}"
            else:
                key = item.as_posix()
            if key in result and result[key] != digest:
                raise RuntimeError(f"Manifest file-key collision for {key!r}.")
            result[key] = digest
    return result


def write_run_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    config: Mapping[str, Any],
    seed_manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    input_files: Iterable[str | Path],
    output_files: Iterable[str | Path],
    stages: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the final manifest only after every output has been flushed."""

    manifest = {
        "manifest_schema": "bwb-reproducible-run-v1",
        "config": _strict_jsonable(config),
        "config_sha256": hash_mapping(dict(config)),
        "seeds": _strict_jsonable(seed_manifest),
        "schema": _strict_jsonable(schema),
        "git": git_state(project_root),
        "environment": environment_report(),
        "inputs": hash_existing_files(input_files, base_directory=project_root),
        "outputs": hash_existing_files(output_files, base_directory=project_root),
        "stages": _strict_jsonable(stages),
        "execution": _strict_jsonable(execution),
    }
    # This digest covers the payload before the self-describing field is added.
    # The exact final file digest is returned to the caller (but cannot be
    # embedded in the file without a self-referential hash).
    manifest["manifest_payload_sha256"] = hash_mapping(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(strict_json_bytes(manifest) + b"\n")
    os.replace(temporary, target)
    returned = dict(manifest)
    returned["manifest_file_sha256"] = sha256_file(target)
    return returned
