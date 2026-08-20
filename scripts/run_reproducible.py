#!/usr/bin/env python3
"""Launch the pipeline with process-level determinism variables set early."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--master-seed", type=int, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    env = os.environ.copy()
    config_path = (root / args.config).resolve()
    if args.master_seed is None:
        with config_path.open("r", encoding="utf-8") as stream:
            configured_seed = int(yaml.safe_load(stream)["master_seed"])
    else:
        configured_seed = int(args.master_seed)
    python_hash_seed = int.from_bytes(
        hashlib.sha256(
            f"{configured_seed}\0process/pythonhash".encode("utf-8")
        ).digest()[:4],
        byteorder="big",
    )
    env.update(
        {
            "PYTHONHASHSEED": str(python_hash_seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(root / ".matplotlib"),
        }
    )
    command = [
        sys.executable,
        "-m",
        "bwb_pipeline.pipeline",
        "--config",
        str(config_path),
        "--project-root",
        str(root),
    ]
    if args.master_seed is not None:
        command.extend(["--master-seed", str(args.master_seed)])
    if args.force_retrain:
        command.append("--force-retrain")
    if args.models_only:
        command.append("--models-only")
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
