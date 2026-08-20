"""One-master-seed reproducibility utilities."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SeedRegistry:
    """Derive stable, namespaced child seeds from one public master seed.

    Namespaced streams avoid accidental correlations caused by repeatedly
    resetting every stochastic component to the same integer.  The master seed
    remains the only user-controlled random input.
    """

    master_seed: int
    _derived: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= int(self.master_seed) < 2**63:
            raise ValueError("master_seed must be a non-negative signed 64-bit integer.")
        self.master_seed = int(self.master_seed)

    def derive(
        self,
        namespace: object,
        *keys: object,
        upper_bound: int = 2**31 - 1,
    ) -> int:
        # Variadic keys keep optimizer calls readable while still registering one
        # unambiguous canonical namespace.
        namespace = "/".join(str(item).strip() for item in (namespace, *keys))
        if not namespace:
            raise ValueError("Seed namespace cannot be empty.")
        if upper_bound < 2:
            raise ValueError("upper_bound must be at least two.")
        payload = f"bwb-seed-v1\0{self.master_seed}\0{namespace}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        seed = 1 + int.from_bytes(digest[:8], "big") % (upper_bound - 1)
        previous = self._derived.get(namespace)
        if previous is not None and previous != seed:
            raise RuntimeError(f"Seed namespace changed value: {namespace!r}.")
        self._derived[namespace] = seed
        return seed

    def numpy_rng(self, namespace: str) -> np.random.Generator:
        return np.random.default_rng(self.derive(namespace, upper_bound=2**32))

    def manifest(self) -> dict[str, Any]:
        return {
            "algorithm": "sha256-namespaced-v1",
            "master_seed": self.master_seed,
            "derived_seeds": dict(sorted(self._derived.items())),
        }


def set_global_determinism(
    seed: int,
    *,
    strict: bool = True,
    torch_num_threads: int = 1,
) -> dict[str, Any]:
    """Seed Python, NumPy, and (when installed) PyTorch deterministically."""

    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("Global seed must fit in an unsigned 32-bit integer.")
    if torch_num_threads < 1:
        raise ValueError("torch_num_threads must be positive.")

    # This must be set before CUDA context creation to affect cuBLAS.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    report: dict[str, Any] = {
        "seed": seed,
        "strict": bool(strict),
        "pythonhashseed_at_process_start": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_available": False,
    }
    try:
        import torch
    except ImportError:
        return report

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(torch_num_threads))
    torch.use_deterministic_algorithms(bool(strict), warn_only=not strict)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    report.update(
        {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_num_threads": int(torch.get_num_threads()),
        }
    )
    return report


def seed_torch_dataloader_worker(worker_id: int) -> None:
    """Worker initializer for callers that deliberately use worker processes."""

    del worker_id
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to seed DataLoader workers.") from exc
    worker_seed = int(torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
