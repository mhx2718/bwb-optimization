"""Fail-closed adapter for the official lift-to-drag surrogate.

The official scalar predictor returns a mapping containing ``LD``, ``CL``,
``CD``, and ``warnings``.  Optimization must not silently discard the warning
field: a finite-looking prediction can still be outside the surrogate's
supported domain.  This module therefore treats every warning, malformed
result, exception, non-finite output, and non-positive drag coefficient as a
hard domain failure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .schema import GEOMETRY_COLUMNS

REQUIRED_LD_FILES: Tuple[str, ...] = (
    "predict_ld.py",
    "regressor.py",
    "flight_conversion.py",
    "reg_full.json",
)


def _stable_text(value: Any) -> str:
    """Convert an arbitrary warning payload to deterministic text."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, Mapping):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).strip()
    return str(value).strip()


def normalize_warnings(raw_warnings: Any) -> Tuple[str, ...]:
    """Return sorted, unique, non-empty warning messages.

    The official implementation has used both a string and a sequence in
    different contexts.  Normalizing here prevents downstream code from
    accidentally treating a non-empty string as an iterable of characters.
    Nested iterables are flattened.  Sets and dictionaries are serialized in a
    stable order so result files are reproducible.
    """

    normalized = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, bytes, Mapping)):
            text = _stable_text(value)
            if text:
                normalized.append(text)
            return
        if isinstance(value, Iterable):
            for item in value:
                visit(item)
            return
        text = _stable_text(value)
        if text:
            normalized.append(text)

    visit(raw_warnings)
    return tuple(sorted(set(normalized)))


def _messages_json(messages: Sequence[str]) -> str:
    return json.dumps(
        list(messages),
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class LDEvaluation:
    """One validated result from the official L/D surrogate."""

    ld: float
    cl: float
    cd: float
    warnings: Tuple[str, ...]
    evaluation_error: Optional[str]
    rejection_reasons: Tuple[str, ...]
    in_domain: bool

    def to_record(self) -> Dict[str, Any]:
        """Return a CSV/JSON-safe result record."""

        warning_json = _messages_json(self.warnings)
        return {
            "LD": float(self.ld),
            "CL": float(self.cl),
            "CD": float(self.cd),
            # ``warnings`` is retained for compatibility with the official API;
            # ``ld_warnings`` is the unambiguous optimization-table name.
            "warnings": warning_json,
            "ld_warnings": warning_json,
            "ld_warning_count": int(len(self.warnings)),
            "ld_evaluation_error": self.evaluation_error,
            "ld_rejection_reasons": _messages_json(self.rejection_reasons),
            "ld_in_domain": bool(self.in_domain),
        }


def _failed_evaluation(error: str) -> LDEvaluation:
    return LDEvaluation(
        ld=float("nan"),
        cl=float("nan"),
        cd=float("nan"),
        warnings=(),
        evaluation_error=error,
        rejection_reasons=("evaluation_error",),
        in_domain=False,
    )


def _resolve_mission_value(
    mission: Mapping[str, Any],
    preferred: str,
    fallback: str,
) -> float:
    if preferred in mission:
        value = mission[preferred]
    elif fallback in mission:
        value = mission[fallback]
    else:
        raise KeyError(
            f"Mission is missing both {preferred!r} and {fallback!r}."
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Mission value {preferred!r} must be finite.")
    return value


class LDAdapter:
    """Strict adapter around ``predict_ld``.

    Parameters
    ----------
    predictor:
        Callable with the official signature ``predict_ld(geom, alt_kft,
        kcas, aoa)``.
    geometry_columns:
        Exact geometry keys supplied to the predictor.
    reject_any_warning:
        Accepted only for an explicit value of ``True``.  The public pipeline
        intentionally does not support a permissive warning policy.
    cache:
        Memoize exact float64 inputs.  Caching changes performance only; it does
        not alter validation or numeric outputs.
    """

    def __init__(
        self,
        predictor: Callable[..., Mapping[str, Any]],
        geometry_columns: Sequence[str] = GEOMETRY_COLUMNS,
        *,
        reject_any_warning: bool = True,
        cache: bool = True,
        max_cache_entries: int = 50_000,
    ) -> None:
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        if reject_any_warning is not True:
            raise ValueError(
                "The reproducible pipeline only supports fail-closed L/D "
                "warning handling (reject_any_warning=True)."
            )
        columns = tuple(str(column) for column in geometry_columns)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("geometry_columns must be non-empty and unique.")
        self.predictor = predictor
        self.geometry_columns = columns
        self.cache_enabled = bool(cache)
        self.max_cache_entries = int(max_cache_entries)
        if self.cache_enabled and self.max_cache_entries < 1:
            raise ValueError("max_cache_entries must be positive when cache=True.")
        self._cache: "OrderedDict[bytes, LDEvaluation]" = OrderedDict()

    def _validated_geometry(
        self,
        geometry: Union[Mapping[str, Any], pd.Series],
    ) -> Dict[str, float]:
        missing = [
            column for column in self.geometry_columns if column not in geometry
        ]
        if missing:
            raise KeyError(f"Missing L/D geometry columns: {missing}")
        values: Dict[str, float] = {}
        for column in self.geometry_columns:
            value = float(geometry[column])
            if not math.isfinite(value):
                raise ValueError(
                    f"L/D geometry value {column!r} must be finite."
                )
            # Canonicalize signed zero for a stable cache key.
            values[column] = 0.0 if value == 0.0 else value
        return values

    def _cache_key(
        self,
        geometry: Mapping[str, float],
        alt_kft: float,
        kcas: float,
        aoa: float,
    ) -> bytes:
        values = [geometry[column] for column in self.geometry_columns]
        values.extend([alt_kft, kcas, aoa])
        return np.asarray(values, dtype="<f8").tobytes()

    def evaluate(
        self,
        geometry: Union[Mapping[str, Any], pd.Series],
        *,
        alt_kft: float,
        kcas: float,
        aoa: float,
    ) -> LDEvaluation:
        """Evaluate one geometry and return a fail-closed result."""

        try:
            validated_geometry = self._validated_geometry(geometry)
            flight_values = tuple(float(value) for value in (alt_kft, kcas, aoa))
            if not all(math.isfinite(value) for value in flight_values):
                raise ValueError("L/D flight-condition values must be finite.")
        except Exception as exc:
            return _failed_evaluation(f"{type(exc).__name__}: {exc}")

        key = self._cache_key(validated_geometry, *flight_values)
        if self.cache_enabled and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        try:
            raw_result = self.predictor(
                dict(validated_geometry),
                alt_kft=flight_values[0],
                kcas=flight_values[1],
                aoa=flight_values[2],
            )
        except Exception as exc:  # Official predictor may raise library errors.
            result = _failed_evaluation(f"{type(exc).__name__}: {exc}")
        else:
            result = self._validate_result(raw_result)

        if self.cache_enabled:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)
        return result

    @staticmethod
    def _validate_result(raw_result: Any) -> LDEvaluation:
        if not isinstance(raw_result, Mapping):
            return _failed_evaluation(
                "TypeError: predict_ld must return a mapping."
            )

        required = ("LD", "CL", "CD", "warnings")
        missing = [key for key in required if key not in raw_result]
        if missing:
            return _failed_evaluation(
                f"KeyError: predict_ld result is missing keys: {missing}"
            )

        try:
            warnings = normalize_warnings(raw_result["warnings"])
            ld = float(raw_result["LD"])
            cl = float(raw_result["CL"])
            cd = float(raw_result["CD"])
        except Exception as exc:
            return _failed_evaluation(f"{type(exc).__name__}: {exc}")

        rejection_reasons = []
        if not all(math.isfinite(value) for value in (ld, cl, cd)):
            rejection_reasons.append("nonfinite_output")
        if math.isfinite(cd) and cd <= 0.0:
            rejection_reasons.append("nonpositive_cd")
        if warnings:
            rejection_reasons.append("predictor_warning")

        return LDEvaluation(
            ld=ld,
            cl=cl,
            cd=cd,
            warnings=warnings,
            evaluation_error=None,
            rejection_reasons=tuple(rejection_reasons),
            in_domain=not rejection_reasons,
        )

    def evaluate_frame(
        self,
        designs: pd.DataFrame,
        *,
        alt_kft: float,
        kcas: float,
        aoa: float,
    ) -> pd.DataFrame:
        """Evaluate a DataFrame without losing row order or index."""

        if not isinstance(designs, pd.DataFrame):
            raise TypeError("designs must be a pandas DataFrame.")
        missing = [
            column for column in self.geometry_columns if column not in designs
        ]
        if missing:
            raise KeyError(f"Missing L/D geometry columns: {missing}")

        records = [
            self.evaluate(
                row,
                alt_kft=alt_kft,
                kcas=kcas,
                aoa=aoa,
            ).to_record()
            for _, row in designs.loc[:, self.geometry_columns].iterrows()
        ]
        if not records:
            empty_columns = LDEvaluation(
                ld=float("nan"),
                cl=float("nan"),
                cd=float("nan"),
                warnings=(),
                evaluation_error=None,
                rejection_reasons=(),
                in_domain=False,
            ).to_record()
            return pd.DataFrame(columns=empty_columns, index=designs.index)
        return pd.DataFrame.from_records(records, index=designs.index)

    def predict_many(
        self,
        designs: pd.DataFrame,
        mission: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Mission-dictionary interface used by the optimizer evaluator."""

        alt_kft = _resolve_mission_value(mission, "Altitude", "alt_kft")
        kcas = _resolve_mission_value(mission, "KCAS", "kcas")
        aoa = _resolve_mission_value(mission, "AOA", "aoa")
        return self.evaluate_frame(
            designs,
            alt_kft=alt_kft,
            kcas=kcas,
            aoa=aoa,
        )


def load_official_predictor(model_dir: Union[str, Path]) -> Callable[..., Any]:
    """Load ``predict_ld`` from an official surrogate directory.

    All expected companion files are checked before importing.  The model
    directory is placed on ``sys.path`` because the official script imports its
    sibling modules by name.
    """

    directory = Path(model_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_LD_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing official L/D surrogate files in {directory}: {missing}"
        )

    script_path = directory / "predict_ld.py"
    content_digest = hashlib.sha256()
    for name in REQUIRED_LD_FILES:
        content_digest.update(name.encode("utf-8") + b"\0")
        content_digest.update((directory / name).read_bytes())
        content_digest.update(b"\0")
    # Content-address the module, not merely its directory. A notebook rerun
    # after an official LD file changes must never execute stale code while the
    # manifest records the new bytes.
    module_suffix = content_digest.hexdigest()[:16]
    module_name = f"_bwb_official_predict_ld_{module_suffix}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        # Execute these three owned modules from their exact source bytes.
        # SourceFileLoader may legally reuse timestamp/size-matched .pyc files;
        # that would let a same-size, same-mtime source edit evade the content
        # hash in a long-lived notebook kernel.
        def execute_source(name: str, path: Path) -> Any:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None:
                raise ImportError(f"Cannot create module spec for {path}")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[name] = loaded
            code = compile(path.read_bytes(), str(path), "exec")
            exec(code, loaded.__dict__)
            return loaded

        owned_names = ("flight_conversion", "regressor", module_name)
        for owned_name in owned_names:
            sys.modules.pop(owned_name, None)
        inserted = False
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
            inserted = True
        try:
            execute_source("flight_conversion", directory / "flight_conversion.py")
            execute_source("regressor", directory / "regressor.py")
            module = execute_source(module_name, script_path)
        except Exception:
            for owned_name in owned_names:
                sys.modules.pop(owned_name, None)
            raise
        finally:
            if inserted:
                sys.path.remove(str(directory))

    predictor = getattr(module, "predict_ld", None)
    if not callable(predictor):
        raise ImportError(f"{script_path} does not define callable predict_ld.")
    return predictor


def make_ld_adapter(
    model_dir: Union[str, Path],
    *,
    cache: bool = True,
    max_cache_entries: int = 50_000,
) -> LDAdapter:
    """Construct the strict adapter from an official model directory."""

    return LDAdapter(
        load_official_predictor(model_dir),
        cache=cache,
        max_cache_entries=max_cache_entries,
    )
