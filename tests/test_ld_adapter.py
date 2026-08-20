import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from bwb_pipeline.ld_adapter import (
    GEOMETRY_COLUMNS,
    LDAdapter,
    load_official_predictor,
    normalize_warnings,
)


def geometry(**updates):
    values = {
        "B1/C1": 0.15,
        "B2/C1": 0.12,
        "B3/C1": 0.52,
        "C2/C1": 0.70,
        "C3/C1": 0.23,
        "C4/C1": 0.075,
        "S1": 50.0,
        "S3": 30.0,
        "X3/C1": 0.575,
        "C1": 3000.0,
    }
    values.update(updates)
    return values


def valid_predictor(geom, *, alt_kft, kcas, aoa):
    assert tuple(geom) == GEOMETRY_COLUMNS
    assert (alt_kft, kcas, aoa) == (15.0, 180.0, 6.0)
    return {"LD": 12.2, "CL": 0.2032, "CD": 0.01665, "warnings": []}


def test_valid_prediction_is_accepted():
    result = LDAdapter(valid_predictor).evaluate(
        geometry(), alt_kft=15.0, kcas=180.0, aoa=6.0
    )
    assert result.in_domain is True
    assert result.rejection_reasons == ()
    assert result.evaluation_error is None
    assert result.ld == pytest.approx(12.2)
    assert result.cd > 0.0


@pytest.mark.parametrize(
    "warning_payload, expected",
    [
        ("outside training envelope", ("outside training envelope",)),
        (
            ["range warning", "", None, "range warning", "Mach warning"],
            ("Mach warning", "range warning"),
        ),
        (
            {"code": "OOD", "variable": "C1"},
            ('{"code":"OOD","variable":"C1"}',),
        ),
    ],
)
def test_every_nonempty_warning_is_fail_closed(warning_payload, expected):
    def predictor(*args, **kwargs):
        return {
            "LD": 12.2,
            "CL": 0.2032,
            "CD": 0.01665,
            "warnings": warning_payload,
        }

    result = LDAdapter(predictor).evaluate(
        geometry(), alt_kft=15.0, kcas=180.0, aoa=6.0
    )
    assert result.in_domain is False
    assert result.warnings == expected
    assert result.rejection_reasons == ("predictor_warning",)
    assert result.evaluation_error is None


@pytest.mark.parametrize(
    "key, value, reason",
    [
        ("LD", np.nan, "nonfinite_output"),
        ("CL", np.inf, "nonfinite_output"),
        ("CD", -np.inf, "nonfinite_output"),
        ("CD", 0.0, "nonpositive_cd"),
        ("CD", -0.01, "nonpositive_cd"),
    ],
)
def test_nonfinite_outputs_and_nonpositive_cd_are_rejected(key, value, reason):
    def predictor(*args, **kwargs):
        result = {"LD": 12.2, "CL": 0.2032, "CD": 0.01665, "warnings": []}
        result[key] = value
        return result

    result = LDAdapter(predictor).evaluate(
        geometry(), alt_kft=15.0, kcas=180.0, aoa=6.0
    )
    assert result.in_domain is False
    assert reason in result.rejection_reasons


@pytest.mark.parametrize(
    "raw_result",
    [
        12.2,
        {"LD": 12.2, "CL": 0.2, "CD": 0.02},
        {"LD": "not numeric", "CL": 0.2, "CD": 0.02, "warnings": []},
    ],
)
def test_malformed_results_are_fail_closed(raw_result):
    result = LDAdapter(lambda *args, **kwargs: raw_result).evaluate(
        geometry(), alt_kft=15.0, kcas=180.0, aoa=6.0
    )
    assert result.in_domain is False
    assert result.evaluation_error
    assert result.rejection_reasons == ("evaluation_error",)


def test_predictor_exception_and_invalid_input_are_fail_closed():
    def predictor(*args, **kwargs):
        raise RuntimeError("surrogate failed")

    adapter = LDAdapter(predictor)
    failed = adapter.evaluate(
        geometry(), alt_kft=15.0, kcas=180.0, aoa=6.0
    )
    assert failed.in_domain is False
    assert failed.evaluation_error == "RuntimeError: surrogate failed"

    invalid = adapter.evaluate(
        geometry(C1=np.nan), alt_kft=15.0, kcas=180.0, aoa=6.0
    )
    assert invalid.in_domain is False
    assert "must be finite" in invalid.evaluation_error


def test_dataframe_interface_preserves_index_and_caches_exact_inputs():
    calls = []

    def predictor(geom, *, alt_kft, kcas, aoa):
        calls.append((geom, alt_kft, kcas, aoa))
        return {"LD": 10.0, "CL": 0.2, "CD": 0.02, "warnings": None}

    designs = pd.DataFrame([geometry(), geometry()], index=["a", "b"])
    adapter = LDAdapter(predictor, cache=True)
    result = adapter.predict_many(
        designs,
        {"Altitude": 15.0, "KCAS": 180.0, "AOA": 6.0},
    )
    assert result.index.tolist() == ["a", "b"]
    assert result["ld_in_domain"].tolist() == [True, True]
    assert result["ld_warning_count"].tolist() == [0, 0]
    assert result["warnings"].map(json.loads).tolist() == [[], []]
    assert len(calls) == 1


def test_cache_is_bounded_lru():
    calls = []

    def predictor(geom, **kwargs):
        calls.append(float(geom["C1"]))
        return {"LD": 10.0, "CL": 0.2, "CD": 0.02, "warnings": []}

    adapter = LDAdapter(predictor, cache=True, max_cache_entries=2)
    for c1 in (3000.0, 3001.0, 3002.0, 3000.0):
        adapter.evaluate(
            geometry(C1=c1), alt_kft=15.0, kcas=180.0, aoa=6.0
        )
    assert len(adapter._cache) == 2
    assert calls == [3000.0, 3001.0, 3002.0, 3000.0]


def test_warning_normalization_is_nested_sorted_and_unique():
    warnings = normalize_warnings(["z", ["a", "z"], None, b"b"])
    assert warnings == ("a", "b", "z")


def test_permissive_warning_policy_is_not_supported():
    with pytest.raises(ValueError, match="fail-closed"):
        LDAdapter(valid_predictor, reject_any_warning=False)


def test_official_loader_uses_source_bytes_not_stale_same_mtime_pyc():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "flight_conversion.py").write_text("# owned helper\n")
        regressor = root / "regressor.py"
        regressor.write_text("VALUE = 1.0\n")
        (root / "predict_ld.py").write_text(
            "from regressor import VALUE\n"
            "def predict_ld(geom, alt_kft, kcas, aoa):\n"
            "    return {'LD': VALUE, 'CL': 0.2, 'CD': 0.02, 'warnings': []}\n"
        )
        (root / "reg_full.json").write_text("{}\n")
        first = load_official_predictor(root)
        assert first({}, alt_kft=1, kcas=2, aoa=3)["LD"] == 1.0
        stat = regressor.stat()
        regressor.write_text("VALUE = 2.0\n")  # exactly the same byte length
        os.utime(regressor, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = load_official_predictor(root)
        assert second({}, alt_kft=1, kcas=2, aoa=3)["LD"] == 2.0
