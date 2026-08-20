from __future__ import annotations

import json

from bwb_pipeline.manifest import hash_mapping, strict_json_bytes


def test_strict_json_is_order_independent_and_nonfinite_safe() -> None:
    first = {"b": float("inf"), "a": [2, 1]}
    second = {"a": [2, 1], "b": float("inf")}
    assert hash_mapping(first) == hash_mapping(second)
    payload = json.loads(strict_json_bytes(first))
    assert payload == {"a": [2, 1], "b": None}

