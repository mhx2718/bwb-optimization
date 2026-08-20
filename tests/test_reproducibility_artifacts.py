from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bwb_pipeline.artifacts import (
    ArtifactIntegrityError,
    ArtifactManifest,
    IncompatibleArtifactError,
    assert_artifact_compatible,
    assert_artifact_payloads,
    atomic_write_json,
    load_json,
)
from bwb_pipeline.reproducibility import SeedRegistry


class ReproducibilityArtifactTests(unittest.TestCase):
    def test_named_seeds_are_stable_and_namespace_specific(self) -> None:
        first = SeedRegistry(20260819)
        second = SeedRegistry(20260819)
        self.assertEqual(first.derive("forward/init"), second.derive("forward/init"))
        self.assertNotEqual(first.derive("forward/init"), first.derive("stress/model"))
        self.assertEqual(
            first.derive("optimization", "case", 1, "repeat", 0),
            second.derive("optimization/case/1/repeat/0"),
        )
        self.assertEqual(
            list(first.manifest()["derived_seeds"]),
            sorted(first.manifest()["derived_seeds"]),
        )

    def test_artifact_manifest_refuses_feature_change(self) -> None:
        base = ArtifactManifest(
            artifact_type="test",
            dataset_fingerprint="data",
            split_fingerprint="split",
            feature_names=("a", "b"),
            target_names=("y",),
            master_seed=7,
            model_config={"width": 8},
            unit_transforms={},
            package_versions={"numpy": "1"},
        ).with_artifact_id()
        assert_artifact_compatible(base, base)
        changed = ArtifactManifest(
            **{
                **base.compatibility_payload(),
                "feature_names": ("a", "b", "flight"),
            }
        ).with_artifact_id()
        with self.assertRaises(IncompatibleArtifactError):
            assert_artifact_compatible(base, changed)

    def test_atomic_json_is_canonical_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_write_json(path, {"z": 1, "a": [2, 3]})
            self.assertEqual(load_json(path), {"a": [2, 3], "z": 1})
            text = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(text), {"a": [2, 3], "z": 1})
            self.assertTrue(text.startswith('{"a"'))

    def test_payload_hashes_detect_tampering_and_missing_files(self) -> None:
        base = ArtifactManifest(
            artifact_type="test",
            dataset_fingerprint="data",
            split_fingerprint="split",
            feature_names=("x",),
            target_names=("y",),
            master_seed=7,
            model_config={},
            unit_transforms={},
            package_versions={},
        ).with_artifact_id()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "weights.bin"
            payload.write_bytes(b"verified bytes")
            persisted = base.with_payload_hashes(root, (payload.name,))
            assert_artifact_payloads(persisted, root)
            payload.write_bytes(b"tampered bytes")
            with self.assertRaises(ArtifactIntegrityError):
                assert_artifact_payloads(persisted, root)
            payload.unlink()
            with self.assertRaises(ArtifactIntegrityError):
                assert_artifact_payloads(persisted, root)

    def test_legacy_manifest_without_payload_hashes_is_rejected(self) -> None:
        manifest = ArtifactManifest(
            artifact_type="test",
            dataset_fingerprint="data",
            split_fingerprint="split",
            feature_names=("x",),
            target_names=("y",),
            master_seed=7,
            model_config={},
            unit_transforms={},
            package_versions={},
        ).with_artifact_id()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ArtifactIntegrityError):
                assert_artifact_payloads(manifest, directory)


if __name__ == "__main__":
    unittest.main()
