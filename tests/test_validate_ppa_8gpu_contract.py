from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


CHECKER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tools"
    / "validate_ppa_8gpu_contract.py"
)
SPEC = importlib.util.spec_from_file_location("ppa_contract_checker_test", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CacheContractTest(unittest.TestCase):
    @staticmethod
    def _write_complete_dataset_clip(clip_dir: Path) -> None:
        (clip_dir / "chunks").mkdir(parents=True)
        (clip_dir / "meta.json").write_text(
            '{"num_frames": 33}', encoding="utf-8"
        )
        (clip_dir / "chunks" / "chunk_000000.npz").write_bytes(b"fixture")

    def test_formal_eight_gpu_batch_contract_is_fixed(self) -> None:
        self.assertEqual(CHECKER.EXPECTED_WORLD_SIZE, 8)
        self.assertEqual(CHECKER.EXPECTED_PER_RANK_BATCH, 1)
        self.assertEqual(CHECKER.EXPECTED_GRAD_ACCUMULATION, 1)
        self.assertEqual(CHECKER.EXPECTED_EFFECTIVE_GLOBAL_BATCH, 8)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.cache = self.root / "cache"
        # train_scene hashes to bucket 46; scene_002 hashes to bucket 2.
        # These are direct children of the formal R2R scene root.
        self.keys = ("train_scene/clip_000001", "scene_002/clip_000002")
        for key in self.keys:
            scene, clip = key.split("/")
            self._write_complete_dataset_clip(self.data / scene / clip)
            clip_cache = self.cache / key
            clip_cache.mkdir(parents=True)
            (clip_cache / CHECKER.CACHE_FILE).write_bytes(b"not-read-by-gate")
            manifest = {
                "schema": CHECKER.CACHE_SCHEMA,
                "clip_key": key,
                "causal": True,
                "num_history": 8,
                "min_history": 5,
                "pose_convention": CHECKER.POSE_CONVENTION,
                "history_pose_convention": CHECKER.HISTORY_POSE_CONVENTION,
                "pose_provider": "amb3r_vo_da3",
                "per_episode_gt_scale_used": False,
                "gt_pose_read_by_exporter": False,
                "endpoint_only": True,
                "row_policy": CHECKER.ROW_POLICY,
                "query_only_at_map_endpoints": True,
                "query_every_frame_from_min_history": False,
                "query_every_frame": False,
                "snapshot_timing": CHECKER.SNAPSHOT_TIMING,
                "future_pose_revisions_used": False,
                "translation_scale": 1.0,
                "frame_count": 33,
                "query_rows": 3,
                "map_init_window": 20,
                "map_every": 8,
            }
            (clip_cache / CHECKER.CACHE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        # The dataset/checker ignore directories with no direct clip_* child.
        (self.data / "annotations").mkdir(parents=True)
        (self.data / "annotations" / "metadata.json").write_text(
            "{}", encoding="utf-8"
        )
        control = self.cache / "_control"
        control.mkdir(parents=True)
        self.ready_path = control / "cache.ready.json"
        self.ready = {
            "schema": CHECKER.READY_SCHEMA,
            "cache_root": str(self.cache.resolve()),
            "dataset_root": str(self.data.resolve()),
            "complete": True,
            "causal": True,
            "endpoint_only": True,
            "failures": 0,
            "clips_total": 2,
            "frames_total": 66,
            "query_rows_total": 6,
            "num_history": 8,
            "min_history": 5,
            "map_init_window": 20,
            "map_every": 8,
            "pose_convention": CHECKER.POSE_CONVENTION,
            "history_pose_convention": CHECKER.HISTORY_POSE_CONVENTION,
            "pose_provider": "amb3r_vo_da3",
            "translation_scale": 1.0,
            "query_only_at_map_endpoints": True,
            "query_every_frame_from_min_history": False,
            "query_every_frame": False,
            "row_policy": CHECKER.ROW_POLICY,
            "snapshot_timing": CHECKER.SNAPSHOT_TIMING,
            "future_pose_revisions_used": False,
            "splits": ["train", "val"],
        }
        self._write_ready()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_ready(self) -> None:
        self.ready_path.write_text(json.dumps(self.ready), encoding="utf-8")

    def test_complete_cache_passes(self) -> None:
        result = CHECKER.validate_cache(
            str(self.cache), str(self.data), ("train", "val")
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["clips"], 2)
        self.assertFalse(result["gt_pose_fallback_allowed"])

    def test_flat_scene_autosplit_matches_dataset_and_ignores_non_scene(self) -> None:
        empty_placeholder = self.data / "train_scene" / "clip_999999"
        (empty_placeholder / "chunks").mkdir(parents=True)
        self.assertEqual(
            CHECKER._discover_dataset_clip_keys(self.data, ("train", "val")),
            self.keys,
        )

    def test_partially_populated_clip_is_rejected(self) -> None:
        partial = self.data / "train_scene" / "clip_999998"
        partial.mkdir(parents=True)
        (partial / "meta.json").write_text('{"num_frames": 33}', encoding="utf-8")
        with self.assertRaisesRegex(CHECKER.ContractError, "partially populated"):
            CHECKER._discover_dataset_clip_keys(self.data, ("train", "val"))

    def test_autosplit_fallback_moves_final_sorted_scene_to_val(self) -> None:
        fallback_root = self.root / "fallback-data"
        # Both buckets are >=10: scene_a=97, scene_b=38.  The dataset fallback
        # therefore pops lexicographically final scene_b into validation.
        self._write_complete_dataset_clip(
            fallback_root / "scene_a" / "clip_000001"
        )
        self._write_complete_dataset_clip(
            fallback_root / "scene_a" / "clip_000003"
        )
        self._write_complete_dataset_clip(
            fallback_root / "scene_b" / "clip_000002"
        )
        self.assertEqual(
            CHECKER._discover_dataset_clip_keys(
                fallback_root, ("train", "val")
            ),
            (
                "scene_a/clip_000001",
                "scene_a/clip_000003",
                "scene_b/clip_000002",
            ),
        )

    def test_explicit_split_layout_is_rejected(self) -> None:
        (self.data / "train" / "nested_scene" / "clip_000003").mkdir(
            parents=True
        )
        with self.assertRaisesRegex(CHECKER.ContractError, "explicit .* layout"):
            CHECKER._discover_dataset_clip_keys(self.data, ("train", "val"))

    def test_ready_splits_must_be_exact_train_then_val(self) -> None:
        self.ready["splits"] = ["val", "train"]
        self._write_ready()
        with self.assertRaisesRegex(CHECKER.ContractError, "cache.ready.splits"):
            CHECKER.validate_cache(
                str(self.cache), str(self.data), ("train", "val")
            )

    def test_ready_dataset_root_must_equal_scene_root(self) -> None:
        wrong_root = self.root / "wrong-scene-root"
        wrong_root.mkdir()
        self.ready["dataset_root"] = str(wrong_root.resolve())
        self._write_ready()
        with self.assertRaisesRegex(
            CHECKER.ContractError, "cache.ready.dataset_root"
        ):
            CHECKER.validate_cache(
                str(self.cache), str(self.data), ("train", "val")
            )

    def test_missing_sidecar_fails(self) -> None:
        (self.cache / self.keys[1] / CHECKER.CACHE_FILE).unlink()
        with self.assertRaisesRegex(CHECKER.ContractError, "missing/empty"):
            CHECKER.validate_cache(
                str(self.cache), str(self.data), ("train", "val")
            )

    def test_symlink_sidecar_fails(self) -> None:
        sidecar = self.cache / self.keys[1] / CHECKER.CACHE_FILE
        external = self.root / "external.npz"
        external.write_bytes(b"external")
        sidecar.unlink()
        sidecar.symlink_to(external)
        with self.assertRaisesRegex(CHECKER.ContractError, "missing/empty"):
            CHECKER.validate_cache(
                str(self.cache), str(self.data), ("train", "val")
            )

    def test_noncausal_marker_fails(self) -> None:
        self.ready["causal"] = False
        self._write_ready()
        with self.assertRaisesRegex(CHECKER.ContractError, "cache.ready.causal"):
            CHECKER.validate_cache(
                str(self.cache), str(self.data), ("train", "val")
            )

    def test_manifest_gt_scale_fails(self) -> None:
        manifest_path = self.cache / self.keys[0] / CHECKER.CACHE_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["per_episode_gt_scale_used"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(
            CHECKER.ContractError, "per_episode_gt_scale_used"
        ):
            CHECKER.validate_cache(
                str(self.cache), str(self.data), ("train", "val")
            )


if __name__ == "__main__":
    unittest.main()
