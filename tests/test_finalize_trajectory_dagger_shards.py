from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest
from scripts.tools import build_r2r_train_dagger_cohort as cohort
from scripts.tools import build_r2r_train_dagger_shards as shard_builder
from scripts.tools import finalize_trajectory_dagger_shards as finalizer


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], 8), (["--expected-num-shards", "4"], 4)],
)
def test_parse_args_supports_configurable_shard_count(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: int,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_trajectory_dagger_shards.py",
            "--plan",
            "/plan.json",
            "--collection-base",
            "/collections",
            "--control-base",
            "/controls",
            "--output",
            "/training_roots.json",
            *extra_args,
        ],
    )

    assert finalizer._parse_args().expected_num_shards == expected


def test_main_forwards_expected_shard_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    collection_base = tmp_path / "collections"
    control_base = tmp_path / "controls"
    collection_base.mkdir()
    control_base.mkdir()
    output = tmp_path / "training_roots.json"
    monkeypatch.setattr(finalizer, "FJL_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_trajectory_dagger_shards.py",
            "--plan",
            str(plan),
            "--collection-base",
            str(collection_base),
            "--control-base",
            str(control_base),
            "--output",
            str(output),
            "--expected-num-shards",
            "4",
        ],
    )
    captured: dict[str, int] = {}

    def stop_after_plan(
        _: Path,
        *,
        expected_num_shards: int,
    ) -> dict:
        captured["expected_num_shards"] = expected_num_shards
        raise finalizer.FinalizeError("stop after forwarding check")

    monkeypatch.setattr(finalizer, "_validate_plan", stop_after_plan)

    assert finalizer.main() == 2
    assert captured == {"expected_num_shards": 4}


def _raw_episodes() -> list[dict]:
    episodes: list[dict] = []
    episode_id = 0
    for scene_index in range(4):
        scene = f"scene_{scene_index}"
        for route_offset in range(4):
            trajectory_id = scene_index * 100 + route_offset
            for paraphrase in range(3):
                episodes.append(
                    {
                        "scene_id": f"mp3d/{scene}/{scene}.glb",
                        "episode_id": episode_id,
                        "trajectory_id": trajectory_id,
                        "instruction": {
                            "instruction_text": (
                                f"Route {trajectory_id}, wording {paraphrase}."
                            )
                        },
                    }
                )
                episode_id += 1
    return episodes


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": _raw_episodes()}, handle)
    return path


def _build_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    dataset_path = _write_dataset(tmp_path)
    output_dir = tmp_path / "cohorts"
    monkeypatch.setattr(shard_builder, "FJL_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_r2r_train_dagger_shards.py",
            "--dataset",
            str(dataset_path),
            "--count",
            str(len(_raw_episodes())),
            "--num-shards",
            "4",
            "--seed",
            "17",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert shard_builder.main() == 0
    return dataset_path, output_dir / "plan.json"


def test_plan_validator_proves_full_episode_and_route_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan_path = _build_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(finalizer, "FJL_ROOT", tmp_path)

    audit = finalizer._validate_plan(
        plan_path,
        expected_episode_count=len(_raw_episodes()),
        expected_num_shards=4,
    )

    assert audit["episode_count"] == len(_raw_episodes())
    assert audit["route_count"] == 16
    assert len(audit["shards"]) == 4
    assert sum(item["episode_count"] for item in audit["shards"]) == len(
        _raw_episodes()
    )


def test_plan_validator_rejects_a_canonical_route_crossing_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path, plan_path = _build_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(finalizer, "FJL_ROOT", tmp_path)
    canonical = cohort._load_episodes(dataset_path)
    canonical_by_key = {cohort._episode_key(item): item for item in canonical}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    payloads = [
        json.loads((plan_path.parent / item["file"]).read_text(encoding="utf-8"))
        for item in plan["shards"]
    ]

    route_rows: dict[tuple[str, int], list[tuple[int, dict]]] = {}
    for shard_index, payload in enumerate(payloads):
        for row in payload["episodes"]:
            key = (row["scene_id"], int(row["episode_id"]))
            route = cohort._route_key(canonical_by_key[key])
            route_rows.setdefault(route, []).append((shard_index, row))
    route, occurrences = next(
        (route, rows)
        for route, rows in route_rows.items()
        if len(rows) >= 2
    )
    source_index = occurrences[0][0]
    moved = occurrences[0][1]
    target_index = (source_index + 1) % len(payloads)
    payloads[source_index]["episodes"].remove(moved)
    payloads[target_index]["episodes"].append(moved)

    for index in {source_index, target_index}:
        payload = payloads[index]
        payload["count"] = len(payload["episodes"])
        path = plan_path.parent / plan["shards"][index]["file"]
        data = shard_builder._json_bytes(payload)
        path.write_bytes(data)
        keys = {
            (row["scene_id"], int(row["episode_id"]))
            for row in payload["episodes"]
        }
        routes = {
            cohort._route_key(canonical_by_key[key])
            for key in keys
        }
        entry = plan["shards"][index]
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        entry["episode_count"] = len(keys)
        entry["route_count"] = len(routes)
        entry["episode_key_sha256"] = shard_builder._key_digest(keys)
        entry["route_key_sha256"] = shard_builder._key_digest(routes)
    plan_path.write_bytes(shard_builder._json_bytes(plan))

    with pytest.raises(finalizer.FinalizeError, match="crosses shards"):
        finalizer._validate_plan(
            plan_path,
            expected_episode_count=len(_raw_episodes()),
            expected_num_shards=4,
        )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_collection_validator_accepts_sealed_native_no_sample_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "FJL_ROOT", tmp_path)
    dataset_path = _write_dataset(tmp_path)
    cohort_path = tmp_path / "shard_00.json"
    _write_json(cohort_path, {"split": "train", "count": 2, "episodes": []})
    collection_base = tmp_path / "collections"
    control_base = tmp_path / "controls"
    collection_root = collection_base / "shard_00"
    control_root = control_base / "shard_00"
    collection_root.mkdir(parents=True)
    control_root.mkdir(parents=True)

    fingerprint = "internnav-native-v1:" + "a" * 64
    progress_path = control_root / "progress.json"
    result_path = control_root / "result.json"
    progress_path.write_text("{}\n{}\n", encoding="utf-8")
    _write_json(
        result_path,
        {
            "total_episodes": 2,
            "rpc_policy_mode": finalizer.NATIVE_MODE,
            "rpc_policy_fingerprint": fingerprint,
            "native_protocol": finalizer.NATIVE_PROTOCOL,
        },
    )
    summary = {
        "expected_episodes": 2,
        "processed_episodes": 2,
        "committed_episodes": 1,
        "no_sample_episodes": 1,
        "samples": 1,
        "frames": 3,
        "jpegs": 13,
        "tar_payload_bytes": 1024,
        "control_progress_sha256": finalizer._sha256(progress_path),
        "control_result_sha256": finalizer._sha256(result_path),
    }
    _write_json(
        collection_root / "collection_manifest.json",
        {
            "schema": finalizer.COLLECTION_SCHEMA,
            "ready": True,
            "contract": {
                "rpc_policy_mode": finalizer.NATIVE_MODE,
                "rpc_policy_fingerprint": fingerprint,
                "policy_fingerprint": fingerprint,
                "native_protocol": finalizer.NATIVE_PROTOCOL,
                "data_path": str(dataset_path),
                "data_sha256": finalizer._sha256(dataset_path),
                "episode_cohort": {
                    "path": str(cohort_path),
                    "sha256": finalizer._sha256(cohort_path),
                    "max_episodes": None,
                },
            },
            "summary": summary,
        },
    )
    _write_json(
        control_root / "collection_wrapper_manifest.json",
        {
            "schema": finalizer.WRAPPER_SCHEMA,
            "verification_status": "sealed_native_verified",
            "verified_policy": {
                "rpc_policy_mode": finalizer.NATIVE_MODE,
                "native_protocol": finalizer.NATIVE_PROTOCOL,
                "policy_fingerprint": fingerprint,
            },
            "identity": {
                "requested_policy": {
                    "system2": "internnav_native_qwen",
                    "system1": "internnav_native_nextdit_async",
                    "external_checkpoint": False,
                    "lora": False,
                    "adapter": False,
                }
            },
        },
    )
    shard = {
        "index": 0,
        "cohort_path": cohort_path,
        "cohort_sha256": finalizer._sha256(cohort_path),
        "episode_count": 2,
        "route_count": 1,
    }
    plan_audit = {
        "dataset_path": dataset_path,
        "dataset_sha256": finalizer._sha256(dataset_path),
    }
    monkeypatch.setattr(
        finalizer,
        "_run_deep_validator",
        lambda collection, control, *, max_bytes: {
            "status": "ok",
            "collection_root": str(collection),
            "manifest_ready": True,
            "episodes": 1,
            "samples": 1,
            "frames": 3,
            "jpegs": 13,
            "tar_payload_bytes": 1024,
            "capacity_bytes": finalizer._tree_snapshot(
                collection,
                max_bytes=max_bytes,
            )[0],
            "sealed_now": False,
        },
    )

    audit = finalizer._validate_collection(
        shard,
        collection_base,
        control_base,
        plan_audit,
        deep_max_bytes=finalizer.ABSOLUTE_MAX_BYTES,
    )
    assert audit["fingerprint"] == fingerprint
    assert audit["samples"] == 1

    manifest_path = collection_root / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ready"] = False
    _write_json(manifest_path, manifest)
    with pytest.raises(finalizer.FinalizeError, match="not sealed"):
        finalizer._validate_collection(
            shard,
            collection_base,
            control_base,
            plan_audit,
            deep_max_bytes=finalizer.ABSOLUTE_MAX_BYTES,
        )


def test_plan_validator_reconstructs_seeded_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan_path = _build_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(finalizer, "FJL_ROOT", tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["seed"] = 18
    plan_path.write_bytes(shard_builder._json_bytes(plan))

    with pytest.raises(
        finalizer.FinalizeError,
        match="deterministic reconstruction",
    ):
        finalizer._validate_plan(
            plan_path,
            expected_episode_count=len(_raw_episodes()),
            expected_num_shards=4,
        )


def test_contract_invariant_allows_only_cohort_identity_to_differ() -> None:
    first = {
        "round_id": 0,
        "candidate_quotas": {"normal_per_episode": 1, "hard_per_episode": 2},
        "episode_cohort": {
            "path": "/allowed/shard_00.json",
            "sha256": "a" * 64,
            "max_episodes": None,
        },
    }
    second = json.loads(json.dumps(first))
    second["episode_cohort"]["path"] = "/allowed/shard_01.json"
    second["episode_cohort"]["sha256"] = "b" * 64
    first_invariant = finalizer._contract_invariant(first)
    second_invariant = finalizer._contract_invariant(second)
    assert first_invariant == second_invariant
    assert finalizer._require_single_contract_invariant(
        [
            {"contract_invariant": first_invariant},
            {"contract_invariant": second_invariant},
        ]
    ) == hashlib.sha256(first_invariant).hexdigest()

    second["candidate_quotas"]["hard_per_episode"] = 3
    with pytest.raises(finalizer.FinalizeError, match="invariants differ"):
        finalizer._require_single_contract_invariant(
            [
                {"contract_invariant": first_invariant},
                {"contract_invariant": finalizer._contract_invariant(second)},
            ]
        )


def test_deep_validator_runner_is_read_only_and_parses_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = tmp_path / "validator.py"
    validator.write_text("# test validator\n", encoding="utf-8")
    collection = tmp_path / "collection"
    control = tmp_path / "control"
    collection.mkdir()
    control.mkdir()
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return finalizer.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"status": "ok"}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(finalizer, "DEEP_VALIDATOR", validator)
    monkeypatch.setattr(finalizer.subprocess, "run", fake_run)
    result = finalizer._run_deep_validator(
        collection,
        control,
        max_bytes=123456,
    )

    assert result == {"status": "ok"}
    command = captured["command"]
    assert isinstance(command, list)
    assert "--collection-root" in command
    assert "--control-root" in command
    assert "--max-bytes" in command
    assert "--seal" not in command


def test_tree_snapshot_content_hash_detects_same_size_mutation(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    artifact = collection / "episode.tar"
    artifact.write_bytes(b"before")
    _, snapshot = finalizer._tree_snapshot(collection, max_bytes=1024)
    artifact.write_bytes(b"after!")
    _, mutated_snapshot = finalizer._tree_snapshot(
        collection,
        max_bytes=1024,
    )
    assert snapshot["episode.tar"][-1] != mutated_snapshot["episode.tar"][-1]

    with pytest.raises(finalizer.FinalizeError, match="changed"):
        finalizer._assert_tree_unchanged(
            collection,
            snapshot,
            max_bytes=1024,
            label="test collection",
        )
