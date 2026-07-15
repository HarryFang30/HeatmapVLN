from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.tools.summarize_task39_stage23_causal import (
    BOOTSTRAP_REPLICATES,
    EXPECTED_CODE_COMMIT,
    EXPECTED_EPISODE_COUNT,
    PROTOCOL_CONTRACT,
    ContractError,
    main,
    summarize,
)


def _cohort() -> list[dict[str, object]]:
    episodes = []
    per_scene = EXPECTED_EPISODE_COUNT // 3
    for index in range(EXPECTED_EPISODE_COUNT):
        scene_index = min(index // per_scene, 2)
        episodes.append(
            {
                "scene_id": f"scene_{scene_index}",
                "episode_id": index,
            }
        )
    return episodes


def _manifest(arm: str) -> dict[str, object]:
    return {
        "created_at": f"2026-07-15T00:00:0{int(arm == 'pano_control')}+00:00",
        "code_commit": EXPECTED_CODE_COMMIT,
        "config": "/fixed/stage3.yaml",
        "base_checkpoint": f"/artifacts/{arm}/task38_lora.pth",
        "stage3_checkpoint": f"/artifacts/{arm}/epoch_002.pth",
        "expected_epoch": 2,
        "scenes_dir": "/datasets/mp3d",
        "data_path": "/datasets/r2r/val_unseen.json.gz",
        "rpc_root": "/runtime/rpc",
        **PROTOCOL_CONTRACT,
        "auto_stop_distance": 0.0,
        "oracle_system2": False,
        "oracle_system2_strategy": "farthest_visible",
        "oracle_system2_lookahead_m": 2.0,
        "oracle_system2_min_ahead_m": 0.5,
        "oracle_system2_max_side_dist_m": 6.0,
        "trajectory_selection": "mean",
        "trajectory_x_sign": 1.0,
        "trajectory_heading_alignment": "none",
        "system1_coord_order": "generated",
    }


def _row(
    episode: dict[str, object],
    *,
    spl: float,
    pano: bool,
) -> dict[str, object]:
    return {
        **episode,
        "success": 1.0,
        "spl": spl,
        "os": 1.0,
        "ne": 3.0 if pano else 4.0,
        "steps": 90 if pano else 100,
        "vlm_calls": 4 if pano else 5,
        "trajectory_calls": 2 if pano else 3,
        **PROTOCOL_CONTRACT,
        "auto_stop_distance": 0.0,
        "oracle_system2": False,
        "trajectory_selection": "mean",
        "trajectory_x_sign": 1.0,
        "trajectory_heading_alignment": "none",
        "system1_coord_order": "generated",
    }


def _result(rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    return {
        "SPL": sum(float(row["spl"]) for row in rows) / count,
        "SR": sum(float(row["success"]) for row in rows) / count,
        "OS": sum(float(row["os"]) for row in rows) / count,
        "NE": sum(float(row["ne"]) for row in rows) / count,
        "total_episodes": count,
        **PROTOCOL_CONTRACT,
        "auto_stop_distance": 0.0,
        "oracle_system2": False,
        "trajectory_selection": "mean",
        "trajectory_x_sign": 1.0,
        "trajectory_heading_alignment": "none",
        "system1_coord_order": "generated",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _ordered_identity_hash(episodes: list[dict[str, object]]) -> str:
    keys = [[episode["scene_id"], episode["episode_id"]] for episode in episodes]
    payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_eval(
    root: Path,
    *,
    arm: str,
    episodes: list[dict[str, object]],
    spl: float,
) -> list[dict[str, object]]:
    root.mkdir()
    pano = arm == "pano_control"
    rows = [_row(episode, spl=spl, pano=pano) for episode in episodes]
    _write_json(root / "eval_manifest.json", _manifest(arm))
    (root / "progress.json").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(root / "result.json", _result(rows))
    return rows


def _comparison(
    tmp_path: Path,
    *,
    spl_delta: float,
) -> tuple[Path, Path, Path]:
    episodes = _cohort()
    cohort_path = tmp_path / "ordered_cohort.json"
    _write_json(
        cohort_path,
        {
            "split": "val_unseen",
            "count": EXPECTED_EPISODE_COUNT,
            "episode_count": EXPECTED_EPISODE_COUNT,
            "ordered_episode_identity_sha256": _ordered_identity_hash(episodes),
            "episodes": episodes,
        },
    )
    warmup = tmp_path / "warmup"
    pano = tmp_path / "pano"
    _write_eval(
        warmup,
        arm="warmup_original",
        episodes=episodes,
        spl=0.40,
    )
    _write_eval(
        pano,
        arm="pano_control",
        episodes=episodes,
        spl=0.40 + spl_delta,
    )
    return warmup, pano, cohort_path


def test_summary_passes_locked_gate_and_reports_all_point_effects(tmp_path):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    output = tmp_path / "summary.json"

    assert (
        main(
            [
                "--warmup-eval-dir",
                str(warmup),
                "--pano-control-eval-dir",
                str(pano),
                "--ordered-cohort",
                str(cohort),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["estimand"] == "pano_control_minus_warmup_original"
    assert report["decision"] == "confirmatory_pass"
    assert report["screening_gate"]["passed"] is True
    assert report["confirmatory_gate"]["passed"] is True
    assert report["contract"]["passed"] is True
    assert report["contract"]["ordered_cohort"]["episodes"] == 200
    assert report["point_effects"] == pytest.approx(
        {
            "SPL": 0.03,
            "SR": 0.0,
            "OS": 0.0,
            "NE": -1.0,
            "steps": -10.0,
            "vlm_calls": -1.0,
            "trajectory_calls": -1.0,
        }
    )
    bootstrap = report["paired_scene_cluster_bootstrap"]
    assert bootstrap["replicates"] == BOOTSTRAP_REPLICATES == 50_000
    assert bootstrap["SPL"]["ci95"] == pytest.approx([0.03, 0.03])
    assert bootstrap["SR"]["ci95_lower"] == pytest.approx(0.0)
    assert report["gate"]["passed"] is True
    assert report["gate"]["locked_thresholds"] == {
        "delta_SPL_minimum": 0.02,
        "SPL_two_sided_95_CI_lower_strictly_above": 0.0,
        "SR_one_sided_95_lower_strictly_above": -0.02,
    }
    original = output.read_bytes()
    assert (
        main(
            [
                "--warmup-eval-dir",
                str(warmup),
                "--pano-control-eval-dir",
                str(pano),
                "--ordered-cohort",
                str(cohort),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == original


def test_summary_emits_a_valid_failed_gate_without_changing_thresholds(tmp_path):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.01)

    report = summarize(warmup, pano, cohort)

    assert report["point_effects"]["SPL"] == pytest.approx(0.01)
    assert report["decision"] == "screening_fail"
    assert report["screening_gate"]["passed"] is False
    assert report["confirmatory_gate"]["passed"] is False
    assert report["gate"]["passed"] is False
    assert report["gate"]["checks"]["delta_SPL_at_least_0.02"] is False
    assert report["gate"]["locked_thresholds"]["delta_SPL_minimum"] == 0.02


def test_summary_rejects_progress_that_is_not_in_exact_cohort_order(tmp_path):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    progress_path = pano / "progress.json"
    lines = progress_path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="does not exactly match ordered cohort"):
        summarize(warmup, pano, cohort)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("rpc_protocol", "heatmapvln-r2r-json-v1"),
        ("rpc_sampling_protocol", "legacy-global-rng"),
        ("rpc_deterministic_sampling_enabled", False),
        ("rpc_protocol_seed", 43),
        ("rpc_require_deterministic_sampling", False),
    ),
)
def test_summary_rejects_any_non_preregistered_protocol_row(
    tmp_path,
    field,
    replacement,
):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    progress_path = pano / "progress.json"
    lines = progress_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[37])
    row[field] = replacement
    lines[37] = json.dumps(row, sort_keys=True)
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match=rf"protocol mismatch.*{field}"):
        summarize(warmup, pano, cohort)


def test_summary_rejects_duplicate_episode_identities(tmp_path):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    cohort_data = json.loads(cohort.read_text(encoding="utf-8"))
    cohort_data["episodes"][-1] = dict(cohort_data["episodes"][0])
    _write_json(cohort, cohort_data)

    with pytest.raises(ContractError, match="ordered cohort contains duplicate episodes"):
        summarize(warmup, pano, cohort)


def test_summary_rejects_a_common_runtime_field_difference(tmp_path):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    manifest_path = pano / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["oracle_system2_strategy"] = "lookahead"
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="common run fields differ"):
        summarize(warmup, pano, cohort)


def test_summary_rejects_same_wrong_code_commit_in_both_arms(tmp_path):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    for root in (warmup, pano):
        manifest_path = root / "eval_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["code_commit"] = "deadbeef"
        _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match="code_commit must be"):
        summarize(warmup, pano, cohort)


def test_summary_checks_declared_ordered_cohort_identity_hash(tmp_path):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    cohort_data = json.loads(cohort.read_text(encoding="utf-8"))
    cohort_data["ordered_episode_identity_sha256"] = "0" * 64
    _write_json(cohort, cohort_data)

    with pytest.raises(ContractError, match="ordered_episode_identity_sha256"):
        summarize(warmup, pano, cohort)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("auto_stop_distance", 0.25, "privileged auto-stop"),
        ("oracle_system2", True, "oracle_system2=false"),
    ),
)
def test_summary_rejects_privileged_evaluation_settings(
    tmp_path,
    field,
    replacement,
    message,
):
    warmup, pano, cohort = _comparison(tmp_path, spl_delta=0.03)
    manifest_path = pano / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    _write_json(manifest_path, manifest)

    with pytest.raises(ContractError, match=message):
        summarize(warmup, pano, cohort)
