#!/usr/bin/env python3
"""Collect one-deviation closed-loop labels from saved candidate-audit states.

Each branch is reconstructed by resetting the exact R2R episode and replaying
the saved navigation prefix, including every historical look-down cycle.  The
selected treatment is then executed once; all later decisions come from the
frozen native InternNav deployment arm.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import scripts.evaluation.r2r_candidate_support_audit_client as base
from scripts.evaluation.candidate_support_audit import AuditShardWriter


SCHEMA = "candidate-continuation-rollout-v1"
TARGET_SCHEMA = "candidate-continuation-targets-v1"
HORIZONS = (1, 3, 5)
ROLE_ORDER = (
    "native_mean",
    "system2_selector",
    "heatmap_token_selector",
    "native_local_oracle",
    "union_local_oracle",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under_fjl(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = base.LOCAL_FJL_ROOT.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay below {root}: {resolved}") from exc
    return resolved


def _load_target_file(shard_id: int) -> tuple[Path, dict[str, Any]]:
    directory_value = os.environ.get("CONTINUATION_TARGETS_DIR", "").strip()
    if not directory_value:
        raise RuntimeError("CONTINUATION_TARGETS_DIR is required")
    directory = _under_fjl(Path(directory_value))
    path = directory / f"targets_shard_{int(shard_id):02d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != TARGET_SCHEMA or int(payload.get("shard_id", -1)) != int(shard_id):
        raise RuntimeError(f"invalid continuation target payload: {path}")
    expected_hash = str(payload.get("payload_sha256") or "")
    unhashed = dict(payload)
    unhashed.pop("payload_sha256", None)
    if _canonical_sha256(unhashed) != expected_hash:
        raise RuntimeError(f"target payload SHA256 mismatch: {path}")
    return path, payload


def _read_source_records(
    source_root: Path, shard_id: int, state_keys: set[str]
) -> dict[str, dict[str, Any]]:
    shard_dir = source_root / f"shard_{int(shard_id):02d}"
    index_path = shard_dir / "records.jsonl"
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("shard_id", -1)) != int(shard_id):
        raise RuntimeError(f"source shard id mismatch: {manifest_path}")
    if _file_sha256(index_path) != manifest.get("records_jsonl_sha256"):
        raise RuntimeError(f"source records hash mismatch: {index_path}")
    selected: dict[str, dict[str, Any]] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("state_key") or "")
        if key in state_keys:
            row["__source_shard_dir"] = str(shard_dir)
            selected[key] = row
    missing = sorted(state_keys - set(selected))
    if missing:
        raise RuntimeError(f"target states absent from source shard: {missing[:5]}")
    return selected


def _load_source_arrays(record: dict[str, Any]) -> dict[str, np.ndarray]:
    shard_dir = Path(record["__source_shard_dir"]).resolve()
    path = (shard_dir / str(record["array_file"])).resolve()
    try:
        path.relative_to(shard_dir)
    except ValueError as exc:
        raise RuntimeError(f"source array path escapes shard: {path}") from exc
    if path.stat().st_size != int(record["array_file_bytes"]):
        raise RuntimeError(f"source array byte count mismatch: {path}")
    if _file_sha256(path) != record["array_file_sha256"]:
        raise RuntimeError(f"source array SHA256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _episode_key(episode: Any) -> tuple[str, int]:
    return episode.scene_id.split("/")[-2], int(episode.episode_id)


def _reset_to_episode(env: Any, episode: Any) -> Any:
    # habitat-lab 0.1.7 chooses the next episode from this private iterator in
    # Env.reset().  Replacing it with a one-item iterator gives every branch a
    # fully clean task/simulator state for the same episode.
    if not hasattr(env, "_episode_iterator"):
        raise RuntimeError("Habitat Env has no _episode_iterator; cannot replay safely")
    env._episode_iterator = iter((episode,))
    observations = env.reset()
    if _episode_key(env.current_episode) != _episode_key(episode):
        raise RuntimeError("Habitat reset did not select the requested episode")
    return observations


def _pose_error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return {
        "translation_m": float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])),
        "rotation_max_abs": float(np.max(np.abs(actual[:3, :3] - expected[:3, :3]))),
        "matrix_max_abs": float(np.max(np.abs(actual - expected))),
    }


def _validate_pose(actual: np.ndarray, expected: np.ndarray, *, label: str) -> dict[str, float]:
    error = _pose_error(actual, expected)
    if error["translation_m"] > 5e-4 or error["rotation_max_abs"] > 5e-4:
        raise RuntimeError(f"deterministic replay diverged at {label}: {error}")
    return error


def _replay_prefix(
    env: Any,
    *,
    source_record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    image_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = np.asarray(arrays["client_executed_action_prefix"], dtype=np.int64).tolist()
    c2w = np.asarray(arrays["client_full_history_c2w"], dtype=np.float32)
    capture_steps = np.asarray(
        arrays["client_full_history_capture_steps"], dtype=np.int64
    ).tolist()
    call_indices = np.asarray(
        arrays["client_full_history_call_indices"], dtype=np.int64
    ).tolist()
    if not (len(c2w) == len(capture_steps) == len(call_indices)):
        raise RuntimeError("saved history arrays have inconsistent lengths")
    if capture_steps != sorted(capture_steps) or any(
        step < 0 or step > len(actions) for step in capture_steps
    ):
        raise RuntimeError("saved history capture steps are non-causal")
    snapshot = source_record["runtime_snapshot"]
    if int(snapshot["executed_action_prefix_length"]) != len(actions):
        raise RuntimeError("runtime snapshot/action prefix length mismatch")
    if capture_steps != [int(value) for value in snapshot["full_history_capture_steps"]]:
        raise RuntimeError("runtime snapshot/history capture-step mismatch")
    if call_indices != [int(value) for value in snapshot["full_history_call_indices"]]:
        raise RuntimeError("runtime snapshot/history call-index mismatch")

    history: list[dict[str, Any]] = []
    c2w_errors: list[dict[str, float]] = []
    cursor = 0
    last_new_call = -1
    lookdown_cycles = 0
    for step in range(len(actions) + 1):
        while cursor < len(capture_steps) and int(capture_steps[cursor]) == step:
            record = base._capture_history_record(env, image_size, step)
            call_index = int(call_indices[cursor])
            record["system2_call_index"] = call_index
            c2w_errors.append(
                _validate_pose(record["c2w"], c2w[cursor], label=f"history[{cursor}]")
            )
            history.append(record)
            if call_index > last_new_call:
                if call_index != last_new_call + 1:
                    raise RuntimeError(
                        f"non-contiguous System2 call replay: {last_new_call} -> {call_index}"
                    )
                base.capture_lookdown_view(
                    env, image_size=base.NATIVE_LOOKDOWN_SENSOR_SIZE
                )
                if env.episode_over:
                    raise RuntimeError("episode ended during replayed look-down cycle")
                last_new_call = call_index
                lookdown_cycles += 1
            cursor += 1
        if step == len(actions):
            break
        action = int(actions[step])
        if action not in (
            int(base.ActionCode.FORWARD),
            int(base.ActionCode.LEFT),
            int(base.ActionCode.RIGHT),
        ):
            raise RuntimeError(f"invalid saved navigation action {action} at step {step}")
        _, done = base._apply_habitat_action(env, action)
        if done and step + 1 < len(actions):
            raise RuntimeError("episode terminated before saved prefix completed")
    if cursor != len(capture_steps):
        raise RuntimeError("not all saved history records were replayed")
    expected_call = int(source_record["system2_call_index"])
    if last_new_call != expected_call:
        raise RuntimeError(
            f"replayed System2 call boundary {last_new_call} != target {expected_call}"
        )
    body_error = _validate_pose(
        base._agent_body_pose(env),
        arrays["client_current_body_pose"],
        label="target_body_pose",
    )
    max_translation = max(
        [body_error["translation_m"]]
        + [error["translation_m"] for error in c2w_errors]
    )
    max_rotation = max(
        [body_error["rotation_max_abs"]]
        + [error["rotation_max_abs"] for error in c2w_errors]
    )
    return history, {
        "status": "exact_prefix_replay_verified",
        "actions": len(actions),
        "history_records": len(history),
        "lookdown_cycles": lookdown_cycles,
        "max_translation_error_m": max_translation,
        "max_rotation_max_abs": max_rotation,
        "target_body_pose_error": body_error,
    }


class BranchDiagnostics:
    def __init__(
        self,
        *,
        env: Any,
        route_tracker: Any,
        goal_position: np.ndarray,
        source_visited_poses: np.ndarray,
        success_radius_m: float,
        start_step_id: int,
    ) -> None:
        self.env = env
        self.route_tracker = route_tracker
        self.goal = np.asarray(goal_position, dtype=np.float64)
        self.success_radius_m = float(success_radius_m)
        self.start_progress_m = float(route_tracker.progress_m)
        self.start_step_id = int(start_step_id)
        initial_pose = base._agent_body_pose(env).astype(np.float32)
        self.poses = [initial_pose]
        self.actions: list[int] = []
        self.phases: list[int] = [0]
        self.travelled_m = 0.0
        self.collision_or_stuck_count = 0
        self.revisit_count = 0
        source = np.asarray(source_visited_poses, dtype=np.float32)
        self.older_positions = (
            source[:, :3, 3].astype(np.float64)
            if source.size
            else np.empty((0, 3), dtype=np.float64)
        )
        initial_distance = float(np.linalg.norm(initial_pose[:3, 3] - self.goal))
        self.min_goal_distance_m = initial_distance
        self.entered_radius = initial_distance <= self.success_radius_m
        self.left_after_enter = False

    def apply(self, action: int, *, phase: int) -> bool:
        before = self.poses[-1]
        _, done = base._apply_habitat_action(self.env, int(action))
        after = base._agent_body_pose(self.env).astype(np.float32)
        distance = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        self.travelled_m += distance
        if int(action) == int(base.ActionCode.FORWARD) and distance < 0.02:
            self.collision_or_stuck_count += 1
        if len(self.poses) >= 4:
            old = np.asarray([pose[:3, 3] for pose in self.poses[:-3]], dtype=np.float64)
            if len(old) and float(np.linalg.norm(old - after[:3, 3], axis=1).min()) <= 0.25:
                self.revisit_count += 1
        if len(self.older_positions):
            if float(
                np.linalg.norm(self.older_positions - after[:3, 3], axis=1).min()
            ) <= 0.25:
                self.revisit_count += 1
        self.route_tracker.observe(after[:3, 3])
        goal_distance = float(np.linalg.norm(after[:3, 3] - self.goal))
        self.min_goal_distance_m = min(self.min_goal_distance_m, goal_distance)
        inside = goal_distance <= self.success_radius_m
        if inside:
            self.entered_radius = True
        elif self.entered_radius:
            self.left_after_enter = True
        self.actions.append(int(action))
        self.poses.append(after)
        self.phases.append(int(phase))
        return bool(done)

    def outcome(
        self,
        *,
        step_id: int,
        future_cycles: int,
        carried_forward: bool = False,
        end_reason: str | None = None,
    ) -> dict[str, Any]:
        pose = self.poses[-1]
        offpath, raw_progress = self.route_tracker.project(pose[:3, 3])
        metrics = self.env.get_metrics()

        def metric(name: str, fallback: float = 0.0) -> float:
            value = metrics.get(name, fallback)
            try:
                result = float(value)
            except (TypeError, ValueError):
                result = fallback
            return result if math.isfinite(result) else fallback

        return {
            "future_system2_cycles": int(future_cycles),
            "navigation_actions_after_branch": len(self.actions),
            "absolute_navigation_step_id": int(step_id),
            "episode_over": bool(self.env.episode_over),
            "carried_forward": bool(carried_forward),
            "end_reason": end_reason,
            "habitat_success": metric("success"),
            "habitat_spl": metric("spl"),
            "habitat_oracle_success": metric("oracle_success"),
            "habitat_distance_to_goal_m": metric(
                "distance_to_goal", float(np.linalg.norm(pose[:3, 3] - self.goal))
            ),
            "euclidean_goal_distance_m": float(
                np.linalg.norm(pose[:3, 3] - self.goal)
            ),
            "min_euclidean_goal_distance_m": self.min_goal_distance_m,
            "entered_euclidean_success_radius": bool(self.entered_radius),
            "left_euclidean_success_radius": bool(self.left_after_enter),
            "route_progress_m": float(self.route_tracker.progress_m),
            "route_progress_delta_m": float(
                self.route_tracker.progress_m - self.start_progress_m
            ),
            "raw_route_progress_m": float(raw_progress),
            "endpoint_offpath_m": float(offpath),
            "travelled_m": float(self.travelled_m),
            "collision_or_stuck_count": int(self.collision_or_stuck_count),
            "revisit_count": int(self.revisit_count),
            "habitat_elapsed_steps": int(getattr(self.env, "_elapsed_steps", -1)),
        }


def _capture_action_history(
    env: Any,
    history: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    step_id: int,
    system2_call_index: int,
) -> None:
    record = base._capture_history_record(env, image_size, step_id)
    record["system2_call_index"] = int(system2_call_index)
    history.append(record)


def _execute_treatment(
    env: Any,
    diagnostics: BranchDiagnostics,
    history: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    image_size: tuple[int, int],
    step_id: int,
    target_call_index: int,
) -> tuple[int, bool]:
    actions = [int(action) for action in spec["actions"]]
    if int(spec["execute_len"]) != len(actions) or not bool(spec["replan_after"]):
        raise RuntimeError(f"unsupported/non-canonical treatment spec: {spec}")
    done = bool(env.episode_over)
    for index, action in enumerate(actions):
        if action == int(base.ActionCode.STOP):
            break
        if index > 0:
            _capture_action_history(
                env,
                history,
                image_size=image_size,
                step_id=step_id,
                system2_call_index=target_call_index,
            )
        done = diagnostics.apply(action, phase=1)
        step_id += 1
        if done:
            break
    return step_id, done


def _native_cycle(
    client: Any,
    env: Any,
    episode: Any,
    diagnostics: BranchDiagnostics,
    history: list[dict[str, Any]],
    *,
    instruction: str,
    image_size: tuple[int, int],
    vlm_image_size: tuple[int, int],
    traj_image_size: tuple[int, int],
    num_history: int,
    step_id: int,
    system2_call_index: int,
    args: Any,
    control_artifacts: dict[str, str],
) -> tuple[int, bool, dict[str, int], dict[str, Any]]:
    current = base._capture_history_record(env, image_size, step_id)
    current["system2_call_index"] = int(system2_call_index)
    prompt = base._sample_history_records(history, num_history)
    lookdown = base.capture_lookdown_view(
        env, image_size=base.NATIVE_LOOKDOWN_SENSOR_SIZE
    )
    history.append(current)
    response, accounting, _response_blobs = base._rpc_plan_panoramic(
        client,
        instruction=instruction,
        current_views=current["views"],
        history_panoramas=[record["views"] for record in prompt],
        current_c2w=current["c2w"],
        history_c2w=[record["c2w"] for record in prompt],
        current_capture_step=current["capture_step"],
        history_capture_steps=[record["capture_step"] for record in prompt],
        lookdown_img=lookdown,
        vlm_image_size=vlm_image_size,
        traj_image_size=traj_image_size,
        system1_coord_order=base._system1_coord_order(
            args, panoramic_internnav_protocol=False
        ),
        trajectory_selection=args.trajectory_selection,
        trajectory_x_sign=args.trajectory_x_sign,
        trajectory_heading_alignment=args.trajectory_heading_alignment,
        jpeg_quality=args.rpc_jpeg_quality,
        scene_id=_episode_key(episode)[0],
        episode_id=_episode_key(episode)[1],
        system2_call_index=int(system2_call_index),
        protocol_seed=args.rpc_protocol_seed,
        require_deterministic_sampling=args.rpc_require_deterministic_sampling,
        expected_control_mode=args.expected_control_mode,
        expected_control_artifacts=control_artifacts,
        oracle_system2=None,
    )
    actions = [int(action) for action in response.get("actions", [])]
    done = bool(env.episode_over)
    if response.get("terminal", False):
        action = actions[0] if actions else int(base.ActionCode.STOP)
        done = diagnostics.apply(action, phase=2)
        step_id += 1
    elif not actions:
        done = diagnostics.apply(int(base.ActionCode.STOP), phase=2)
        step_id += 1
    else:
        first = actions.pop(0)
        if first != int(base.ActionCode.STOP):
            done = diagnostics.apply(first, phase=2)
            step_id += 1
            for action in actions:
                if done:
                    break
                _capture_action_history(
                    env,
                    history,
                    image_size=image_size,
                    step_id=step_id,
                    system2_call_index=system2_call_index,
                )
                if action == int(base.ActionCode.STOP):
                    break
                done = diagnostics.apply(action, phase=2)
                step_id += 1
    response_summary = {
        "kind": response.get("kind"),
        "terminal": bool(response.get("terminal", False)),
        "llm_output": str(response.get("llm_output", "")),
        "actions": [int(value) for value in response.get("actions", [])],
        "sampling": response.get(base.HEATMAPVLN_RPC_SAMPLING_FIELD),
    }
    return step_id, bool(done), accounting, response_summary


def _merge_counter_totals(target: dict[str, int], source: dict[str, Any]) -> None:
    base.add_control_counters(target, source)


def _continuation_rpc_counts(row: dict[str, Any]) -> tuple[int, int]:
    counters = dict(row.get("control_counters") or {})
    rpc_calls = int(counters.get("control_rpc_calls", -1))
    trace = list(row.get("future_response_trace") or [])
    if rpc_calls < 0 or len(trace) != rpc_calls:
        raise RuntimeError(
            "continuation response trace/control RPC mismatch at "
            f"{row.get('state_key')}: {len(trace)} != {rpc_calls}"
        )
    trajectory_calls = sum(
        1 for response in trace if response.get("kind") == "trajectory"
    )
    return rpc_calls, trajectory_calls


def _migrate_progress_rpc_accounting(
    progress_file: Path, writer: AuditShardWriter
) -> None:
    """Upgrade pre-contract continuation progress rows from sealed branches."""
    progress_file = Path(progress_file)
    if not progress_file.exists() or progress_file.stat().st_size == 0:
        return
    branch_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in writer._records.values():
        branch_rows[(str(row["scene_id"]), int(row["episode_id"]))].append(row)

    rows: list[dict[str, Any]] = []
    changed = 0
    with progress_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["scene_id"]), int(row["episode_id"]))
            records = branch_rows.get(key) or []
            if not records:
                raise RuntimeError(
                    f"progress row lacks sealed continuation branches at line {line_number}: {key}"
                )
            expected_counters = base.empty_control_counters()
            expected_rpc_calls = 0
            expected_trajectory_calls = 0
            for record in records:
                base.add_control_counters(
                    expected_counters, dict(record.get("control_counters") or {})
                )
                rpc_calls, trajectory_calls = _continuation_rpc_counts(record)
                expected_rpc_calls += rpc_calls
                expected_trajectory_calls += trajectory_calls
            if expected_rpc_calls != expected_counters["control_rpc_calls"]:
                raise RuntimeError(f"sealed branch RPC accounting mismatch for {key}")
            for field, expected in expected_counters.items():
                actual = row.get(field)
                if field == "control_token_count_max":
                    if actual != expected:
                        row[field] = int(expected)
                        changed += 1
                elif actual != expected:
                    raise RuntimeError(
                        f"progress/sealed counter mismatch for {key} {field}: "
                        f"{actual!r} != {expected}"
                    )
            for field, expected in (
                ("vlm_calls", expected_rpc_calls),
                ("trajectory_calls", expected_trajectory_calls),
            ):
                actual = row.get(field)
                if actual is None:
                    row[field] = int(expected)
                    changed += 1
                elif actual != expected:
                    raise RuntimeError(
                        f"progress/sealed call mismatch for {key} {field}: "
                        f"{actual!r} != {expected}"
                    )
            rows.append(row)
    if changed == 0:
        return
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=progress_file.parent,
            prefix=f".{progress_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, progress_file)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(
        f"[continuation] atomically migrated {len(rows)} progress rows "
        f"({changed} corrected accounting fields)",
        flush=True,
    )


def _find_treatment(record: dict[str, Any], treatment_id: str) -> dict[str, Any]:
    matches = [
        treatment
        for treatment in record["candidate_set"]["treatments"]
        if str(treatment["treatment_id"]) == str(treatment_id)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"treatment {treatment_id} is not unique at {record['state_key']}"
        )
    return matches[0]


def _branch_key(source_state_key: str, treatment_id: str) -> str:
    return f"{source_state_key}:continuation:{treatment_id}"


def _collect_branch(
    *,
    client: Any,
    env: Any,
    episode: Any,
    target: dict[str, Any],
    source_record: dict[str, Any],
    source_arrays: dict[str, np.ndarray],
    treatment_id: str,
    roles: list[str],
    run_to_end: bool,
    args: Any,
    control_artifacts: dict[str, str],
    image_size: tuple[int, int],
    vlm_image_size: tuple[int, int],
    traj_image_size: tuple[int, int],
    smoke_mode: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _reset_to_episode(env, episode)
    instruction = base._normalize_instruction(episode.instruction.instruction_text)
    if instruction != str(source_record["instruction"]):
        raise RuntimeError(f"instruction mismatch at {source_record['state_key']}")
    history, replay = _replay_prefix(
        env,
        source_record=source_record,
        arrays=source_arrays,
        image_size=image_size,
    )
    import scripts.evaluation.trajectory_dagger as dagger_api

    route_tracker = dagger_api.MonotonicRouteTracker(
        np.asarray(source_arrays["client_reference_path"], dtype=np.float32)
    )
    route_tracker.load_state_dict(source_record["route_state"])
    if abs(
        float(route_tracker.progress_m)
        - float(source_record["current_route_progress_m"])
    ) > 1e-5:
        raise RuntimeError("saved route tracker progress mismatch")
    diagnostics = BranchDiagnostics(
        env=env,
        route_tracker=route_tracker,
        goal_position=source_arrays["client_goal_position"],
        source_visited_poses=source_arrays["client_visited_body_poses"],
        success_radius_m=float(args.candidate_audit_success_radius_m),
        start_step_id=int(source_record["step_id"]),
    )
    treatment = _find_treatment(source_record, treatment_id)
    step_id, done = _execute_treatment(
        env,
        diagnostics,
        history,
        spec=treatment["spec"],
        image_size=image_size,
        step_id=int(source_record["step_id"]),
        target_call_index=int(source_record["system2_call_index"]),
    )
    future_cycles = 0
    requested_horizons = (1,) if smoke_mode else HORIZONS
    horizon_outcomes: dict[str, dict[str, Any]] = {}
    response_trace: list[dict[str, Any]] = []
    control_counters = base.empty_control_counters()
    end_reason: str | None = "habitat_episode_over" if done else None
    max_future_cycles = int(os.environ.get("CONTINUATION_MAX_FUTURE_CYCLES", "80"))
    max_horizon = max(requested_horizons)

    while not done:
        if step_id >= int(args.max_steps_per_episode):
            end_reason = "max_navigation_steps"
            break
        if future_cycles >= max_future_cycles:
            end_reason = "max_future_system2_cycles"
            break
        if not run_to_end and future_cycles >= max_horizon:
            end_reason = "requested_horizon_complete"
            break
        call_index = int(source_record["system2_call_index"]) + 1 + future_cycles
        step_id, done, accounting, response_summary = _native_cycle(
            client,
            env,
            episode,
            diagnostics,
            history,
            instruction=instruction,
            image_size=image_size,
            vlm_image_size=vlm_image_size,
            traj_image_size=traj_image_size,
            num_history=int(args.num_history),
            step_id=step_id,
            system2_call_index=call_index,
            args=args,
            control_artifacts=control_artifacts,
        )
        base.add_control_counters(control_counters, accounting)
        response_trace.append(response_summary)
        future_cycles += 1
        if future_cycles in requested_horizons:
            horizon_outcomes[str(future_cycles)] = diagnostics.outcome(
                step_id=step_id,
                future_cycles=future_cycles,
                end_reason="habitat_episode_over" if done else None,
            )
        if done:
            end_reason = "habitat_episode_over"

    terminal_or_truncated = done or end_reason in {
        "max_navigation_steps",
        "max_future_system2_cycles",
    }
    carry = diagnostics.outcome(
        step_id=step_id,
        future_cycles=future_cycles,
        carried_forward=True,
        end_reason=end_reason,
    )
    for horizon in requested_horizons:
        horizon_outcomes.setdefault(str(horizon), dict(carry))
    end_outcome = (
        diagnostics.outcome(
            step_id=step_id,
            future_cycles=future_cycles,
            end_reason=end_reason,
        )
        if (run_to_end or done)
        else None
    )
    record = {
        "continuation_schema": SCHEMA,
        "source_state_key": str(source_record["state_key"]),
        "source_shard_id": int(target["source_shard_id"]),
        "dataset_split": str(source_record["dataset_split"]),
        "scene_split": str(target["scene_split"]),
        "scene_id": str(source_record["scene_id"]),
        "episode_id": int(source_record["episode_id"]),
        "instruction": instruction,
        "branch_system2_call_index": int(source_record["system2_call_index"]),
        "branch_step_id": int(source_record["step_id"]),
        "treatment_id": str(treatment_id),
        "treatment_spec": dict(treatment["spec"]),
        "selector_roles": sorted(roles, key=ROLE_ORDER.index),
        "run_to_episode_end": bool(run_to_end),
        "smoke_mode": bool(smoke_mode),
        "continuation_policy": {
            "first_chunk": "fixed_saved_treatment",
            "after_first_chunk": "frozen_native_internnav_pi0",
            "future_system2_rerun_per_branch": True,
            "paired_deterministic_noise_by_scene_episode_call": True,
            "history_reconstruction": "episode_reset_action_prefix_and_lookdown_replay",
        },
        "replay_verification": replay,
        "horizon_outcomes": horizon_outcomes,
        "episode_end_outcome": end_outcome,
        "episode_end_authoritative": bool(done),
        "termination": {
            "reason": end_reason,
            "future_system2_cycles": int(future_cycles),
            "terminal_or_truncated": bool(terminal_or_truncated),
        },
        "future_response_trace": response_trace,
        "control_counters": {key: int(value) for key, value in control_counters.items()},
        "source_state_strata": dict(source_record.get("state_strata") or {}),
        "diagnostic_selection": dict(target["diagnostic_selection"]),
    }
    arrays = {
        "continuation_pose_trace": np.stack(diagnostics.poses).astype(np.float32),
        "continuation_action_trace": np.asarray(diagnostics.actions, dtype=np.int8),
        "continuation_pose_phase": np.asarray(diagnostics.phases, dtype=np.int8),
        "source_current_body_pose": np.asarray(
            source_arrays["client_current_body_pose"], dtype=np.float32
        ),
    }
    return record, arrays


def _representative_outcome(row: dict[str, Any]) -> dict[str, Any]:
    end = row.get("episode_end_outcome")
    if isinstance(end, dict):
        return end
    horizons = row.get("horizon_outcomes") or {}
    for key in ("5", "3", "1"):
        if isinstance(horizons.get(key), dict):
            return horizons[key]
    raise RuntimeError("continuation row has no outcome")


def run_eval_rpc_continuation(args: Any) -> None:
    import yaml
    from vla_rpc.client import VLAClient

    base.ensure_vln_measures_registered()
    with open(args.config) as handle:
        train_cfg = yaml.safe_load(handle)
    if not bool(
        train_cfg.get("data", {}).get("trajectory", {}).get("panoramic_vlm_input", False)
    ):
        raise RuntimeError("continuation collection requires panoramic_vlm_input")
    vlm_image_size, traj_image_size = base._eval_image_sizes(train_cfg)
    image_size = vlm_image_size
    target_path, target_payload = _load_target_file(args.candidate_audit_shard_id)
    source_root = _under_fjl(Path(target_payload["source_audit_root"]))
    explicit_source = os.environ.get("CONTINUATION_SOURCE_AUDIT_ROOT", "").strip()
    if explicit_source and _under_fjl(Path(explicit_source)) != source_root:
        raise RuntimeError("target payload/source audit root mismatch")
    targets = list(target_payload["targets"])
    source_records = _read_source_records(
        source_root,
        int(args.candidate_audit_shard_id),
        {str(target["state_key"]) for target in targets},
    )
    smoke_mode = os.environ.get("CANDIDATE_COLLECTION_MODE", "full") == "smoke"

    control_artifacts = {
        "frozen_heatmap_checkpoint_sha256": args.expected_heatmap_checkpoint_sha256,
        "control_ema_checkpoint_sha256": args.expected_control_checkpoint_sha256,
        "native_model_manifest_sha256": args.expected_native_model_manifest_sha256,
    }
    client = VLAClient(
        server_addr=args.rpc_server,
        timeout_ms=args.rpc_timeout_ms,
        jpeg_quality=args.rpc_jpeg_quality,
    )
    client.connect()
    if not client.health_check():
        raise RuntimeError(f"RPC model server is not healthy: {args.rpc_server}")
    info = client.get_server_info()
    expected_model = "candidate-support-r2r-paired:deployment-native"
    if info is None or info.version != base.HEATMAPVLN_RPC_PROTOCOL_VERSION:
        raise RuntimeError(f"RPC protocol/server info mismatch: {info}")
    if info.model_version != expected_model:
        raise RuntimeError(
            f"continuation requires frozen native deployment arm: {info.model_version!r}"
        )
    if base.CANDIDATE_EXPORT_PROTO_VERSION not in set(info.supported_formats):
        raise RuntimeError("RPC server lacks paired candidate export capability")

    hab_cfg = base.build_habitat_config(args)
    env = base.habitat.Env(config=hab_cfg)
    episode_map = {_episode_key(episode): episode for episode in list(env.episodes)}
    target_list, target_set = base._episode_list_from_args(args)
    if target_set is None:
        target_set = set(episode_map)
        target_list = sorted(target_set)
    by_episode: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        key = (str(target["scene_id"]), int(target["episode_id"]))
        if key in target_set:
            by_episode[key].append(target)
    missing_target_episodes = sorted(set(by_episode) - set(episode_map))
    if missing_target_episodes:
        raise RuntimeError(f"target episodes absent from dataset shard: {missing_target_episodes[:5]}")

    audit_root = _under_fjl(Path(args.candidate_audit_root))
    writer = AuditShardWriter(
        audit_root,
        shard_id=int(args.candidate_audit_shard_id),
        max_bytes=int(
            round(float(args.candidate_audit_max_gb_per_shard) * 1_000_000_000)
        ),
    )
    progress_file = base._prepare_progress_file(args, args.output_path)
    _migrate_progress_rpc_accounting(progress_file, writer)
    sucs, spls, oss, nes, done_set = base._load_progress(
        progress_file,
        expected_rpc_sampling_contract=base.build_rpc_progress_sampling_contract(
            protocol_seed=int(args.rpc_protocol_seed),
            require_deterministic_sampling=bool(
                args.rpc_require_deterministic_sampling
            ),
        ),
        expected_control_mode=args.expected_control_mode,
        expected_control_artifacts=control_artifacts,
    )
    pending = [
        key
        for key in (target_list or sorted(by_episode))
        if key in by_episode and key not in done_set
    ]
    if args.max_episodes is not None:
        pending = pending[: max(0, int(args.max_episodes))]
    print(
        "[continuation] "
        f"targets={target_path} states={len(targets)} episodes={len(by_episode)} "
        f"pending={len(pending)} resumed_records={writer.record_count} smoke={smoke_mode}",
        flush=True,
    )

    for episode_number, key in enumerate(pending, start=1):
        episode = episode_map[key]
        episode_targets = sorted(
            by_episode[key], key=lambda value: (int(value["system2_call_index"]), value["state_key"])
        )
        if smoke_mode:
            episode_targets = episode_targets[:1]
        episode_control = base.empty_control_counters()
        episode_rpc_calls = 0
        episode_trajectory_calls = 0
        representative: dict[str, Any] | None = None
        branch_count = 0
        print(
            f"[continuation] episode {episode_number}/{len(pending)} "
            f"{key[0]}_{key[1]:04d} states={len(episode_targets)}",
            flush=True,
        )
        for target in episode_targets:
            source_record = source_records[str(target["state_key"])]
            source_arrays = _load_source_arrays(source_record)
            role_map = dict(target["treatment_roles"])
            treatment_roles: dict[str, list[str]] = defaultdict(list)
            for role in ROLE_ORDER:
                treatment_roles[str(role_map[role])].append(role)
            treatment_ids = sorted(
                treatment_roles,
                key=lambda treatment_id: min(
                    ROLE_ORDER.index(role) for role in treatment_roles[treatment_id]
                ),
            )
            if smoke_mode:
                treatment_ids = treatment_ids[:2]
            for treatment_id in treatment_ids:
                branch_key = _branch_key(str(target["state_key"]), treatment_id)
                if writer.contains(branch_key):
                    row = dict(writer._records[branch_key])
                    _merge_counter_totals(episode_control, row.get("control_counters") or {})
                    print(f"  [resume] {branch_key}", flush=True)
                else:
                    record, arrays = _collect_branch(
                        client=client,
                        env=env,
                        episode=episode,
                        target=target,
                        source_record=source_record,
                        source_arrays=source_arrays,
                        treatment_id=treatment_id,
                        roles=treatment_roles[treatment_id],
                        run_to_end=(
                            bool(target["run_to_episode_end"]) and not smoke_mode
                        ),
                        args=args,
                        control_artifacts=control_artifacts,
                        image_size=image_size,
                        vlm_image_size=vlm_image_size,
                        traj_image_size=traj_image_size,
                        smoke_mode=smoke_mode,
                    )
                    row = writer.commit(
                        state_key=branch_key, record=record, arrays=arrays
                    )
                    _merge_counter_totals(episode_control, record["control_counters"])
                    print(
                        "  [branch] "
                        f"state={target['state_key']} roles={treatment_roles[treatment_id]} "
                        f"cycles={record['termination']['future_system2_cycles']} "
                        f"end={record['termination']['reason']} "
                        f"replay_t={record['replay_verification']['max_translation_error_m']:.2e}",
                        flush=True,
                    )
                branch_rpc_calls, branch_trajectory_calls = _continuation_rpc_counts(row)
                episode_rpc_calls += branch_rpc_calls
                episode_trajectory_calls += branch_trajectory_calls
                branch_count += 1
                if representative is None or "native_mean" in row.get("selector_roles", []):
                    representative = _representative_outcome(row)
            del source_arrays
        if representative is None:
            raise RuntimeError(f"episode produced no continuation branch: {key}")
        if episode_rpc_calls != int(episode_control["control_rpc_calls"]):
            raise RuntimeError(
                f"episode branch/RPC accounting mismatch for {key}: "
                f"{episode_rpc_calls} != {episode_control['control_rpc_calls']}"
            )
        result = {
            "scene_id": key[0],
            "episode_id": key[1],
            "success": float(representative["habitat_success"]),
            "spl": float(representative["habitat_spl"]),
            "os": float(representative["habitat_oracle_success"]),
            "ne": float(representative["habitat_distance_to_goal_m"]),
            "steps": int(representative["absolute_navigation_step_id"]),
            "episode_instruction": base._normalize_instruction(
                episode.instruction.instruction_text
            ),
            "vlm_calls": int(episode_rpc_calls),
            "trajectory_calls": int(episode_trajectory_calls),
            "continuation_branches": branch_count,
            "rpc_server": args.rpc_server,
            "rpc_protocol": base.HEATMAPVLN_RPC_PROTOCOL_VERSION,
            "rpc_sampling_protocol": base.HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
            "rpc_deterministic_sampling_enabled": True,
            "rpc_protocol_seed": int(args.rpc_protocol_seed),
            "rpc_require_deterministic_sampling": bool(
                args.rpc_require_deterministic_sampling
            ),
            "control_mode": str(args.expected_control_mode),
            **control_artifacts,
            **episode_control,
        }
        with open(progress_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        sucs.append(result["success"])
        spls.append(result["spl"])
        oss.append(result["os"])
        nes.append(result["ne"])
        done_set.add(key)

    manifest = writer.seal(
        extra={
            "collector": SCHEMA,
            "source_audit_root": str(source_root),
            "target_file": str(target_path),
            "target_file_sha256": _file_sha256(target_path),
            "continuation_policy": "one_deviation_then_frozen_native_pi0",
            "requested_horizons": [1] if smoke_mode else list(HORIZONS),
            "smoke_mode": bool(smoke_mode),
            "client_source_sha256": _file_sha256(Path(__file__)),
        }
    )
    env.close()
    final = base.aggregate_navigation_metrics(sucs, spls, oss, nes)
    final.update(
        {
            "continuation_schema": SCHEMA,
            "candidate_records": int(manifest["record_count"]),
            "candidate_array_bytes": int(manifest["array_bytes"]),
        }
    )
    result_path = Path(args.output_path) / "result.json"
    result_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def main() -> None:
    # Reuse the audited launcher's exact CLI/validation surface while replacing
    # only the RPC panoramic evaluation loop.
    base.run_eval_rpc_panoramic = run_eval_rpc_continuation
    base.main()


if __name__ == "__main__":
    main()
