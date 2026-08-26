from __future__ import annotations

import gzip
import io
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.evaluation import trajectory_dagger as td


ALLOWED_TMP_ROOT = Path("/mnt/afs/liwenhao/agent/370910109/tmp")
CONTRACT = {
    "dataset": "r2r",
    "split": "train",
    "collector": "trajectory_dagger",
    "stores_heatmaps": False,
}


@pytest.fixture
def collection_root():
    ALLOWED_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = ALLOWED_TMP_ROOT / f"trajectory_dagger_test_{uuid.uuid4().hex}"
    root.mkdir(parents=False, exist_ok=False)
    assert root.parent == ALLOWED_TMP_ROOT
    try:
        yield root
    finally:
        shutil.rmtree(root)


@pytest.fixture
def finalization_roots():
    ALLOWED_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    collection = ALLOWED_TMP_ROOT / f"trajectory_dagger_seal_{token}"
    control = ALLOWED_TMP_ROOT / f"trajectory_dagger_control_{token}"
    resources = ALLOWED_TMP_ROOT / f"trajectory_dagger_resources_{token}"
    for path in (collection, control, resources):
        path.mkdir(parents=False, exist_ok=False)
    try:
        yield collection, control, resources
    finally:
        for path in (collection, control, resources):
            shutil.rmtree(path)


def _validator_path() -> Path:
    return Path(td.__file__).resolve().parents[1] / "tools" / (
        "validate_trajectory_dagger_collection.py"
    )


def _pose(*, x: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def _yaw_pose(degrees: float, *, x: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = _pose(x=x, z=z)
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    pose[:3, :3] = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )
    return pose


def _observation(frame_id: int, *, z: float = 0.0) -> td.HistoryObservation:
    jpeg_buffer = io.BytesIO()
    pixels = np.full((8, 8, 3), frame_id % 256, dtype=np.uint8)
    Image.fromarray(pixels).save(jpeg_buffer, format="JPEG", quality=75)
    jpeg = jpeg_buffer.getvalue()
    return td.HistoryObservation(
        frame_id=frame_id,
        pose=_pose(z=z),
        view_jpegs={name: jpeg for name in td.VIEW_NAMES},
        primitive_step=frame_id,
        system2_call_index=frame_id,
    )


def _sample(
    sample_key: str,
    *,
    current_frame_id: int,
    trajectory_value: float = 0.0,
    history_frame_ids: list[int] | None = None,
) -> dict:
    history_ids = list(history_frame_ids or [])
    return {
        "key": sample_key,
        "source_type": "dagger_normal",
        "native_kind": "trajectory",
        "current_frame_id": current_frame_id,
        "history_frame_ids": history_ids,
        "history_valid_mask": [1] * len(history_ids),
        "history_age_steps": list(range(len(history_ids), 0, -1)),
        "trajectory": np.full((32, 3), trajectory_value, dtype=np.float32),
        "oracle_future_poses": np.stack([_pose(z=0.0), _pose(z=-0.5)]),
    }


def _record_one(root: Path, episode_key: str, *, trajectory_value: float = 0.0):
    state = td.prepare_collection(root, CONTRACT, resume=True)
    recorder = td.EpisodeTarRecorder(state)
    observation = _observation(0)
    return recorder.record_episode(
        episode_key=episode_key,
        episode_metadata={"scene_id": "test_scene"},
        observations=[observation],
        samples=[
            _sample(
                f"{episode_key}:0000",
                current_frame_id=observation.frame_id,
                trajectory_value=trajectory_value,
            )
        ],
    )


def _multiprocess_record_worker(root_text: str, episode_key: str) -> None:
    _record_one(Path(root_text), episode_key)


def _build_finalization_case(
    collection: Path,
    control: Path,
    resources: Path,
    *,
    episode_count: int,
    committed_indices: set[int],
    progress_count: int | None = None,
    reported_committed_indices: set[int] | None = None,
    scene_ids: list[str] | None = None,
    progress_order: list[int] | None = None,
) -> td.EpisodeTarRecorder:
    if scene_ids is None:
        scene_ids = [f"scene{index}" for index in range(episode_count)]
    if len(scene_ids) != episode_count:
        raise ValueError("scene_ids must match episode_count")
    if progress_order is None:
        progress_order = list(range(episode_count))
    if sorted(progress_order) != list(range(episode_count)):
        raise ValueError("progress_order must permute every episode index")
    episodes = [
        {
            "scene_id": f"mp3d/{scene_ids[index]}/{scene_ids[index]}.glb",
            "episode_id": index,
        }
        for index in range(episode_count)
    ]
    dataset_path = resources / "train.json.gz"
    with gzip.open(dataset_path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": episodes}, handle, sort_keys=True)
    cohort_path = resources / "cohort.json"
    cohort_path.write_text(
        json.dumps({"episodes": episodes}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract = {
        "schema": "heatmapvln-trajectory-dagger-contract-v3",
        "dataset_split": "train",
        "data_path": str(dataset_path),
        "data_sha256": td.sha256_file(dataset_path),
        "episode_cohort": {
            "path": str(cohort_path),
            "sha256": td.sha256_file(cohort_path),
            "max_episodes": 1,
        },
        "round_id": 0,
    }
    state = td.prepare_collection(collection, contract, resume=False)
    recorder = td.EpisodeTarRecorder(state)
    commits: dict[int, td.EpisodeCommit] = {}
    for index in sorted(committed_indices):
        observation = _observation(0)
        scene_id = scene_ids[index]
        key = f"round00_{scene_id}_{index:06d}"
        commits[index] = recorder.record_episode(
            episode_key=key,
            episode_metadata={"scene_id": scene_id, "episode_id": index},
            observations=[observation],
            samples=[_sample(f"{key}:0000", current_frame_id=0)],
        )

    reported = committed_indices if reported_committed_indices is None else reported_committed_indices
    count = episode_count if progress_count is None else progress_count
    progress_rows = []
    for index in progress_order[:count]:
        scene_id = scene_ids[index]
        key = f"round00_{scene_id}_{index:06d}"
        is_reported = index in reported
        row = {
            "scene_id": scene_id,
            "episode_id": index,
            "collect_trajectory_dagger": True,
            "trajectory_dagger_episode_key": key,
            "trajectory_dagger_committed": is_reported,
        }
        if is_reported:
            commit = commits[index]
            row["trajectory_dagger_commit"] = {
                "tar_sha256": commit.tar_sha256,
                "tar_bytes": commit.tar_bytes,
                "sample_count": commit.sample_count,
                "frame_count": commit.frame_count,
            }
        progress_rows.append(row)
    (control / "progress.json").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress_rows),
        encoding="utf-8",
    )
    (control / "result.json").write_text(
        json.dumps({"total_episodes": count}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return recorder


def _run_validator(
    collection: Path,
    *,
    control: Path | None = None,
    seal: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(_validator_path()),
        "--collection-root",
        str(collection),
        "--max-bytes",
        "100000000",
    ]
    if control is not None:
        command.extend(["--control-root", str(control)])
    if seal:
        command.append("--seal")
    return subprocess.run(command, capture_output=True, text=True)


class FakeShadowBackend:
    def __init__(self, pose: np.ndarray, *, fail_planner: bool = False) -> None:
        self.pose = pose.copy()
        self.goal = pose[:3, 3].copy()
        self.fail_planner = fail_planner
        self.restore_count = 0

    def snapshot(self):
        return self.pose.copy()

    def restore(self, snapshot) -> None:
        self.pose = np.asarray(snapshot, dtype=np.float32).copy()
        self.restore_count += 1

    def reset(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float32).copy()

    def get_pose(self) -> np.ndarray:
        return self.pose.copy()

    def next_action(self, goal_position: np.ndarray) -> int | None:
        if self.fail_planner:
            raise FakeGreedyFollowerError("no navigable path")
        self.goal = np.asarray(goal_position, dtype=np.float32).copy()
        return 1

    def step(self, action: int) -> np.ndarray:
        assert action == 1
        delta = self.goal[[0, 2]] - self.pose[[0, 2], 3]
        distance = float(np.linalg.norm(delta))
        if distance > 0.0:
            step = min(0.5, distance) * delta / distance
            self.pose[0, 3] += step[0]
            self.pose[2, 3] += step[1]
        return self.pose.copy()


class FakeGreedyFollowerError(RuntimeError):
    pass


def test_decimal_capacity_contract_is_exact() -> None:
    assert td.HARD_CAPACITY_BYTES == 300_000_000_000
    assert td.COMMIT_CEILING_BYTES == 295_000_000_000


def test_default_image_encoding_is_q75_and_history_sampling_matches_rpc() -> None:
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    default = td.encode_rgb_to_jpeg(image)
    q75 = td.encode_rgb_to_jpeg(image, quality=75)
    q74 = td.encode_rgb_to_jpeg(image, quality=74)

    assert default == q75
    assert default != q74
    decoded = Image.open(io.BytesIO(default))
    assert decoded.format == "JPEG"
    assert decoded.mode == "RGB"
    assert td.sample_history_indices(10, 4) == [0, 3, 6, 9]


def test_module_load_does_not_execute_src_data_package() -> None:
    module_path = Path(td.__file__).resolve()
    code = f"""
import importlib.abc
import importlib.util
import sys

class BlockSrcData(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'src.data' or fullname.startswith('src.data.'):
            raise AssertionError('src.data package import attempted: ' + fullname)
        return None

sys.meta_path.insert(0, BlockSrcData())
spec = importlib.util.spec_from_file_location('_trajectory_dagger_probe', {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.HARD_CAPACITY_BYTES == 300_000_000_000
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_route_progress_freezes_beyond_one_metre_off_path() -> None:
    tracker = td.MonotonicRouteTracker([[0.0, 0.0, 0.0], [0.0, 0.0, -10.0]])
    first = tracker.observe([0.0, 0.0, -1.0])
    off_route = tracker.observe([2.0, 0.0, -8.0])
    recovered = tracker.observe([0.0, 0.0, -8.0])

    assert first.progress_m == pytest.approx(1.0)
    assert off_route.offpath_m == pytest.approx(2.0)
    assert off_route.raw_progress_m == pytest.approx(8.0)
    assert off_route.progress_m == pytest.approx(first.progress_m)
    assert off_route.progress_delta_m == 0.0
    assert recovered.progress_m == pytest.approx(3.0)


def test_route_projection_disambiguates_overlapping_floors() -> None:
    tracker = td.MonotonicRouteTracker(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 3.0, -2.0], [0.0, 3.0, 0.0]]
    )
    offpath, progress = tracker.project([0.0, 3.0, -1.0])
    assert offpath == pytest.approx(0.0)
    assert progress == pytest.approx(6.0)


def test_path_disagreement_uses_equal_primitive_step_prefix() -> None:
    native = np.stack([_pose(z=value) for value in (0.0, -0.25, -0.5)])
    oracle = np.stack(
        [_pose(z=value) for value in (0.0, -0.25, -0.5, -0.75, -1.0, -1.25)]
    )

    assert td._path_disagreement(native, oracle) == pytest.approx(0.0)
    assert np.isinf(td._path_disagreement(native[:1], oracle))

    long_native = np.stack([_pose(z=-0.25 * index) for index in range(18)])
    long_oracle = long_native.copy()
    long_oracle[16, 0, 3] = 1.0
    assert td._path_disagreement(long_native, long_oracle) > 0.5


def test_wrong_branch_signal_uses_route_geometry_and_allows_recovery() -> None:
    tracker = td.MonotonicRouteTracker([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0]])
    oracle_poses = np.stack([_pose(z=value) for value in (0.0, -0.25, -0.5, -0.75)])
    oracle = td.OracleRelabelResult(
        trajectory=np.zeros((32, 3), dtype=np.float32),
        trajectory_valid=1.0,
        future_poses=oracle_poses,
        actions=(1, 1, 1),
        oracle_kind="route_recovery",
        terminal=False,
        route_progress_m=0.0,
        travelled_m=0.75,
    )
    history = np.empty((0, 4, 4), dtype=np.float32)

    on_route = td.build_candidate_signals(
        tracker,
        _pose(),
        history,
        oracle_poses[:3],
        oracle,
    )
    diverging = td.build_candidate_signals(
        tracker,
        _pose(),
        history,
        np.stack([_pose(), _pose(x=0.5), _pose(x=1.0)]),
        oracle,
    )
    recovering = td.build_candidate_signals(
        tracker,
        _pose(x=1.0, z=-1.0),
        history,
        np.stack(
            [
                _pose(x=1.0, z=-1.0),
                _pose(x=0.75, z=-0.75),
                _pose(x=0.5, z=-0.5),
            ]
        ),
        oracle,
    )

    assert not on_route.wrong_branch
    assert on_route.native_route_progress_delta_m == pytest.approx(0.5)
    assert diverging.wrong_branch
    assert diverging.native_endpoint_offpath_m == pytest.approx(1.0)
    assert not recovering.wrong_branch


def test_turn_only_heading_disagreement_has_normal_middle_and_hard_bands() -> None:
    tracker = td.MonotonicRouteTracker([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0]])
    oracle_poses = np.stack([_pose(), _pose()])
    oracle = td.OracleRelabelResult(
        trajectory=np.zeros((32, 3), dtype=np.float32),
        trajectory_valid=1.0,
        future_poses=oracle_poses,
        actions=(2,),
        oracle_kind="route_recovery",
        terminal=False,
        route_progress_m=0.0,
        travelled_m=0.0,
    )
    history = np.empty((0, 4, 4), dtype=np.float32)

    def selection(degrees: float) -> tuple[td.CandidateSignals, td.CandidateSelection]:
        signals = td.build_candidate_signals(
            tracker,
            _pose(),
            history,
            np.stack([_pose(), _yaw_pose(degrees)]),
            oracle,
        )
        return signals, td.classify_candidate(signals)

    normal_signals, normal = selection(10.0)
    middle_signals, middle = selection(30.0)
    hard_signals, hard = selection(60.0)

    assert normal_signals.native_oracle_heading_disagreement_deg == pytest.approx(10.0, abs=1e-4)
    assert normal.bucket == "dagger_normal"
    assert middle_signals.native_oracle_heading_disagreement_deg == pytest.approx(30.0, abs=1e-4)
    assert middle.bucket == "discard"
    assert hard_signals.native_oracle_heading_disagreement_deg == pytest.approx(60.0, abs=1e-4)
    assert hard.bucket == "dagger_hard"
    assert "heading_disagreement" in hard.tags


def test_pose_target_uses_habitat_minus_z_as_forward() -> None:
    poses = np.stack([_pose(z=z) for z in (0.0, -0.4, -0.8, -1.2)])
    trajectory, valid = td.poses_to_nextdit_target(poses)

    assert valid == 1.0
    assert trajectory.shape == (32, 3)
    assert np.isfinite(trajectory).all()
    assert float(trajectory[:, 0].sum()) > 0.0
    assert np.allclose(trajectory[:, 1], 0.0, atol=1e-5)


def test_shadow_oracle_generates_target_and_restores_real_state() -> None:
    initial = _pose(x=1.25, z=2.0)
    backend = FakeShadowBackend(initial)
    tracker = td.MonotonicRouteTracker([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0]])

    result = td.relabel_with_shadow_oracle(
        backend,
        route_tracker=tracker,
        current_pose=_pose(),
        route_progress_m=0.0,
        goal_position=[0.0, 0.0, -4.0],
        config=td.OracleRelabelConfig(
            target_path_length_m=1.0,
            anchor_lookahead_m=0.5,
            anchor_spacing_m=0.5,
            goal_tolerance_m=0.1,
        ),
    )

    assert result.valid
    assert result.oracle_kind == "route_recovery"
    assert result.travelled_m >= 1.0
    assert result.trajectory.shape == (32, 3)
    assert backend.restore_count == 1
    assert np.array_equal(backend.pose, initial)


def test_planner_errors_produce_invalid_relabel_and_still_restore() -> None:
    initial = _pose(x=3.0, z=2.0)
    backend = FakeShadowBackend(initial, fail_planner=True)
    tracker = td.MonotonicRouteTracker([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0]])

    result = td.relabel_with_shadow_oracle(
        backend,
        route_tracker=tracker,
        current_pose=_pose(),
        route_progress_m=0.0,
        goal_position=[0.0, 0.0, -4.0],
    )

    assert not result.valid
    assert result.oracle_kind == "goal_fallback"
    assert "FakeGreedyFollowerError" in str(result.fallback_reason)
    assert backend.restore_count == 1
    assert np.array_equal(backend.pose, initial)


def test_candidate_policy_retains_only_nextdit_trajectory_calls() -> None:
    normal = td.classify_candidate(td.CandidateSignals("trajectory", 0.1, 0.1, 0.1))
    hard = td.classify_candidate(td.CandidateSignals("trajectory", 1.1, 0.0, 0.1))
    stop = td.classify_candidate(td.CandidateSignals("stop", 2.0, 0.0, 2.0))

    assert normal.bucket == "dagger_normal"
    assert hard.bucket == "dagger_hard"
    assert "off_route" in hard.tags
    assert stop.bucket == "discard"


def test_episode_tar_schema_and_idempotent_resume(collection_root: Path) -> None:
    td.prepare_collection(collection_root, CONTRACT, resume=False)
    commit = _record_one(collection_root, "episode_0001")

    assert commit.sample_count == 1
    assert commit.frame_count == 1
    with tarfile.open(commit.tar_path, "r") as archive:
        names = set(archive.getnames())
        assert {
            "episode.json",
            "frames.jsonl",
            "samples.jsonl",
            "arrays/trajectories.npy",
            "arrays/oracle_future_poses.npy",
            "arrays/oracle_future_offsets.npy",
            *{f"frames/000000_{name}.jpg" for name in td.VIEW_NAMES},
        } <= names
        assert not any("heatmap" in name.lower() for name in names)
        trajectories = np.load(
            io.BytesIO(archive.extractfile("arrays/trajectories.npy").read()),
            allow_pickle=False,
        )
        sample_row = json.loads(archive.extractfile("samples.jsonl").readline())
    assert trajectories.shape == (1, 32, 3)
    assert sample_row["trajectory_index"] == 0
    assert sample_row["current_frame_id"] == 0

    repeated = _record_one(collection_root, "episode_0001")
    assert repeated.already_committed
    assert repeated.tar_sha256 == commit.tar_sha256


def test_standalone_validator_matches_writer_key_and_sorted_commit_contract(
    collection_root: Path,
) -> None:
    state = td.prepare_collection(collection_root, CONTRACT, resume=False)
    recorder = td.EpisodeTarRecorder(state)
    observation = _observation(0)
    recorder.record_episode(
        episode_key="episode_validator",
        episode_metadata={"scene_id": "test_scene"},
        observations=[observation],
        samples=[
            _sample("episode_validator:z", current_frame_id=0),
            _sample("episode_validator:a", current_frame_id=0),
        ],
    )

    validator = Path(td.__file__).resolve().parents[1] / "tools" / (
        "validate_trajectory_dagger_collection.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--collection-root",
            str(collection_root),
            "--max-bytes",
            "100000000",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "ok"
    assert result["episodes"] == 1
    assert result["samples"] == 2


def test_stable_episode_key_collision_fails_closed(collection_root: Path) -> None:
    td.prepare_collection(collection_root, CONTRACT, resume=False)
    _record_one(collection_root, "episode_collision", trajectory_value=0.0)

    with pytest.raises(RuntimeError, match="Stable episode key collision"):
        _record_one(collection_root, "episode_collision", trajectory_value=1.0)


def test_episode_metadata_cannot_override_commit_identity(collection_root: Path) -> None:
    state = td.prepare_collection(collection_root, CONTRACT, resume=False)
    recorder = td.EpisodeTarRecorder(state)
    with pytest.raises(ValueError, match="reserved fields"):
        recorder.record_episode(
            episode_key="episode_reserved",
            episode_metadata={"episode_key": "forged"},
            observations=[_observation(0)],
            samples=[_sample("episode_reserved:0000", current_frame_id=0)],
        )
    assert not (collection_root / "episodes" / "episode_reserved").exists()


def test_resume_rebuilds_missing_progress_ledger(collection_root: Path) -> None:
    td.prepare_collection(collection_root, CONTRACT, resume=False)
    commit = _record_one(collection_root, "episode_recover")
    ledger = collection_root / "collection_progress.jsonl"
    ledger.unlink()

    resumed = td.prepare_collection(collection_root, CONTRACT, resume=True, verify_commits=True)

    assert resumed.committed_episode_keys == frozenset({"episode_recover"})
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["episode_key"] == "episode_recover"
    assert rows[0]["tar_sha256"] == commit.tar_sha256


def test_resume_contract_mismatch_fails_closed(collection_root: Path) -> None:
    td.prepare_collection(collection_root, CONTRACT, resume=False)
    changed = {**CONTRACT, "split": "val_unseen"}
    with pytest.raises(RuntimeError, match="fingerprint"):
        td.prepare_collection(collection_root, changed, resume=True)


def test_episode_commit_ceiling_rejects_before_partial_commit(collection_root: Path) -> None:
    state = td.prepare_collection(
        collection_root,
        CONTRACT,
        resume=False,
        hard_capacity_bytes=100_000,
        commit_ceiling_bytes=2_000,
    )
    recorder = td.EpisodeTarRecorder(state)
    with pytest.raises(td.CapacityExceededError, match="ceiling"):
        recorder.record_episode(
            episode_key="too_large",
            episode_metadata={},
            observations=[_observation(0)],
            samples=[_sample("too_large:0000", current_frame_id=0)],
        )
    assert not (collection_root / "episodes" / "too_large").exists()


def test_same_host_multiprocess_commits_share_one_locked_ledger(collection_root: Path) -> None:
    td.prepare_collection(collection_root, CONTRACT, resume=False)
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_multiprocess_record_worker,
            args=(str(collection_root), f"episode_rank_{rank}"),
        )
        for rank in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    resumed = td.prepare_collection(collection_root, CONTRACT, resume=True, verify_commits=True)
    expected_keys = frozenset(f"episode_rank_{rank}" for rank in range(8))
    assert resumed.committed_episode_keys == expected_keys
    rows = [
        json.loads(line)
        for line in (collection_root / "collection_progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["episode_key"] for row in rows] == sorted(expected_keys)
    assert td.logical_usage_bytes(collection_root) <= td.COMMIT_CEILING_BYTES


def test_complete_control_cohort_seals_atomically_and_is_idempotent(
    finalization_roots,
) -> None:
    collection, control, resources = finalization_roots
    recorder = _build_finalization_case(
        collection,
        control,
        resources,
        episode_count=2,
        committed_indices={0},
    )

    first = _run_validator(collection, control=control, seal=True)
    assert first.returncode == 0, first.stderr
    first_result = json.loads(first.stdout)
    assert first_result["manifest_ready"] is True
    assert first_result["sealed_now"] is True
    manifest_path = collection / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["ready"] is True
    assert manifest["summary"]["expected_episodes"] == 2
    assert manifest["summary"]["committed_episodes"] == 1
    assert manifest["summary"]["no_sample_episodes"] == 1

    sealed_bytes = manifest_path.read_bytes()
    second = _run_validator(collection, control=control, seal=True)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["sealed_now"] is False
    assert manifest_path.read_bytes() == sealed_bytes

    with pytest.raises(RuntimeError, match="sealed"):
        recorder.record_episode(
            episode_key="round00_scene1_000001",
            episode_metadata={"scene_id": "scene1", "episode_id": 1},
            observations=[_observation(0)],
            samples=[_sample("round00_scene1_000001:0000", current_frame_id=0)],
        )
    assert not (collection / "episodes" / "round00_scene1_000001").exists()


def test_habitat_scene_grouped_control_order_can_seal(
    finalization_roots,
) -> None:
    collection, control, resources = finalization_roots
    _build_finalization_case(
        collection,
        control,
        resources,
        episode_count=3,
        committed_indices={0, 1, 2},
        scene_ids=["scene_a", "scene_b", "scene_a"],
        progress_order=[0, 2, 1],
    )

    completed = _run_validator(collection, control=control, seal=True)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(
        (collection / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["ready"] is True
    assert manifest["summary"]["processed_episodes"] == 3


def test_validator_rejects_non_habitat_grouped_control_order(
    finalization_roots,
) -> None:
    collection, control, resources = finalization_roots
    _build_finalization_case(
        collection,
        control,
        resources,
        episode_count=3,
        committed_indices=set(),
        scene_ids=["scene_a", "scene_b", "scene_a"],
        progress_order=[1, 0, 2],
    )

    completed = _run_validator(collection, control=control, seal=True)

    assert completed.returncode == 2
    assert "out of Habitat scene-grouped cohort order" in completed.stderr


def test_partial_control_cannot_seal_and_manifest_is_unchanged(
    finalization_roots,
) -> None:
    collection, control, resources = finalization_roots
    _build_finalization_case(
        collection,
        control,
        resources,
        episode_count=2,
        committed_indices={0},
        progress_count=1,
    )
    manifest_path = collection / "collection_manifest.json"
    before = manifest_path.read_bytes()

    completed = _run_validator(collection, control=control, seal=True)

    assert completed.returncode == 2
    assert "incomplete" in completed.stderr
    assert manifest_path.read_bytes() == before
    assert json.loads(before)["ready"] is False


def test_fully_processed_zero_commit_collection_can_seal(
    finalization_roots,
) -> None:
    collection, control, resources = finalization_roots
    _build_finalization_case(
        collection,
        control,
        resources,
        episode_count=2,
        committed_indices=set(),
    )
    assert not (collection / "collection_progress.jsonl").exists()

    completed = _run_validator(collection, control=control, seal=True)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(
        (collection / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["summary"]["committed_episodes"] == 0
    assert manifest["summary"]["no_sample_episodes"] == 2


def test_commit_before_control_append_recovery_can_seal(
    finalization_roots,
) -> None:
    collection, control, resources = finalization_roots
    _build_finalization_case(
        collection,
        control,
        resources,
        episode_count=1,
        committed_indices={0},
        reported_committed_indices=set(),
    )

    completed = _run_validator(collection, control=control, seal=True)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(
        (collection / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["summary"]["committed_episodes"] == 1
    assert manifest["summary"]["no_sample_episodes"] == 0


def test_validator_recomputes_manifest_identity(collection_root: Path) -> None:
    td.prepare_collection(collection_root, CONTRACT, resume=False)
    _record_one(collection_root, "episode_identity")
    manifest_path = collection_root / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract"]["split"] = "tampered"
    manifest_path.write_bytes(td.canonical_json_bytes(manifest, newline=True))

    completed = _run_validator(collection_root)

    assert completed.returncode == 2
    assert "fingerprint" in completed.stderr


def test_validator_rejects_extra_heatmap_array_member(collection_root: Path) -> None:
    td.prepare_collection(collection_root, CONTRACT, resume=False)
    commit = _record_one(collection_root, "episode_extra_array")
    temporary = commit.tar_path.with_name("episode.repacked.tar")
    with tarfile.open(commit.tar_path, "r:") as source, tarfile.open(
        temporary, "w", format=tarfile.USTAR_FORMAT
    ) as target:
        for member in source.getmembers():
            extracted = source.extractfile(member)
            payload = extracted.read() if extracted is not None else None
            target.addfile(member, io.BytesIO(payload) if payload is not None else None)
        array_buffer = io.BytesIO()
        np.save(array_buffer, np.zeros((1, 1), dtype=np.float32), allow_pickle=False)
        payload = array_buffer.getvalue()
        extra = tarfile.TarInfo("arrays/heatmap.npy")
        extra.size = len(payload)
        target.addfile(extra, io.BytesIO(payload))
    temporary.replace(commit.tar_path)

    commit_path = commit.tar_path.with_name("commit.json")
    marker = json.loads(commit_path.read_text(encoding="utf-8"))
    marker.update(
        {
            "tar_sha256": td.sha256_file(commit.tar_path),
            "tar_bytes": commit.tar_path.stat().st_size,
        }
    )
    marker_bytes = td.canonical_json_bytes(marker, newline=True)
    commit_path.write_bytes(marker_bytes)
    (collection_root / "collection_progress.jsonl").write_bytes(marker_bytes)

    completed = _run_validator(collection_root)

    assert completed.returncode == 2
    assert "not official" in completed.stderr


NATIVE_POLICY_FINGERPRINT = "internnav-native-v1:" + "b" * 64
NATIVE_PROTOCOL = "internnav-native-joint-front-history-lookdown-v1"


def _sized_jpeg(width: int, height: int, value: int) -> bytes:
    output = io.BytesIO()
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(output, format="JPEG", quality=75)
    return output.getvalue()


def _record_native_validator_case(
    root: Path,
    *,
    system1_lookdown_size: tuple[int, int] = (224, 224),
    include_current_lookdown: bool = True,
) -> None:
    contract = {
        "rpc_policy_mode": "internnav_native",
        "rpc_policy_fingerprint": NATIVE_POLICY_FINGERPRINT,
        "native_protocol": NATIVE_PROTOCOL,
        "observation": {
            "vlm_image_size": [384, 384],
            "lookdown_image_size": [640, 480],
            "system1_lookdown_image_size": list(
                system1_lookdown_size
            ),
        },
    }
    state = td.prepare_collection(root, contract, resume=False)
    view_jpeg = _sized_jpeg(384, 384, 32)
    lookdown_jpeg = (
        _sized_jpeg(640, 480, 64)
        if include_current_lookdown
        else None
    )
    observation = td.HistoryObservation(
        frame_id=0,
        pose=_pose(),
        view_jpegs={
            name: view_jpeg for name in td.VIEW_NAMES
        },
        primitive_step=0,
        system2_call_index=0,
        lookdown_jpeg=lookdown_jpeg,
    )
    sample = _sample(
        "native_validator:0000",
        current_frame_id=0,
    )
    sample["native"] = {
        "policy_backend": "internnav_native",
        "policy_fingerprint": NATIVE_POLICY_FINGERPRINT,
        "native_protocol": NATIVE_PROTOCOL,
        "native_front_only": True,
        "native_checkpoint_only": True,
        "system2_source": "internnav_native",
        "system1_source": "internnav_native_nextdit_async",
        "trajectory_x_sign": 1.0,
        "trajectory_heading_alignment": "none",
        "native_lookdown_turns": 0,
    }
    td.EpisodeTarRecorder(state).record_episode(
        episode_key="native_validator",
        episode_metadata={"scene_id": "native_scene"},
        observations=[observation],
        samples=[sample],
    )


def test_validator_native_requires_system1_lookdown_224(
    collection_root: Path,
) -> None:
    _record_native_validator_case(
        collection_root,
        system1_lookdown_size=(4, 6),
    )

    completed = _run_validator(collection_root)

    assert completed.returncode == 2
    assert (
        "system1_lookdown_image_size=[224,224]"
        in completed.stderr
    )


def test_validator_native_requires_current_lookdown(
    collection_root: Path,
) -> None:
    _record_native_validator_case(
        collection_root,
        include_current_lookdown=False,
    )

    completed = _run_validator(collection_root)

    assert completed.returncode == 2
    assert "must contain a lookdown observation" in completed.stderr
