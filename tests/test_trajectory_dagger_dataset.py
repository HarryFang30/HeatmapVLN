from __future__ import annotations

import io
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from scripts.evaluation import trajectory_dagger as td
from src.data.factory import build_dataset
from src.data.trajectory_dagger_dataset import (
    DeterministicMixtureSampler,
    IndexedSourceDataset,
    SourceMixtureDataset,
    TrajectoryDaggerDataset,
    trajectory_dagger_collate_fn,
)


ALLOWED_TMP_ROOT = Path("/mnt/afs/lixiaoou/intern/fjl/tmp")
CONTRACT = {
    "schema": "heatmapvln-trajectory-dagger-contract-test-v1",
    "observation": {
        "view_order": list(td.VIEW_NAMES),
        "vlm_image_size": [8, 8],
        "lookdown_image_size": [8, 8],
        "jpeg_quality": 75,
        "num_history": 8,
        "history_sampler": "endpoint_linspace_unique_v1",
    },
    "target": {
        "predict_horizon": 32,
        "action_scale": 4.0,
        "camera_forward_axis": "-z",
    },
}
NATIVE_FINGERPRINT = "internnav-native-v1:" + "a" * 64
NATIVE_PROTOCOL = "internnav-native-joint-front-history-lookdown-v1"


@pytest.fixture
def collection_workspace():
    ALLOWED_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = (
        ALLOWED_TMP_ROOT
        / f"trajectory_dagger_dataset_test_{uuid.uuid4().hex}"
    )
    root.mkdir(parents=False, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _pose(frame_id: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = -0.25 * frame_id
    return pose


def _jpeg(value: int) -> bytes:
    pixels = np.full((8, 8, 3), value % 256, dtype=np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(
        output,
        format="JPEG",
        quality=75,
    )
    return output.getvalue()


def _observation(
    frame_id: int,
    *,
    current_lookdown_jpeg: bytes | None = None,
) -> td.HistoryObservation:
    views = {
        view: _jpeg(frame_id * 16 + view_index)
        for view_index, view in enumerate(td.VIEW_NAMES)
    }
    return td.HistoryObservation(
        frame_id=frame_id,
        pose=_pose(frame_id),
        view_jpegs=views,
        primitive_step=frame_id,
        system2_call_index=frame_id,
        lookdown_jpeg=(
            (
                current_lookdown_jpeg
                if current_lookdown_jpeg is not None
                else _jpeg(240)
            )
            if frame_id == 8
            else None
        ),
    )


def _sample(
    *,
    key: str,
    source_type: str,
    value: float,
    native: dict | None = None,
) -> dict:
    return {
        "key": key,
        "source_type": source_type,
        "native_kind": "trajectory",
        "scene_id": "synthetic_scene",
        "episode_id": "synthetic_episode",
        "instruction": "walk to the end of the hall",
        "current_frame_id": 8,
        "current_camera_pose": _pose(8),
        "current_agent_pose": _pose(8),
        "history_frame_ids": list(range(8)),
        "history_valid_mask": [1, 1, 1, 1, 0, 1, 1, 1],
        "history_age_steps": list(range(8, 0, -1)),
        "trajectory": np.full(
            (32, 3),
            value,
            dtype=np.float32,
        ),
        "trajectory_valid": 1.0,
        "oracle_future_poses": np.stack(
            [_pose(8), _pose(9)],
            axis=0,
        ),
        "native": (
            {
                "actions": [2, 1],
                "pano_goal_view": "front",
                "pixel_goal": [3, 4],
            }
            if native is None
            else native
        ),
        "oracle": {"kind": "synthetic"},
        "candidate_signals": {"offpath_m": value},
        "failure_tags": (
            ["wrong_branch"]
            if source_type == "dagger_hard"
            else []
        ),
    }


def _set_ready(root: Path, ready: bool) -> None:
    path = root / "collection_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["ready"] = bool(ready)
    path.write_bytes(
        td.canonical_json_bytes(manifest, newline=True)
    )


def _make_collection(
    root: Path,
    *,
    key_prefix: str,
    ready: bool = True,
    contract: dict | None = None,
    native: dict | None = None,
    current_lookdown_jpeg: bytes | None = None,
) -> Path:
    state = td.prepare_collection(
        root,
        CONTRACT if contract is None else contract,
        resume=False,
    )
    recorder = td.EpisodeTarRecorder(state)
    recorder.record_episode(
        episode_key=f"{key_prefix}_episode",
        episode_metadata={"scene_id": "synthetic_scene"},
        observations=[
            _observation(
                index,
                current_lookdown_jpeg=current_lookdown_jpeg,
            )
            for index in range(9)
        ],
        samples=[
            _sample(
                key=f"{key_prefix}:normal",
                source_type="dagger_normal",
                value=1.0,
                native=native,
            ),
            _sample(
                key=f"{key_prefix}:hard",
                source_type="dagger_hard",
                value=2.0,
                native=native,
            ),
        ],
    )
    _set_ready(root, ready)
    return root


def test_unsealed_collection_is_rejected_by_default(
    collection_workspace: Path,
) -> None:
    root = _make_collection(
        collection_workspace,
        key_prefix="unsealed",
        ready=False,
    )

    with pytest.raises(RuntimeError, match="not sealed"):
        TrajectoryDaggerDataset(root)

    debug = TrajectoryDaggerDataset(
        root,
        allow_unsealed_debug=True,
    )
    assert len(debug) == 2


def test_lazy_dataset_and_fixed_training_contract(
    collection_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_collection(
        collection_workspace,
        key_prefix="sealed",
    )
    calls = 0
    original = TrajectoryDaggerDataset._decode_jpeg

    def counted_decode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        TrajectoryDaggerDataset,
        "_decode_jpeg",
        staticmethod(counted_decode),
    )
    dataset = TrajectoryDaggerDataset(
        root,
        verify_tar_sha256=True,
    )
    assert calls == 0
    assert dataset.sample_keys == (
        "sealed:normal",
        "sealed:hard",
    )
    assert dataset.source_indices == {
        "dagger_normal": (0,),
        "dagger_hard": (1,),
    }

    normal = dataset[0]
    hard = dataset[1]
    assert calls > 0
    assert normal["current_views"].shape == (4, 3, 8, 8)
    assert normal["current_frame"].shape == (3, 8, 8)
    assert normal["history_panoramas"].shape == (
        8,
        4,
        3,
        8,
        8,
    )
    assert normal["history_frames"].shape == (8, 3, 8, 8)
    assert normal["history_poses"].shape == (8, 4, 4)
    assert normal["history_rel_poses"].shape == (8, 4)
    assert normal["history_frame_ids"].tolist() == list(range(8))
    assert normal["history_valid_mask"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert normal["history_age_steps"].tolist() == list(
        range(8, 0, -1)
    )
    assert normal["trajectory"].shape == (32, 3)
    assert torch.all(normal["trajectory"] == 1.0)
    assert torch.all(hard["trajectory"] == 2.0)
    assert normal["sample_key"] == "sealed:normal"
    assert hard["source_type"] == "dagger_hard"
    assert normal["lookdown_frame"].shape == (3, 8, 8)
    assert "heatmap" not in normal
    assert "depth" not in normal

    batch = trajectory_dagger_collate_fn([normal, hard])
    assert batch["current_views"].shape == (2, 4, 3, 8, 8)
    assert batch["history_panoramas"].shape == (
        2,
        8,
        4,
        3,
        8,
        8,
    )
    assert batch["history_valid_mask"].shape == (2, 8)
    assert batch["trajectory"].shape == (2, 32, 3)
    assert batch["sample_key"] == [
        "sealed:normal",
        "sealed:hard",
    ]
    assert batch["source_type"] == [
        "dagger_normal",
        "dagger_hard",
    ]
    assert "heatmap" not in batch


def test_native_raw_lookdown_uses_bicubic_and_factory_policy_gate(
    collection_workspace: Path,
) -> None:
    native_contract = json.loads(json.dumps(CONTRACT))
    native_contract.update(
        {
            "rpc_policy_mode": "internnav_native",
            "rpc_policy_fingerprint": NATIVE_FINGERPRINT,
            "native_protocol": NATIVE_PROTOCOL,
        }
    )
    native_contract["observation"]["lookdown_image_size"] = [640, 480]
    native_contract["observation"]["system1_lookdown_image_size"] = [
        224,
        224,
    ]
    native = {
        "actions": [2, 1],
        "pano_goal_view": "front",
        "pixel_goal": [3, 4],
        "policy_backend": "internnav_native",
        "policy_fingerprint": NATIVE_FINGERPRINT,
        "native_protocol": NATIVE_PROTOCOL,
        "native_front_only": True,
        "native_checkpoint_only": True,
        "system2_source": "internnav_native",
        "system1_source": "internnav_native_nextdit_async",
        "trajectory_x_sign": 1.0,
        "trajectory_heading_alignment": "none",
        "native_lookdown_turns": 1,
    }
    pixels = np.random.default_rng(7).integers(
        0,
        256,
        size=(480, 640, 3),
        dtype=np.uint8,
    )
    jpeg_buffer = io.BytesIO()
    Image.fromarray(pixels).save(
        jpeg_buffer,
        format="JPEG",
        quality=75,
    )
    lookdown_jpeg = jpeg_buffer.getvalue()
    root = _make_collection(
        collection_workspace,
        key_prefix="native_lookdown",
        contract=native_contract,
        native=native,
        current_lookdown_jpeg=lookdown_jpeg,
    )
    cfg = {
        "data": {
            "dataset_type": "trajectory_dagger",
            "image_size": [8, 8],
            "trajectory_dagger": {
                "collection_roots": [str(root)],
                "expected_policy_mode": "internnav_native",
                "expected_policy_fingerprint": NATIVE_FINGERPRINT,
            },
        },
    }

    dataset = build_dataset(cfg, split="train")
    assert dataset.require_lookdown is True
    assert dataset.expected_policy_fingerprint == NATIVE_FINGERPRINT
    assert dataset.lookdown_image_size == (640, 480)
    assert dataset.system2_lookdown_image_size == (640, 480)
    assert dataset.system1_lookdown_image_size == (224, 224)
    assert dataset.lookdown_output_size == (224, 224)
    sample = dataset[0]
    assert sample["lookdown_frame"].shape == (3, 480, 640)
    assert sample["traj_images"].shape == (2, 224, 224, 3)
    assert torch.equal(
        sample["traj_images"][0],
        sample["traj_images"][1],
    )
    actual_system2 = (
        sample["lookdown_frame"]
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    actual_system1 = (
        sample["traj_images"][0]
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    with Image.open(io.BytesIO(lookdown_jpeg)) as image:
        rgb = image.convert("RGB")
        expected_system2 = np.asarray(rgb)
        expected_system1 = np.asarray(
            rgb.resize((224, 224), Image.Resampling.BICUBIC)
        )
        bilinear = np.asarray(
            rgb.resize((224, 224), Image.Resampling.BILINEAR)
        )
    assert np.array_equal(actual_system2, expected_system2)
    assert np.array_equal(actual_system1, expected_system1)
    assert not np.array_equal(actual_system1, bilinear)


def test_native_policy_gate_rejects_legacy_and_bad_sample_provenance(
    collection_workspace: Path,
) -> None:
    legacy = _make_collection(
        collection_workspace / "legacy",
        key_prefix="legacy",
    )
    with pytest.raises(ValueError, match="exact"):
        TrajectoryDaggerDataset(
            legacy,
            expected_policy_mode="internnav_native",
        )
    with pytest.raises(ValueError, match="policy mode mismatch"):
        TrajectoryDaggerDataset(
            legacy,
            expected_policy_mode="internnav_native",
            expected_policy_fingerprint=NATIVE_FINGERPRINT,
        )

    native_contract = json.loads(json.dumps(CONTRACT))
    native_contract.update(
        {
            "rpc_policy_mode": "internnav_native",
            "rpc_policy_fingerprint": NATIVE_FINGERPRINT,
            "native_protocol": NATIVE_PROTOCOL,
        }
    )
    native_contract["observation"]["lookdown_image_size"] = [640, 480]
    native_contract["observation"]["system1_lookdown_image_size"] = [
        224,
        224,
    ]
    bad = _make_collection(
        collection_workspace / "bad_native",
        key_prefix="bad_native",
        contract=native_contract,
        native={"actions": [1]},
    )
    with pytest.raises(ValueError, match="sample policy provenance"):
        TrajectoryDaggerDataset(
            bad,
            expected_policy_mode="internnav_native",
            expected_policy_fingerprint=NATIVE_FINGERPRINT,
        )


def test_multiple_sealed_roots_are_indexed_without_copying(
    collection_workspace: Path,
) -> None:
    first = _make_collection(
        collection_workspace / "first",
        key_prefix="first",
    )
    second = _make_collection(
        collection_workspace / "second",
        key_prefix="second",
    )

    dataset = TrajectoryDaggerDataset([first, second])
    assert len(dataset) == 4
    assert dataset.sample_keys == (
        "first:normal",
        "first:hard",
        "second:normal",
        "second:hard",
    )
    assert dataset.source_indices == {
        "dagger_normal": (0, 2),
        "dagger_hard": (1, 3),
    }


def test_factory_dispatches_trajectory_dagger(
    collection_workspace: Path,
) -> None:
    root = _make_collection(
        collection_workspace,
        key_prefix="factory",
    )
    cfg = {
        "data": {
            "dataset_type": "trajectory_dagger",
            "image_size": [8, 8],
            "trajectory_dagger": {
                "collection_roots": [str(root)],
            },
        },
    }
    dataset = build_dataset(cfg, split="train")
    assert isinstance(dataset, TrajectoryDaggerDataset)
    assert len(dataset) == 2


class _TinyDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return {"value": index}


class _CapabilityDataset(_TinyDataset):
    def __init__(
        self,
        length: int,
        *,
        is_panoramic: bool,
        single_view_rgb_input: bool,
        dynamic_sampling_enabled: bool,
    ) -> None:
        super().__init__(length)
        self._is_panoramic = is_panoramic
        self.single_view_rgb_input = single_view_rgb_input
        self.dynamic_sampling_enabled = dynamic_sampling_enabled
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def test_indexed_sources_preserve_observation_and_sampling_contract() -> None:
    expert = _CapabilityDataset(
        3,
        is_panoramic=True,
        single_view_rgb_input=False,
        dynamic_sampling_enabled=True,
    )
    dagger = _CapabilityDataset(
        2,
        is_panoramic=True,
        single_view_rgb_input=False,
        dynamic_sampling_enabled=False,
    )
    indexed_expert = IndexedSourceDataset(
        expert,
        (0, 1, 2),
        "expert",
    )
    indexed_dagger = IndexedSourceDataset(
        dagger,
        (0, 1),
        "dagger_normal",
    )
    mixture = SourceMixtureDataset(
        {
            "expert": indexed_expert,
            "dagger_normal": indexed_dagger,
        }
    )

    assert indexed_expert._is_panoramic is True
    assert indexed_expert.single_view_rgb_input is False
    assert indexed_expert.dynamic_sampling_enabled is True
    assert mixture._is_panoramic is True
    assert mixture.single_view_rgb_input is False
    assert mixture.dynamic_sampling_enabled is True

    mixture.set_epoch(4)
    assert expert.epochs == [4]
    assert dagger.epochs == [4]


def _source_counts(
    dataset: SourceMixtureDataset,
    plan: tuple[int, ...],
) -> Counter:
    return Counter(
        dataset[index]["source_type"] for index in plan
    )


def test_weighted_sampler_is_deterministic_ddp_safe_and_resumable() -> None:
    mixture = SourceMixtureDataset(
        {
            "expert": _TinyDataset(10),
            "dagger_normal": _TinyDataset(4),
            "dagger_hard": _TinyDataset(6),
        }
    )
    sampler = DeterministicMixtureSampler(
        mixture,
        epoch_size=10,
        seed=123,
        num_replicas=2,
        rank=0,
    )
    assert sampler.profile == "expert50_normal20_hard30"
    assert sampler.source_counts_for_epoch() == {
        "expert": 5,
        "dagger_normal": 2,
        "dagger_hard": 3,
    }
    epoch_zero = sampler.global_plan()
    assert _source_counts(mixture, epoch_zero) == {
        "expert": 5,
        "dagger_normal": 2,
        "dagger_hard": 3,
    }
    rank_zero = tuple(iter(sampler))
    rank_one_sampler = DeterministicMixtureSampler(
        mixture,
        epoch_size=10,
        seed=123,
        num_replicas=2,
        rank=1,
    )
    rank_one = tuple(iter(rank_one_sampler))
    assert rank_zero == epoch_zero[0::2]
    assert rank_one == epoch_zero[1::2]
    rank_one_sampler.load_state_dict(sampler.state_dict())
    assert rank_one_sampler.rank == 1
    assert tuple(iter(rank_one_sampler)) == epoch_zero[1::2]

    sampler.set_epoch(7)
    epoch_seven = sampler.global_plan()
    assert epoch_seven != epoch_zero
    state = sampler.state_dict()
    restored = DeterministicMixtureSampler(
        mixture,
        epoch_size=10,
        seed=123,
        num_replicas=2,
        rank=0,
    )
    restored.load_state_dict(state)
    assert restored.global_plan() == epoch_seven
    assert restored.state_dict() == state


def test_empty_positive_bucket_fails_closed_and_profile_is_recorded() -> None:
    mixture = SourceMixtureDataset(
        {
            "expert": _TinyDataset(10),
            "dagger_normal": _TinyDataset(0),
            "dagger_hard": _TinyDataset(6),
        }
    )
    with pytest.raises(
        ValueError,
        match="non-empty source 'dagger_normal'",
    ):
        DeterministicMixtureSampler(
            mixture,
            epoch_size=10,
        )

    sampler = DeterministicMixtureSampler(
        mixture,
        profile="expert60_hard40",
        epoch_size=10,
    )
    assert sampler.source_counts_for_epoch() == {
        "expert": 6,
        "dagger_normal": 0,
        "dagger_hard": 4,
    }
    assert sampler.state_dict()["profile"] == "expert60_hard40"
    assert sampler.state_dict()["weights"] == {
        "expert": 0.6,
        "dagger_normal": 0.0,
        "dagger_hard": 0.4,
    }
