import json
import math
from types import SimpleNamespace

import pytest
import torch

from src.models.adapters import GeometryAwarePanoToNextDiTAdapter, view_ids_to_indices
from src.models.adapters.pano_latent_adapter import GEOMETRY_CONVENTION_LEGACY_CAMERA
from src.data.pano_teacher_alignment import (
    NATIVE_TEACHER_ALIGNMENT_VERSION,
    NATIVE_TEACHER_SIDECAR_SCHEMA,
    aligned_native_sidecar_contract,
    sidecar_alignment_metadata,
)
from scripts.training.train_pano_latent_adapter import (
    AdapterTrainBatch,
    _compute_adapter_objective,
    _filter_records_with_pano_goals,
    _load_validated_tensor_sidecar_payload,
    _load_teacher_latents,
    _load_teacher_records,
    _sample_from_record,
    _split_train_val,
)


def test_geometry_scalars_use_view_yaw_and_pixel_offset():
    view_indices = view_ids_to_indices(["front", "right", "back", "left"])
    pixel_xy = torch.tensor(
        [
            [128.0, 128.0],
            [128.0, 128.0],
            [128.0, 128.0],
            [128.0, 128.0],
        ]
    )
    image_hw = torch.tensor([[256.0, 256.0]]).expand(4, 2)

    scalars = GeometryAwarePanoToNextDiTAdapter.geometry_scalars(
        view_indices,
        pixel_xy,
        image_hw,
    )

    assert torch.allclose(scalars[:, :2], torch.full((4, 2), 0.5))
    assert torch.allclose(scalars[0, 2:], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(scalars[1, 2:], torch.tensor([-1.0, 0.0]), atol=1e-6)
    assert torch.allclose(scalars[2, 2:], torch.tensor([0.0, -1.0]), atol=1e-6)
    assert torch.allclose(scalars[3, 2:], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_geometry_scalars_apply_horizontal_fov():
    view_indices = view_ids_to_indices(["front"])
    pixel_xy = torch.tensor([[192.0, 128.0]])
    image_hw = torch.tensor([[256.0, 256.0]])

    scalars = GeometryAwarePanoToNextDiTAdapter.geometry_scalars(
        view_indices,
        pixel_xy,
        image_hw,
        horizontal_fov_deg=90.0,
    )

    expected_theta = math.radians(-22.5)
    assert torch.allclose(
        scalars[0, 2:],
        torch.tensor([math.sin(expected_theta), math.cos(expected_theta)]),
        atol=1e-6,
    )


def test_geometry_scalars_preserve_explicit_legacy_camera_convention():
    view_indices = view_ids_to_indices(["right"])
    scalars = GeometryAwarePanoToNextDiTAdapter.geometry_scalars(
        view_indices,
        torch.tensor([[192.0, 128.0]]),
        torch.tensor([[256.0, 256.0]]),
        horizontal_fov_deg=90.0,
        geometry_convention=GEOMETRY_CONVENTION_LEGACY_CAMERA,
    )
    expected_theta = math.radians(112.5)
    assert torch.allclose(
        scalars[0, 2:],
        torch.tensor([math.sin(expected_theta), math.cos(expected_theta)]),
        atol=1e-6,
    )


def test_geometry_aware_adapter_outputs_nextdit_condition_shape_and_grad():
    adapter = GeometryAwarePanoToNextDiTAdapter(
        student_dim=8,
        adapter_dim=16,
        output_dim=6,
        num_query=4,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        geometry_embed_dim=4,
    )
    student_latents = torch.randn(2, 4, 8)
    view_indices = view_ids_to_indices(["front", "right"])
    pixel_xy = torch.tensor([[128.0, 128.0], [200.0, 64.0]])
    image_hw = torch.tensor([[256.0, 256.0], [256.0, 256.0]])

    out = adapter(student_latents, view_indices, pixel_xy, image_hw)

    assert out.shape == (2, 4, 6)
    out.square().mean().backward()
    assert adapter.output_queries.grad is not None


def test_native_teacher_loader_rejects_legacy_dataset_sidecar(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "dataset_index": 0,
                "teacher": {
                    "coord_source": "dataset",
                    "mode": "dataset_coord",
                    "coord_uv": [151, 202],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _load_teacher_records(
        path,
        require_tensor=False,
        require_coord_uv=False,
        require_native_teacher=True,
    ) == []


def test_native_v2_tensor_fingerprint_must_match_jsonl(tmp_path):
    tensor_path = tmp_path / "teacher.pt"
    contract = {"goal_frame_idx": 12}
    torch.save(
        {
            "dataset_index": 0,
            "sidecar_schema": NATIVE_TEACHER_SIDECAR_SCHEMA,
            "alignment_version": NATIVE_TEACHER_ALIGNMENT_VERSION,
            "alignment_fingerprint": "tensor-fingerprint",
            "alignment_contract": contract,
            "traj_latents": torch.ones(1, 4, 6),
        },
        tensor_path,
    )
    rec = {
        "dataset_index": 0,
        "_tensor_path": str(tensor_path),
        "_sidecar_coord_source": "aligned_native",
        "sidecar_schema": NATIVE_TEACHER_SIDECAR_SCHEMA,
        "alignment_version": NATIVE_TEACHER_ALIGNMENT_VERSION,
        "alignment_fingerprint": "jsonl-fingerprint",
        "alignment_contract": contract,
    }
    with pytest.raises(RuntimeError, match="alignment_fingerprint mismatch"):
        _load_validated_tensor_sidecar_payload(rec)


class _FakeDataset:
    def __init__(self):
        self.sample_index = [(0, 5)]
        self._sample_subsequence_range = {0: (0, 20)}
        self.samples = {
            (0, 5): {
                "pano_sample_kind": "pixel",
                "pano_view_id": "front",
                "pano_pixel_goal": [128, 128],
            },
            (1, 7): {
                "pano_sample_kind": "turn",
                "pano_view_id": "view_turn",
            },
        }

    def __len__(self):
        return len(self.sample_index)

    def _load_meta(self, clip_idx):
        return {"num_frames": 20 + int(clip_idx)}

    def _build_sample(self, idx):
        return self.samples[tuple(self.sample_index[idx])]


def test_sample_from_record_uses_clip_frame_when_dataset_index_is_stale():
    dataset = _FakeDataset()
    rec = {"dataset_index": 999, "clip_idx": 0, "current_t": 5}

    sample = _sample_from_record(dataset, rec)

    assert sample["pano_view_id"] == "front"
    assert dataset.sample_index == [(0, 5)]


def test_sample_from_record_prefers_clip_dir_when_clip_index_shifted(tmp_path):
    dataset = _FakeDataset()
    stale_clip = tmp_path / "new_scene" / "clip_000001"
    target_clip = tmp_path / "old_scene" / "clip_000001"
    stale_clip.mkdir(parents=True)
    target_clip.mkdir(parents=True)
    dataset.clips = [stale_clip, target_clip]
    dataset._clip_dir_to_idx = {str(stale_clip): 0, str(target_clip): 1}
    dataset.samples[(1, 5)] = {
        "pano_sample_kind": "pixel",
        "pano_view_id": "right",
        "pano_pixel_goal": [64, 128],
    }
    rec = {
        "dataset_index": 0,
        "clip_idx": 0,
        "clip_dir": str(target_clip),
        "current_t": 5,
    }

    sample = _sample_from_record(dataset, rec)

    assert sample["pano_view_id"] == "right"
    assert dataset.sample_index == [(0, 5)]


def test_filter_records_with_pano_goals_is_global_before_ddp_sharding():
    dataset = _FakeDataset()
    records = [
        {"dataset_index": 0, "clip_idx": 0, "current_t": 5},
        {"dataset_index": 1, "clip_idx": 1, "current_t": 7},
    ]

    filtered = _filter_records_with_pano_goals(records, dataset=dataset)

    assert filtered == [records[0]]


def test_filter_records_drops_sidecar_without_pano_metadata():
    dataset = _FakeDataset()
    records = [
        {
            "dataset_index": 0,
            "clip_idx": 0,
            "current_t": 5,
            "_tensor_path": "unused.pt",
            "_sidecar_pano_view_id": None,
            "_sidecar_pano_pixel_goal": None,
        },
    ]

    filtered = _filter_records_with_pano_goals(
        records,
        dataset=dataset,
        validate_sidecar_metadata=True,
    )

    assert filtered == []


def test_filter_records_drops_tensor_sidecar_with_wrong_dataset_index(tmp_path):
    tensor_path = tmp_path / "teacher.pt"
    torch.save({"dataset_index": 999, "traj_latents_768": torch.ones(1, 4, 6)}, tensor_path)
    dataset = _FakeDataset()
    records = [
        {
            "dataset_index": 0,
            "clip_idx": 0,
            "current_t": 5,
            "_tensor_path": str(tensor_path),
            "_sidecar_pano_view_id": "front",
            "_sidecar_pano_pixel_goal": [128, 128],
        },
    ]

    filtered = _filter_records_with_pano_goals(
        records,
        dataset=dataset,
        validate_sidecar_metadata=True,
    )

    assert filtered == []


def test_load_teacher_latents_rejects_tensor_sidecar_with_wrong_dataset_index(tmp_path):
    tensor_path = tmp_path / "teacher.pt"
    torch.save({"dataset_index": 999, "traj_latents_768": torch.ones(1, 4, 6)}, tensor_path)

    with pytest.raises(RuntimeError, match="dataset_index mismatch"):
        _load_teacher_latents(
            [{"dataset_index": 0, "_tensor_path": str(tensor_path)}],
            torch.device("cpu"),
            target_dim=6,
        )


def test_tensor_sidecar_accepts_stable_key_when_dataset_index_shifted(tmp_path):
    clip = tmp_path / "scene" / "clip_000001"
    clip.mkdir(parents=True)
    tensor_path = tmp_path / "teacher.pt"
    torch.save(
        {
            "dataset_index": 999,
            "clip_dir": str(clip),
            "current_t": 5,
            "traj_latents_768": torch.ones(1, 4, 6),
        },
        tensor_path,
    )

    payload = _load_validated_tensor_sidecar_payload(
        {
            "dataset_index": 0,
            "clip_dir": str(clip),
            "current_t": 5,
            "_tensor_path": str(tensor_path),
        }
    )

    assert int(payload["dataset_index"]) == 999


def test_filter_records_accepts_strictly_aligned_sidecar(tmp_path):
    tensor_path = tmp_path / "teacher.pt"
    torch.save({"dataset_index": 0, "traj_latents_768": torch.ones(1, 4, 6)}, tensor_path)
    dataset = _FakeDataset()
    records = [
        {
            "dataset_index": 0,
            "clip_idx": 0,
            "current_t": 5,
            "_tensor_path": str(tensor_path),
            "_sidecar_pano_view_id": "front",
            "_sidecar_pano_pixel_goal": [128, 128],
        },
    ]

    filtered = _filter_records_with_pano_goals(
        records,
        dataset=dataset,
        validate_sidecar_metadata=True,
    )

    assert filtered == records


def _trusted_native_record(tmp_path, dataset):
    clip = tmp_path / "scene" / "clip_000001"
    clip.mkdir(parents=True, exist_ok=True)
    dataset.clips = [clip]
    dataset._clip_dir_to_idx = {str(clip): 0}
    dataset._clip_valid_frames = {0: [5]}
    sample = {
        "pano_sample_kind": "pixel",
        "pano_view_id": "front",
        "pano_pixel_goal": [123, 111],
        "pano_pixel_goal_relative_len": 7,
        "pano_goal_frame_idx": 12,
        "aligned_native_pixel_goal_uv": [151, 202],
        "aligned_native_goal_frame_idx": 12,
        "aligned_native_visible": True,
    }
    stable_key = f"{clip}|t=5"
    tensor_path = tmp_path / "native_teacher.pt"
    contract = aligned_native_sidecar_contract(
        sample,
        stable_sample_key=stable_key,
        current_t=5,
    )
    torch.save(
        {
            "dataset_index": 999,
            "stable_sample_key": stable_key,
            **contract,
            "traj_latents": torch.ones(1, 4, 6),
            "traj_latents_768": torch.ones(1, 4, 6),
        },
        tensor_path,
    )
    return {
        "status": "ok",
        "dataset_index": 999,
        "clip_idx": 999,
        "clip_dir": str(clip),
        "current_t": 5,
        "stable_sample_key": stable_key,
        "_tensor_path": str(tensor_path),
        "_sidecar_coord_source": "aligned_native",
        "_sidecar_mode": "aligned_native_coord",
        "_sidecar_pano_view_id": "front",
        "_sidecar_pano_pixel_goal": [123, 111],
        **contract,
        "teacher": {
            "coord_source": "aligned_native",
            "mode": "aligned_native_coord",
            "coord_uv": [151, 202],
            "internnav_pixel_goal_yx": [202, 151],
            "conditioned_coord_text": "202 151",
            "pano_view_id": "front",
            "goal_frame_idx": 12,
        },
        "dataset_label": sidecar_alignment_metadata(sample),
    }


def test_trusted_native_fast_filter_binds_live_dataset_without_loading_sample(tmp_path):
    dataset = _FakeDataset()
    record = _trusted_native_record(tmp_path, dataset)

    def must_not_materialize(_idx):
        raise AssertionError("trusted native startup must not load RGB sample")

    dataset._build_sample = must_not_materialize
    filtered = _filter_records_with_pano_goals(
        [record],
        dataset=dataset,
        validate_sidecar_metadata=True,
        require_native_teacher_sidecar=True,
        trust_sidecar_pano_labels=True,
    )
    assert filtered == [record]


def test_trusted_native_fast_filter_rejects_missing_clip_path(tmp_path):
    dataset = _FakeDataset()
    record = _trusted_native_record(tmp_path, dataset)
    record["clip_dir"] = str(tmp_path / "other" / "missing_clip")

    filtered = _filter_records_with_pano_goals(
        [record],
        dataset=dataset,
        require_native_teacher_sidecar=True,
        trust_sidecar_pano_labels=True,
    )
    assert filtered == []


def test_trusted_native_fast_filter_rejects_teacher_coordinate_mismatch(tmp_path):
    dataset = _FakeDataset()
    record = _trusted_native_record(tmp_path, dataset)
    record["teacher"]["coord_uv"] = [202, 151]

    filtered = _filter_records_with_pano_goals(
        [record],
        dataset=dataset,
        require_native_teacher_sidecar=True,
        trust_sidecar_pano_labels=True,
    )
    assert filtered == []


def test_train_val_split_never_leaks_frames_from_the_same_trajectory():
    records = [
        {
            "dataset_index": idx,
            "scene_id": "scene",
            "trajectory_id": f"traj-{idx // 3}",
            "clip_dir": f"/data/clip-{idx // 3}",
        }
        for idx in range(12)
    ]
    train, val = _split_train_val(
        records,
        SimpleNamespace(val_records=0, val_ratio=0.25, seed=42),
    )
    train_groups = {(r["scene_id"], r["trajectory_id"]) for r in train}
    val_groups = {(r["scene_id"], r["trajectory_id"]) for r in val}
    assert train_groups.isdisjoint(val_groups)
    assert len(val) == 3


class _FakeSystem1Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = torch.nn.Linear(3, 4)
        self.cond_projector = torch.nn.Linear(6, 6, bias=False)
        torch.nn.init.eye_(self.cond_projector.weight)

    @staticmethod
    def _expand_sequence_training_inputs(traj_cond, gt_trajectory, traj_images, trajectory_valid):
        if traj_images is None or traj_images.ndim != 5 or gt_trajectory.ndim != 4:
            return traj_cond, gt_trajectory, traj_images, trajectory_valid
        batch_size, num_frames = traj_images.shape[:2]
        anchor_images = traj_images[:, 0:1].repeat(1, num_frames, 1, 1, 1).flatten(0, 1)
        current_images = traj_images.flatten(0, 1)
        traj_image_pairs = torch.stack([anchor_images, current_images], dim=1)
        traj_cond = traj_cond.unsqueeze(1).repeat(1, num_frames, 1, 1).flatten(0, 1)
        gt_trajectory = gt_trajectory.flatten(0, 1)
        if trajectory_valid is not None:
            trajectory_valid = trajectory_valid.flatten(0, 1)
        return traj_cond, gt_trajectory, traj_image_pairs, trajectory_valid

    @staticmethod
    def sample_flow_matching_inputs(gt_trajectory):
        noisy = torch.zeros_like(gt_trajectory)
        timesteps = torch.zeros(gt_trajectory.shape[0], dtype=torch.long, device=gt_trajectory.device)
        target = torch.ones_like(gt_trajectory)
        return noisy, timesteps, target

    @staticmethod
    def predict_velocity_from_projected(traj_cond, noisy_trajectory, timesteps, traj_images=None):
        del timesteps, traj_images
        scale = traj_cond.mean(dim=(1, 2)).view(-1, 1, 1)
        return noisy_trajectory + scale

    @staticmethod
    def masked_velocity_mse(pred, target, trajectory_valid=None):
        loss = (pred - target).square().mean(dim=(1, 2))
        if trajectory_valid is None:
            return loss.mean()
        mask = trajectory_valid.float()
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class _FakeModel:
    def __init__(self):
        self.nextdit_action_head = _FakeSystem1Head()


def test_adapter_objective_keeps_gradient_to_adapter_output():
    pred_raw = torch.randn(2, 4, 6, requires_grad=True)
    teacher_cond = torch.randn(2, 4, 6)
    batch = AdapterTrainBatch(
        student_latents=torch.empty(0),
        teacher_latents=None,
        teacher_cond=teacher_cond,
        records=[],
        trajectory=torch.zeros(2, 3, 4, 3),
        trajectory_valid=torch.ones(2, 3),
        traj_images=torch.zeros(2, 3, 2, 2, 3),
    )
    args = SimpleNamespace(
        raw_distill_weight=0.0,
        raw_norm_weight=0.0,
        cond_distill_weight=1.0,
        cond_cosine_weight=1.0,
        cond_smooth_l1_beta=1.0,
        gt_weight=1.0,
    )

    loss, metrics, pred_cond = _compute_adapter_objective(
        model=_FakeModel(),
        pred_raw=pred_raw,
        batch=batch,
        args=args,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert metrics["cond"] > 0.0
    assert metrics["gt"] > 0.0
    assert pred_cond.requires_grad
    assert pred_raw.grad is not None
    assert torch.any(pred_raw.grad != 0)
