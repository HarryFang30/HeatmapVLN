import math

import pytest
import torch

from src.models.adapters import GeometryAwarePanoToNextDiTAdapter, view_ids_to_indices
from scripts.training.train_pano_latent_adapter import (
    AdapterTrainBatch,
    _filter_records_with_pano_goals,
    _load_teacher_latents,
    _policy_and_gt_losses,
    _sample_from_record,
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
    assert torch.allclose(scalars[1, 2:], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(scalars[2, 2:], torch.tensor([0.0, -1.0]), atol=1e-6)
    assert torch.allclose(scalars[3, 2:], torch.tensor([-1.0, 0.0]), atol=1e-6)


def test_geometry_scalars_apply_horizontal_fov():
    view_indices = view_ids_to_indices(["front"])
    pixel_xy = torch.tensor([[256.0, 128.0]])
    image_hw = torch.tensor([[256.0, 256.0]])

    scalars = GeometryAwarePanoToNextDiTAdapter.geometry_scalars(
        view_indices,
        pixel_xy,
        image_hw,
        horizontal_fov_deg=90.0,
    )

    expected_theta = math.radians(45.0)
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


class _FakeSystem1Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = torch.nn.Linear(3, 4)

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


def test_policy_and_gt_losses_keep_gradient_to_adapter_condition():
    pred_cond = torch.randn(2, 4, 6, requires_grad=True)
    teacher_cond = torch.randn(2, 4, 6)
    batch = AdapterTrainBatch(
        student_latents=torch.empty(0),
        teacher_latents=teacher_cond,
        view_indices=torch.empty(0, dtype=torch.long),
        goal_pixels=torch.empty(0),
        image_hw=torch.empty(0),
        trajectory=torch.zeros(2, 3, 4, 3),
        trajectory_valid=torch.ones(2, 3),
        traj_images=torch.zeros(2, 3, 2, 2, 3),
        records=[],
    )

    loss, metrics = _policy_and_gt_losses(
        model=_FakeModel(),
        pred_cond=pred_cond,
        teacher_cond=teacher_cond,
        batch=batch,
        policy_weight=1.0,
        gt_weight=1.0,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert metrics["policy_loss"] > 0.0
    assert metrics["gt_loss"] > 0.0
    assert pred_cond.grad is not None
    assert torch.any(pred_cond.grad != 0)
