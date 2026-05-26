import math

import torch

from src.models.adapters import GeometryAwarePanoToNextDiTAdapter, view_ids_to_indices
from scripts.training.train_pano_latent_adapter import (
    _filter_records_with_pano_goals,
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
