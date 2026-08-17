from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

from src.vo.online_amb3r import StatefulAMB3RBackend


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)

    def __getitem__(self, item):
        return _FakeTensor(self.value[item])

    def __setitem__(self, item, value) -> None:
        self.value[item] = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.value.copy()


class _FakeBatch:
    def __init__(self, frames: np.ndarray) -> None:
        self.frames = np.asarray(frames)

    def to(self, device: str):
        return self


class _FakeModel:
    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        return self


class _FakeMemory:
    def __init__(self, cfg, num_frames: int, height: int, width: int) -> None:
        self.cfg = cfg
        self.num_frames = num_frames
        self.height = height
        self.width = width
        poses = np.broadcast_to(
            np.eye(4, dtype=np.float32),
            (num_frames, 4, 4),
        ).copy()
        self.poses = _FakeTensor(poses)
        self.cur_kf_idx = _FakeTensor(np.asarray([0, 5, 19], dtype=np.int64))


class _FakePipeline:
    last_instance = None

    def __init__(self, model, cfg_path: str) -> None:
        self.model = model
        self.cfg = SimpleNamespace(device="cuda:0")
        self.keyframe_memory = None
        self.initialize_batches: list[np.ndarray] = []
        self.mapping_calls: list[dict] = []
        type(self).last_instance = self

    def initialize_map(self, views, cfg) -> None:
        self.initialize_batches.append(views["images"].frames.copy())

    def mapping(self, views, cfg) -> None:
        self.mapping_calls.append(
            {
                "frames": views["images"].frames.copy(),
                "start_idx": views["start_idx"],
                "end_idx": views["end_idx"],
            }
        )


def _install_fake_slam(monkeypatch) -> None:
    slam = types.ModuleType("slam")
    pipeline = types.ModuleType("slam.pipeline")
    memory = types.ModuleType("slam.memory")
    pipeline.AMB3R_VO = _FakePipeline
    memory.SLAMemory = _FakeMemory
    slam.pipeline = pipeline
    slam.memory = memory
    monkeypatch.setitem(sys.modules, "slam", slam)
    monkeypatch.setitem(sys.modules, "slam.pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "slam.memory", memory)


def test_frame20_initializes_and_frame21_maps_only_active_kfs_plus_new_tail(
    monkeypatch,
) -> None:
    """Lock the exact arguments consumed by upstream pipeline/memory.

    ``SLAMemory.update`` builds its global ``map_idx`` as
    ``cur_kf_idx + arange(start_idx, end_idx + 1)``.  Therefore frame 21 must
    present predictions in precisely active-keyframe order followed by global
    frame 20, while ``start_idx == end_idx == 20``.
    """

    _install_fake_slam(monkeypatch)
    backend = StatefulAMB3RBackend(
        _FakeModel(),
        cfg_path="unused-test-config.yaml",
        device="cuda:0",
        map_init_window=20,
        map_every=8,
    )
    monkeypatch.setattr(
        backend,
        "_as_model_batch",
        lambda frames: _FakeBatch(np.asarray(frames)),
    )
    backend.reset(max_frames=21)

    frames = np.stack(
        [np.full((3, 5, 3), index, dtype=np.uint8) for index in range(21)],
        axis=0,
    )
    initialized = backend.initialize(frames[:20])
    assert initialized.shape == (20, 4, 4)

    pipeline = _FakePipeline.last_instance
    assert pipeline is not None
    assert pipeline.initialize_batches[0].shape == (20, 3, 5, 3)

    mapped = backend.map_increment(
        list(frames),
        start_index=20,
        end_index=20,
    )
    assert mapped.shape == (21, 4, 4)
    assert len(pipeline.mapping_calls) == 1
    call = pipeline.mapping_calls[0]
    assert call["start_idx"] == 20
    assert call["end_idx"] == 20
    assert call["frames"].shape == (4, 3, 5, 3)
    assert call["frames"][:, 0, 0, 0].tolist() == [0, 5, 19, 20]
