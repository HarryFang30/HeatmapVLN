from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.vo.amb3r_pose import history_rel_poses_from_amb3r
from src.vo.clip_io import center_crop_resize_for_amb3r
from src.vo.online_amb3r import OnlineAMB3RSession


_CV_BASIS = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _opencv_trajectory(frame_count: int) -> np.ndarray:
    poses = []
    for index in range(frame_count):
        habitat = np.eye(4, dtype=np.float32)
        habitat[2, 3] = -float(index)
        poses.append(habitat @ _CV_BASIS)
    return np.stack(poses)


class MockIncrementalBackend:
    def __init__(self) -> None:
        self.reset_calls: list[int] = []
        self.direct_calls: list[int] = []
        self.initialize_calls: list[int] = []
        self.increment_calls: list[tuple[int, int, int]] = []
        self._poses: np.ndarray | None = None

    def reset(self, *, max_frames: int) -> None:
        self.reset_calls.append(max_frames)
        self._poses = None

    def predict_direct(self, frames_rgb: np.ndarray) -> np.ndarray:
        self.direct_calls.append(len(frames_rgb))
        return _opencv_trajectory(len(frames_rgb))

    def initialize(self, frames_rgb: np.ndarray) -> np.ndarray:
        self.initialize_calls.append(len(frames_rgb))
        self._poses = _opencv_trajectory(len(frames_rgb))
        return self._poses.copy()

    def map_increment(
        self,
        frames_rgb: np.ndarray,
        *,
        start_index: int,
        end_index: int,
    ) -> np.ndarray:
        self.increment_calls.append((start_index, end_index, len(frames_rgb)))
        self._poses = _opencv_trajectory(end_index + 1)
        return self._poses.copy()

    def poses(self, *, frame_count: int) -> np.ndarray:
        if self._poses is None:
            raise RuntimeError("mock map is not initialized")
        return self._poses[:frame_count].copy()


def _identity_frame_processor(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame)


def _frame(index: int) -> np.ndarray:
    return np.full((3, 5, 3), index, dtype=np.uint8)


def _session(
    backend: MockIncrementalBackend,
    *,
    map_init_window: int = 4,
    map_every: int = 2,
) -> OnlineAMB3RSession:
    return OnlineAMB3RSession(
        backend,
        map_init_window=map_init_window,
        map_every=map_every,
        max_history=8,
        resolution=(5, 3),
        frame_processor=_identity_frame_processor,
    )


def _ingest_range(
    session: OnlineAMB3RSession,
    session_id: str,
    start: int,
    stop: int,
) -> None:
    for frame_id in range(start, stop):
        session.ingest(
            session_id,
            frame_id=frame_id,
            frame_rgb=_frame(frame_id),
            capture_step=frame_id,
        )


def test_reset_and_ingest_require_active_session_and_contiguous_ids() -> None:
    backend = MockIncrementalBackend()
    session = _session(backend)

    with pytest.raises(RuntimeError, match="not been reset"):
        session.ingest("episode-a", frame_id=0, frame_rgb=_frame(0), capture_step=0)

    reset = session.reset("episode-a", max_frames=10)
    assert reset["session_id"] == "episode-a"
    assert backend.reset_calls == [10]
    session.ingest("episode-a", frame_id=0, frame_rgb=_frame(0), capture_step=2)

    with pytest.raises(ValueError, match="strictly contiguous"):
        session.ingest("episode-a", frame_id=2, frame_rgb=_frame(2), capture_step=2)
    with pytest.raises(ValueError, match="Stale"):
        session.ingest("episode-b", frame_id=1, frame_rgb=_frame(1), capture_step=1)
    with pytest.raises(ValueError, match="monotonic"):
        session.ingest("episode-a", frame_id=1, frame_rgb=_frame(1), capture_step=1)

    session.reset("episode-b", max_frames=7)
    assert session.frame_count == 0
    assert backend.reset_calls == [10, 7]
    session.ingest("episode-b", frame_id=0, frame_rgb=_frame(0), capture_step=3)


def test_map_warmup_is_unavailable_and_never_runs_direct_prediction() -> None:
    backend = MockIncrementalBackend()
    session = _session(backend)
    session.reset("episode", max_frames=20)
    _ingest_range(session, "episode", 0, 2)

    first = session.query(
        "episode",
        current_frame_id=1,
        history_frame_ids=[0],
    )
    second = session.query(
        "episode",
        current_frame_id=1,
        history_frame_ids=[0, 1, 1],
    )
    assert first.ready is False
    assert first.provider_phase == "map_warmup"
    assert first.last_mapped_frame_id is None
    assert first.history_rel_poses.shape == (0, 4)
    assert second.ready is False
    assert second.provider_phase == "map_warmup"
    assert second.history_rel_poses.shape == (0, 4)
    assert backend.direct_calls == []

    session.ingest("episode", frame_id=2, frame_rgb=_frame(2), capture_step=2)
    third = session.query(
        "episode",
        current_frame_id=2,
        history_frame_ids=[0, 1],
    )
    assert third.ready is False
    assert third.provider_phase == "map_warmup"
    assert third.history_rel_poses.shape == (0, 4)
    assert backend.direct_calls == []


def test_twentieth_frame_equivalent_initializes_once_then_maps_only_new_tails() -> None:
    backend = MockIncrementalBackend()
    session = _session(backend, map_init_window=4, map_every=2)
    session.reset("episode", max_frames=20)
    _ingest_range(session, "episode", 0, 4)

    assert backend.initialize_calls == [4]
    initialized = session.query(
        "episode",
        current_frame_id=3,
        history_frame_ids=[0, 2],
    )
    assert initialized.provider_phase == "stateful_backend"
    assert initialized.last_mapped_frame_id == 3
    assert backend.direct_calls == []
    assert backend.increment_calls == []

    # A planning query before the regular two-frame cadence flushes only the
    # one pending frame, not the mapped prefix.
    session.ingest("episode", frame_id=4, frame_rgb=_frame(4), capture_step=4)
    queried_tail = session.query(
        "episode",
        current_frame_id=4,
        history_frame_ids=[1, 3],
    )
    assert queried_tail.last_mapped_frame_id == 4
    assert backend.increment_calls == [(4, 4, 5)]

    # After that early flush, the next two never-before-mapped frames are
    # committed together automatically on ingest of frame 6.
    session.ingest("episode", frame_id=5, frame_rgb=_frame(5), capture_step=5)
    session.ingest("episode", frame_id=6, frame_rgb=_frame(6), capture_step=6)
    assert backend.increment_calls == [(4, 4, 5), (5, 6, 7)]
    final = session.query(
        "episode",
        current_frame_id=6,
        history_frame_ids=[0, 4, 5],
    )
    assert final.last_mapped_frame_id == 6
    assert backend.increment_calls == [(4, 4, 5), (5, 6, 7)]


def test_query_rejects_noncausal_or_unknown_frame_ids() -> None:
    backend = MockIncrementalBackend()
    session = _session(backend)
    session.reset("episode", max_frames=10)
    _ingest_range(session, "episode", 0, 3)

    with pytest.raises(ValueError, match="latest"):
        session.query("episode", current_frame_id=1, history_frame_ids=[0])
    with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
        session.query("episode", current_frame_id=2, history_frame_ids=[3])
    with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
        session.query("episode", current_frame_id=2, history_frame_ids=[-1])
    with pytest.raises(ValueError, match="non-decreasing"):
        session.query("episode", current_frame_id=2, history_frame_ids=[2, 1])
    with pytest.raises(ValueError, match="Stale"):
        session.query("other", current_frame_id=2, history_frame_ids=[0])


def test_zero_history_is_explicitly_not_ready_and_does_not_run_model() -> None:
    backend = MockIncrementalBackend()
    session = _session(backend)
    session.reset("episode", max_frames=10)
    session.ingest("episode", frame_id=0, frame_rgb=_frame(0), capture_step=0)

    result = session.query(
        "episode",
        current_frame_id=0,
        history_frame_ids=[],
    )
    assert result.ready is False
    assert result.provider_phase == "insufficient_history"
    assert result.history_rel_poses.shape == (0, 4)
    assert backend.direct_calls == []
    assert result.to_payload()["pose_provider"] == "amb3r_vo_da3"
    assert "checkpoint_sha256" not in result.to_payload()


def test_fixed_translation_scale_is_applied_only_at_query_boundary() -> None:
    backend = MockIncrementalBackend()
    session = _session(backend)
    session.reset("episode", max_frames=10)
    _ingest_range(session, "episode", 0, 4)

    native = session.query(
        "episode",
        current_frame_id=3,
        history_frame_ids=[0, 2],
    )
    scaled = session.query(
        "episode",
        current_frame_id=3,
        history_frame_ids=[0, 2],
        translation_scale=2.5,
    )
    np.testing.assert_allclose(
        scaled.history_rel_poses[:, :2],
        native.history_rel_poses[:, :2] * 2.5,
    )
    np.testing.assert_allclose(
        scaled.history_rel_poses[:, 2:],
        native.history_rel_poses[:, 2:],
    )
    assert backend.direct_calls == []


def test_habitat_frame_preprocessing_matches_released_amb3r_demo() -> None:
    rows = np.arange(480, dtype=np.uint16)[:, None]
    columns = np.arange(640, dtype=np.uint16)[None, :]
    frame = np.stack(
        (
            np.broadcast_to(columns % 256, (480, 640)),
            np.broadcast_to(rows % 256, (480, 640)),
            (rows + columns) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    # Released slam/datasets/demo.py crops width to
    # round(480 * 518/392) == 634 at x=3, then resizes to 518x392.
    expected = np.asarray(
        Image.fromarray(frame, mode="RGB")
        .crop((3, 0, 637, 480))
        .resize((518, 392), resampling)
    )
    actual = center_crop_resize_for_amb3r(frame[None], resolution=(518, 392))[0]
    np.testing.assert_array_equal(actual, expected)


def test_amb3r_preprocessing_rejects_non_uint8_or_unbatched_input() -> None:
    with pytest.raises(ValueError, match=r"uint8 \[T,H,W,3\]"):
        center_crop_resize_for_amb3r(
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
    with pytest.raises(ValueError, match=r"uint8 \[T,H,W,3\]"):
        center_crop_resize_for_amb3r(
            np.zeros((1, 480, 640, 3), dtype=np.float32),
        )
