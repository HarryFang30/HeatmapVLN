import io

import numpy as np
import pytest
from PIL import Image

from src.data.s2_records import (
    EpisodeRecord,
    EpisodeRecordWriter,
    depth_from_png_bytes,
    depth_to_png_bytes,
    iter_record_paths,
    label_path_for,
)


def _jpeg(value: int, size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _writer(tmp_path):
    return EpisodeRecordWriter(
        root=tmp_path,
        scene_id="sceneA",
        episode_id=7,
        episode={"instruction": "walk to the door", "split": "train"},
    )


def test_round_trip_keeps_exact_bytes_and_drops_unreferenced_fronts(tmp_path):
    writer = _writer(tmp_path)
    for step in range(4):
        writer.add_step(step, [0.0, 0.0, float(step)], [1.0, 0.0, 0.0, 0.0])
        writer.add_front(step, _jpeg(10 * step))
    depth = np.full((48, 64), 2.3456, dtype=np.float32)
    writer.add_call(
        {"call_index": 0, "step_id": 3, "history_step_ids": [0, 2], "llm_output": "12 34", "kind": "trajectory"},
        lookdown_jpeg=_jpeg(200, (640, 480)),
        lookdown_depth_m=depth,
    )
    path = writer.commit()

    record = EpisodeRecord(path, verify=True)
    assert [p.name for p in iter_record_paths(tmp_path)] == [path.name]
    assert record.scene_id == "sceneA" and record.episode_id == 7
    assert record.front_jpeg(2) == _jpeg(20)
    assert record.lookdown_jpeg(0) == _jpeg(200, (640, 480))
    with pytest.raises(KeyError):
        record.front_jpeg(1)  # never part of a prompt
    np.testing.assert_allclose(record.lookdown_depth(0), 2.346, atol=1e-6)
    assert record.calls[0]["history_step_ids"] == [0, 2]
    assert record.step(3)["position"] == [0.0, 0.0, 3.0]
    assert label_path_for(path, tmp_path / "labels").name == "sceneA_000007.s2labels.jsonl"


def test_resent_front_with_different_bytes_is_refused(tmp_path):
    writer = _writer(tmp_path)
    writer.add_front(0, _jpeg(1))
    writer.add_front(0, _jpeg(1))
    with pytest.raises(ValueError, match="resent with different bytes"):
        writer.add_front(0, _jpeg(2))


def test_call_must_reference_stored_fronts(tmp_path):
    writer = _writer(tmp_path)
    writer.add_front(0, _jpeg(1))
    with pytest.raises(ValueError, match="never stored"):
        writer.add_call(
            {"call_index": 0, "step_id": 5, "history_step_ids": [0], "llm_output": "STOP", "kind": "stop"},
            lookdown_jpeg=_jpeg(3),
            lookdown_depth_m=None,
        )


def test_uncommitted_record_is_invisible(tmp_path):
    writer = _writer(tmp_path)
    writer.add_front(0, _jpeg(1))
    path = writer.commit()
    path.with_name(path.name + ".sha256").unlink()
    assert list(iter_record_paths(tmp_path)) == []
    with pytest.raises(FileNotFoundError):
        EpisodeRecord(path)


def test_depth_png_is_lossless_to_the_millimetre():
    depth = np.array([[0.0, 0.0004, 1.2345], [9.9999, 3.0, 0.5]], dtype=np.float64)
    back = depth_from_png_bytes(depth_to_png_bytes(depth))
    np.testing.assert_allclose(back, np.rint(depth * 1000) / 1000, atol=1e-6)
