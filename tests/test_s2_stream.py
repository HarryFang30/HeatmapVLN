import json
import threading

import numpy as np
import pytest

from src.data import s2_stream
from src.data.s2_records import EpisodeRecord
from src.data.s2_stream import (
    PackedSample,
    StreamShardDataset,
    done_marker,
    draw_order,
    read_shard,
    shard_name,
    write_shard,
)
from tests.test_s2_sft_dataset import _calls, _jpeg, _write_labels, _write_record


def _sample(i, n_history=2):
    return PackedSample(
        key=f"k{i}", source="dagger", instruction="go", turns=("↓", f"{i} 7"), vlm_image_size=(384, 384),
        history_jpegs=tuple(bytes([i, k]) for k in range(n_history)), current_jpeg=bytes([i, 99]),
        lookdown_jpeg=bytes([i, 200]),
    )


def test_shard_round_trip_keeps_bytes_turns_and_order(tmp_path):
    samples = [_sample(i, n_history=i % 3) for i in range(5)]
    write_shard(tmp_path / shard_name(0), samples)
    assert read_shard(tmp_path / shard_name(0)) == samples


def test_draw_order_is_seeded_and_weighted():
    a = draw_order([1.0, 3.0], 20000, seed=7)
    assert np.array_equal(a, draw_order([1.0, 3.0], 20000, seed=7))
    assert not np.array_equal(a, draw_order([1.0, 3.0], 20000, seed=8))
    assert abs((a == 1).mean() - 0.75) < 0.02
    with pytest.raises(ValueError):
        draw_order([0.0, 0.0], 3, seed=0)


@pytest.fixture
def no_tokenizer(monkeypatch):
    """Stand-ins for the decode + tokenise step, which the dataset tests cover."""
    monkeypatch.setattr(s2_stream, "packed_inputs", lambda sample: (sample.key, None, None))
    import src.data.s2_sft_dataset as sft

    monkeypatch.setattr(sft, "encode_conversation", lambda processor, messages, images, lookdown, turns: {"m": messages})


def test_ranks_read_disjoint_slices_and_the_last_reader_deletes(tmp_path, no_tokenizer):
    for s in range(3):
        write_shard(tmp_path / shard_name(s), [_sample(4 * s + i) for i in range(4)])
    common = dict(world_size=2, start_shard=0, num_shards=3, shard_size=4, poll_s=0.01, wait_timeout_s=1)
    rank0 = [x["key"] for x in StreamShardDataset(tmp_path, None, rank=0, **common)]
    assert rank0 == ["k0", "k2", "k4", "k6", "k8", "k10"]
    assert all((tmp_path / shard_name(s)).exists() for s in range(3))  # rank 1 has not read yet
    assert all(done_marker(tmp_path, s, 0).exists() for s in range(3))
    rank1 = [x["key"] for x in StreamShardDataset(tmp_path, None, rank=1, **common)]
    assert rank1 == ["k1", "k3", "k5", "k7", "k9", "k11"]
    assert list(tmp_path.iterdir()) == []


def test_a_resumed_reader_starts_at_its_shard_and_waits_for_late_ones(tmp_path, no_tokenizer):
    write_shard(tmp_path / shard_name(2), [_sample(i) for i in range(2)])

    def late():
        write_shard(tmp_path / shard_name(3), [_sample(10 + i) for i in range(2)])

    timer = threading.Timer(0.2, late)
    timer.start()
    keys = [x["key"] for x in StreamShardDataset(
        tmp_path, None, rank=0, world_size=1, start_shard=2, num_shards=4, shard_size=2, poll_s=0.02, wait_timeout_s=5)]
    timer.join()
    assert keys == ["k0", "k1", "k10", "k11"]
    with pytest.raises(TimeoutError):
        list(StreamShardDataset(tmp_path, None, rank=0, world_size=1, start_shard=5, num_shards=6, shard_size=2,
                                poll_s=0.01, wait_timeout_s=0.05))


def test_a_short_shard_is_refused(tmp_path, no_tokenizer):
    write_shard(tmp_path / shard_name(0), [_sample(0)])
    with pytest.raises(ValueError, match="expected 2"):
        list(StreamShardDataset(tmp_path, None, rank=0, world_size=1, start_shard=0, num_shards=1, shard_size=2))


def test_builder_packs_the_trainers_selection_with_the_records_bytes(tmp_path):
    pytest.importorskip("vla_rpc.core.image")
    from scripts.tools.build_s2_train_stream import main as build

    records, labels, out = tmp_path / "rec", tmp_path / "lab", tmp_path / "stream"
    for episode in range(3):
        path = _write_record(records, "1LXtFkjw3qL", episode, _calls(4))  # not a held-out scene
        _write_labels(path, labels, [
            {"call_index": 0, "agree": True, "oracle_kind": "pixel", "oracle_turn1": "↓", "oracle_turn2": "240 300",
             "distance_to_goal": 5.0},
            {"call_index": 1, "agree": False, "oracle_kind": "turn", "oracle_turn1": "→", "oracle_turn2": None,
             "distance_to_goal": 5.0},
            {"call_index": 2, "agree": False, "oracle_kind": "stop", "oracle_turn1": "STOP", "oracle_turn2": None,
             "distance_to_goal": 1.0},
            {"call_index": 3, "agree": None, "oracle_kind": "skip", "oracle_turn1": None, "oracle_turn2": None,
             "distance_to_goal": 5.0},
        ])
    assert build(["--record-root", str(records), "--label-root", str(labels), "--out-dir", str(out),
                  "--shard-size", "8", "--global-batch", "8", "--world-size", "2", "--epochs", "2",
                  "--workers", "1"]) == 0
    manifest = json.loads((out / "stream_manifest.json").read_text())
    assert manifest["total_samples"] == 24 and manifest["num_shards"] == 3  # ceil(2 * 9 / 8) * 8
    assert manifest["selection"]["dagger_selected"] == 9 and manifest["drawn"]["dagger"] == 24

    by_key = {}
    for path in sorted(records.glob("*/*.s2rec.tar")):
        record = EpisodeRecord(path)
        for call in record.calls:
            by_key[f"{record.scene_id}_{record.episode_id:06d}#{call['call_index']}"] = (record, call)
    expected_turns = {0: ("↓", "245 320"), 1: ("→",), 2: ("STOP",)}
    for shard in manifest["shards"]:
        for sample in read_shard(out / shard["name"]):
            record, call = by_key[sample.key]
            assert sample.turns == expected_turns[call["call_index"]]
            assert sample.current_jpeg == record.front_jpeg(call["step_id"])
            assert list(sample.history_jpegs) == [record.front_jpeg(s) for s in call["history_step_ids"]]
            assert sample.lookdown_jpeg == record.lookdown_jpeg(call["call_index"])
            assert sample.instruction == record.episode["instruction"] and sample.vlm_image_size == (384, 384)


def test_transfer_pushes_verifies_backs_off_and_skips(tmp_path):
    import socket
    import time

    from scripts.tools import stream_transfer as st

    dest, src = tmp_path / "dest", tmp_path / "src"
    dest.mkdir()
    src.mkdir()
    (src / "a.tar").write_bytes(b"x" * 1000)
    (src / "b.tar").write_bytes(b"y" * 1000)
    server = st._Server(("127.0.0.1", 0), st._Receiver)
    server.state = st._State(dest, cap_bytes=1500)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert st._push_one(port, src, "a.tar", 10) == "DONE"
        assert (dest / "a.tar").read_bytes() == b"x" * 1000
        assert st._push_one(port, src, "a.tar", 10) == "EXISTS"
        assert st._push_one(port, src, "b.tar", 10) == "BUSY"  # 1000 + 1000 > 1500
        (dest / "a.tar").unlink()  # the trainer consumed it
        assert st._push_one(port, src, "b.tar", 10) == "DONE"
        # a corrupted body is refused and leaves nothing behind
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(json.dumps({"name": "c.tar", "size": 3, "sha256": "0" * 64}).encode() + b"\n")
            assert sock.makefile("rb").readline().strip() == b"GO"
            sock.sendall(b"abc")
            assert sock.makefile("rb").readline().strip() == b"BAD"
        time.sleep(0.05)
        assert sorted(p.name for p in dest.iterdir()) == ["b.tar"]
        for name in ("../escape", "a/../../b", "a/b/c/d", "/abs"):  # at most two subdir levels, never upward
            with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
                sock.sendall(json.dumps({"name": name, "size": 1, "sha256": "0" * 64}).encode() + b"\n")
                assert sock.makefile("rb").readline().strip() == b"BAD"
    finally:
        server.shutdown()


def test_push_into_a_foreign_listener_waits_and_never_counts_as_pushed(tmp_path, capsys):
    import socketserver

    from scripts.tools import stream_transfer as st

    class Foreign(socketserver.StreamRequestHandler):  # what round 1 met: a model server on the port
        def handle(self):
            self.rfile.readline()
            self.wfile.write(b"\x80\x04\x95\xfe\xff not a receiver\n")

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.tar").write_bytes(b"x" * 10)
    (tmp_path / "list").write_text("a.tar\n")
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Foreign)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert st._push_one(port, src, "a.tar", 10).startswith("FOREIGN")
        code = st.main(["push", "--src", str(src), "--ports", str(port), "--files", str(tmp_path / "list"),
                        "--retries", "2", "--foreign-sleep", "0"])
    finally:
        server.shutdown()
    out = capsys.readouterr().out
    assert code == 1
    assert "not a receiver yet" in out and "FAIL a.tar: FOREIGN" in out and "push finished: 0/1 ok" in out
