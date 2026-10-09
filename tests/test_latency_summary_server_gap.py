"""The latency summary must not pass off a client-only run as a complete one.

HEATMAPVLN_TIMING takes effect per process: a client timed against servers that
were not still writes well-formed records, and the summary built from them
looks complete with every server-side stage silently absent.  These records are
shaped like PlanCallTimingLog's (tests/test_latency_timing.py builds the same
ones): an untimed server answers without timing_ms, so the client stores
``model_server_ms: None`` and a ``vo_rpc`` entry with ``server_ms: None``.

Torch-free (the summariser is stdlib only); tests/conftest.py imports torch, so
on a machine without it run this file with --noconftest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools import summarize_latency as summary_tool  # noqa: E402

PLAN = {"pano_capture": 40.0, "vo_query": 5.0, "lookdown_capture": 30.0, "model_encode": 10.0, "model_rpc": 200.0}
TIMED_SERVER = {"request_decode": 5.0, "system2_turn1_generate": 100.0, "handler_total": 190.0}
TIMED_INGEST = {"method": "ingest_frame", "rpc_ms": 5.0, "server_ms": {"jpeg_decode": 1.0, "ingest": 2.0, "total": 4.0}}
TIMED_QUERY = {"method": "query_relative_poses", "rpc_ms": 4.0, "server_ms": {"query": 1.0, "total": 2.0}}
UNTIMED_INGEST = {"method": "ingest_frame", "rpc_ms": 5.0, "server_ms": None, "server_cuda_mib": None}
UNTIMED_QUERY = {"method": "query_relative_poses", "rpc_ms": 4.0, "server_ms": None, "server_cuda_mib": None}
WARNING_CAUSE = "were not running with HEATMAPVLN_TIMING=1"


def _record(call_index, *, server, vo_rpc, episode="zsNo4HB9uLZ/0001"):
    return {
        "schema": "heatmapvln-latency-v1",
        "episode": episode,
        "scene_id": episode.split("/")[0],
        "episode_id": 1,
        "call_index": call_index,
        "step": 0,
        "plan_ms": PLAN,
        "kind": "trajectory",
        "pose_ready": False,
        "ppa_applied": False,
        "actions_returned": 4,
        "model_server_ms": server,
        "model_cuda_mib": None,
        "actions_executed": 2,
        "cycle_wall_ms": 500.0,
        "step_ms": {"vo_ingest": [6.0, 8.0], "env_step": [20.0, 30.0], "pano_capture": [40.0]},
        "vo_rpc": vo_rpc,
    }


def _run(tmp_path, records):
    timing = tmp_path / "workers" / "shard_00" / "timing"
    timing.mkdir(parents=True)
    (timing / "client.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "out"
    rc = summary_tool.main([str(tmp_path / "workers"), "--output-dir", str(out)])
    summary = json.loads((out / "latency_summary.json").read_text())
    markdown = (out / "latency_summary.md").read_text()
    return rc, summary, markdown


def test_servers_that_reported_their_stages_raise_nothing(tmp_path, capsys):
    records = [
        _record(0, server=TIMED_SERVER, vo_rpc=[TIMED_INGEST, TIMED_QUERY]),
        _record(1, server=TIMED_SERVER, vo_rpc=[TIMED_INGEST]),
    ]
    rc, summary, markdown = _run(tmp_path, records)

    assert rc == 0
    assert summary["server_timing_missing"] == {
        "model_server_calls": 0,
        "model_server_calls_seen": 2,
        "vo_rpcs": 0,
        "vo_rpcs_seen": 3,
    }
    assert "WARNING" not in markdown
    assert "WARNING" not in capsys.readouterr().err
    assert markdown.startswith("# Latency summary\n\n2 plan calls, 4 executed actions")


def test_untimed_servers_are_counted_and_flagged(tmp_path, capsys):
    records = [
        _record(0, server=None, vo_rpc=[UNTIMED_INGEST, UNTIMED_QUERY]),
        _record(1, server=None, vo_rpc=[UNTIMED_INGEST]),
        _record(2, server=None, vo_rpc=[]),  # a chunk without new frames makes no VO call
    ]
    rc, summary, markdown = _run(tmp_path, records)

    assert rc == 3
    assert summary["server_timing_missing"] == {
        "model_server_calls": 3,
        "model_server_calls_seen": 3,
        "vo_rpcs": 3,
        "vo_rpcs_seen": 3,
    }
    # Every existing field is still there: only the server tables are empty.
    assert summary["calls"] == 3 and summary["actions_executed"] == 6
    group = summary["groups"]["all"]
    assert group["model_server"] == {} and group["cuda_memory_mib"] == {}
    assert list(group["vo_server"]) == ["ingest_frame.round_trip", "query_relative_poses.round_trip"]
    assert "model_server" not in group["per_call"] and group["per_call"]["model"]["n"] == 3

    warning = (
        "**WARNING: 3 of 3 model-server calls and 3 of 3 VO RPCs carry no server-side stages: "
        "the server processes behind them were not running with HEATMAPVLN_TIMING=1, so this summary "
        "covers client-side stages only for them and no end-to-end latency may be quoted from it.**"
    )
    assert markdown.startswith("# Latency summary\n\n" + warning + "\n\n3 plan calls, 6 executed actions")
    assert warning.strip("*") in capsys.readouterr().err


def test_a_mix_counts_only_the_untimed_calls(tmp_path):
    """One server timed and the other not, or a server restarted with a different environment."""
    records = [
        _record(0, server=TIMED_SERVER, vo_rpc=[UNTIMED_INGEST, UNTIMED_QUERY]),  # VO server untimed
        _record(1, server=TIMED_SERVER, vo_rpc=[UNTIMED_INGEST]),
        _record(0, server=None, vo_rpc=[TIMED_INGEST, TIMED_QUERY], episode="8194nk5LbLH/0002"),
        _record(1, server={}, vo_rpc=[TIMED_INGEST], episode="8194nk5LbLH/0002"),
    ]
    rc, summary, markdown = _run(tmp_path, records)

    assert rc == 3
    assert summary["server_timing_missing"] == {
        "model_server_calls": 2,
        "model_server_calls_seen": 4,
        "vo_rpcs": 3,
        "vo_rpcs_seen": 6,
    }
    assert summary["groups"]["all"]["model_server"]["handler_total"]["n"] == 2
    assert summary["groups"]["all"]["vo_server"]["ingest_frame.total"]["n"] == 2
    head = markdown.split("\n\n", 2)[1]
    assert head.startswith("**WARNING: 2 of 4 model-server calls and 3 of 6 VO RPCs carry no server-side stages")
    assert WARNING_CAUSE in head


def test_only_the_vo_server_untimed_still_warns(tmp_path):
    records = [_record(0, server=TIMED_SERVER, vo_rpc=[UNTIMED_QUERY])]
    rc, summary, markdown = _run(tmp_path, records)
    assert rc == 3
    assert summary["server_timing_missing"]["model_server_calls"] == 0
    assert summary["server_timing_missing"]["vo_rpcs"] == 1
    assert "**WARNING: 0 of 1 model-server calls and 1 of 1 VO RPCs" in markdown


def test_a_call_the_model_server_never_answered_is_not_counted():
    """Without end_plan the record has no model_server_ms key: no response, nothing to time."""
    unanswered = _record(1, server=None, vo_rpc=[])
    del unanswered["model_server_ms"], unanswered["model_cuda_mib"], unanswered["kind"]
    records = [_record(0, server=TIMED_SERVER, vo_rpc=[]), unanswered]
    summary = summary_tool.summarise(records, ["x.jsonl"])
    assert summary["server_timing_missing"] == {
        "model_server_calls": 0,
        "model_server_calls_seen": 1,
        "vo_rpcs": 0,
        "vo_rpcs_seen": 0,
    }
    assert "WARNING" not in summary_tool.to_markdown(summary)
