"""The client launcher's guards for capped runs, external servers and pinned episode sets.

scripts/run_ppa_r2r_val_unseen_cuda.sh runs both the certified 4090 evaluation (local
CUDA servers) and the client half of the Ascend split (external servers through an SSH
tunnel).  docs/ops/ascend_910b_open_problems.md sections 2.2-2.4 describe four ways it
could hand back a wrong result without an error:

- a capped canary that restarts or is relaunched runs more episodes than its cap,
  because --max_episodes counts only episodes not already done;
- the Ascend slots use the 4090's own default ports, so the servers.json check passed
  as long as ports, slot count and bridge_off agreed, whoever was listening;
- an uncapped pinned-list run could record too few or too many episodes and still
  print "passed";
- a restart waited only for an open port, which in external mode is the local ssh
  forwarder, and could not tell "not ready yet" from "not the recorded servers".

Every test here runs the shell or the embedded Python: either the heredoc on its own,
located by the marker that opens it, or the whole launcher against a fake tree in
tmp_path with stub servers, a stub client, a stub Xvfb and a fake ``vla_rpc`` client.
Nothing here needs torch, a GPU, an NPU or the simulator.

macOS ships bash 3.2, whose ``set -u`` rejects ``"${array[@]}"`` on an empty array;
the deployment runs bash 5.  The launcher expands two arrays that are empty in the
default configuration (the model server's extra flags and the client's episode cap),
so the full-launcher tests choose flags that keep both non-empty, and the two tests
that need an uncapped run skip under a bash that cannot run it.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"
NPU_LAUNCHER = REPO / "scripts" / "ascend" / "run_ppa_servers_npu.sh"

BASH = shutil.which("bash")
GIT = shutil.which("git")

COMMIT = "0123456789abcdef0123456789abcdef01234567"
INSTANCE = "20261009_164000_4242"
VO_SEED = 1234
# What the launcher records in servers.json and the model server reports live.
MODEL_VERSION = "ppa-refine-v2"
SHARD0_EPISODES = [("sceneA", 1), ("sceneA", 2), ("sceneB", 3)]


# --------------------------------------------------------------------------- #
# Heredoc extraction
# --------------------------------------------------------------------------- #
def _heredocs(path: Path) -> list[tuple[str, str]]:
    """Every quoted heredoc in a script, as (opening line, body), in file order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    found = []
    index = 0
    while index < len(lines):
        match = re.search(r"<<'(\w+)'", lines[index])
        if not match:
            index += 1
            continue
        tag, body, end = match.group(1), [], index + 1
        while lines[end] != tag:
            body.append(lines[end])
            end += 1
        found.append((lines[index], "\n".join(body) + "\n"))
        index = end + 1
    return found


def _external_check_source() -> str:
    bodies = [body for opener, body in _heredocs(LAUNCHER) if "<<'EXT'" in opener]
    assert len(bodies) == 1, "the launcher should have exactly one <<'EXT' heredoc"
    return bodies[0]


def _episode_set_check_source() -> str:
    """The heredoc that follows the one printing the "passed" summary."""
    docs = _heredocs(LAUNCHER)
    summary = [i for i, (_, body) in enumerate(docs) if '"status": "passed"' in body]
    assert len(summary) == 1, "the launcher should print exactly one passed summary"
    assert summary[0] + 1 < len(docs), "no heredoc follows the summary"
    return docs[summary[0] + 1][1]


def _servers_json_writer_source() -> str:
    bodies = [body for _, body in _heredocs(NPU_LAUNCHER) if '"schema": "heatmapvln-npu-servers-v2"' in body]
    assert len(bodies) == 1, "the NPU launcher should write servers.json from exactly one heredoc"
    return bodies[0]


def _write_servers_json(path: Path, *, slots=2, model_base=52400, vo_base=52500, commit=COMMIT,
                        rng_seed=VO_SEED, timing=0, bridge_off=0, instance=INSTANCE, dirty=0,
                        num_sample_trajs="", num_inference_steps="") -> dict:
    """servers.json exactly as scripts/ascend/run_ppa_servers_npu.sh writes it.

    Running the writer's own heredoc, rather than a hand-made record, is what makes the
    good-record tests below mean "what the NPU launcher writes passes the client's
    check": a writer that changed a field's type or name would fail them.
    """
    argv = [str(path), str(slots), str(model_base), str(vo_base), "0,1", "2,3", commit,
            str(rng_seed), str(timing), str(bridge_off), instance, MODEL_VERSION, str(dirty),
            num_sample_trajs, num_inference_steps]
    result = subprocess.run([sys.executable, "-", *argv], input=_servers_json_writer_source(),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# (2) The external-mode servers.json check, run on its own
# --------------------------------------------------------------------------- #
def _run_external_check(path: Path, *, slots=2, model_base=52400, vo_base=52500, bridge_off=0,
                        timing=0, num_sample_trajs="", num_inference_steps="",
                        client_commit=COMMIT, allow_commit_mismatch=0):
    # The same positional order as the launcher's "$CLIENT_PYTHON" - ... <<'EXT' call.
    argv = [str(path), str(slots), str(model_base), str(vo_base), str(bridge_off), str(timing),
            num_sample_trajs, num_inference_steps, client_commit, str(allow_commit_mismatch)]
    return subprocess.run([sys.executable, "-", *argv], input=_external_check_source(),
                          capture_output=True, text=True, timeout=60)


def test_a_record_written_by_the_npu_launcher_passes_the_client_check(tmp_path):
    path = tmp_path / "servers.json"
    _write_servers_json(path)
    result = _run_external_check(path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["external_servers"]["server_instance"] == INSTANCE


def _mutate(record: dict, key: str, value) -> dict:
    record = dict(record)
    if value is _DELETE:
        record.pop(key, None)
    else:
        record[key] = value
    return record


_DELETE = object()

# One mutation per fatal field.  Each names the phrase the operator must see in the
# refusal, so a check that fails for the wrong reason does not count as a pass.
FATAL_MUTATIONS = [
    # The old checks, still in force.
    ("slots", 1, "servers expose 1 slot(s), this run wants 2"),
    ("model_ports", [52400, 52402], "model ports"),
    ("vo_ports", [52501, 52502], "VO ports"),
    ("bridge_off", 1, "servers bridge_off=1"),
    # A v1 record is what servers started before the identity fields existed write.
    ("schema", "heatmapvln-npu-servers-v1", "unexpected schema 'heatmapvln-npu-servers-v1'"),
    # The fields that tell an Ascend server set from the 4090's own CUDA pair.
    ("device", "cuda", "device='cuda', expected 'npu'"),
    ("device", _DELETE, "device=None, expected 'npu'"),
    ("server_instance", "", "has no server_instance"),
    ("server_instance", _DELETE, "has no server_instance"),
    ("repo_commit", "0123456", "repo_commit='0123456' is not a commit"),
    ("repo_commit", COMMIT.upper(), "is not a commit"),
    ("repo_commit", _DELETE, "repo_commit=None is not a commit"),
    ("repo_dirty", 1, "dirty working tree (repo_dirty=1)"),
    # Timing is per process: untimed servers behind a timed client used to pass.
    ("timing", 1, "servers timing=1, this run sets 0"),
    # The sampling configuration is part of the arm and lives on the server.
    ("num_sample_trajs", "16", "servers num_sample_trajs='16', this run sets ''"),
    ("num_inference_steps", "10", "servers num_inference_steps='10', this run sets ''"),
    # A valid commit that is not the client's.
    ("repo_commit", "f" * 40, "set PPA_EVAL_ALLOW_COMMIT_MISMATCH=1"),
]


@pytest.mark.parametrize(("key", "value", "phrase"), FATAL_MUTATIONS,
                         ids=[f"{key}={'deleted' if value is _DELETE else value}" for key, value, _ in FATAL_MUTATIONS])
def test_each_fatal_servers_json_field_is_refused_by_name(tmp_path, key, value, phrase):
    path = tmp_path / "servers.json"
    record = _mutate(_write_servers_json(path), key, value)
    path.write_text(json.dumps(record), encoding="utf-8")
    result = _run_external_check(path)
    assert result.returncode == 1
    assert "external server mismatch" in result.stderr
    assert phrase in result.stderr


def test_run_side_settings_must_match_the_servers_too(tmp_path):
    """A mismatch is a mismatch from either side: the run's flags against the record."""
    path = tmp_path / "servers.json"
    _write_servers_json(path, timing=0, num_sample_trajs="", num_inference_steps="")
    timed = _run_external_check(path, timing=1)
    assert timed.returncode == 1 and "servers timing=0, this run sets 1" in timed.stderr
    sampled = _run_external_check(path, num_sample_trajs="16")
    assert sampled.returncode == 1 and "servers num_sample_trajs='', this run sets '16'" in sampled.stderr

    _write_servers_json(path, timing=1, num_sample_trajs="16", num_inference_steps="10")
    agreed = _run_external_check(path, timing=1, num_sample_trajs="16", num_inference_steps="10")
    assert agreed.returncode == 0, agreed.stderr


def test_commit_mismatch_escape_hatch_covers_only_the_commit(tmp_path):
    """PPA_EVAL_ALLOW_COMMIT_MISMATCH=1 accepts another build, never an unidentifiable one."""
    path = tmp_path / "servers.json"
    _write_servers_json(path, commit="f" * 40)
    assert _run_external_check(path, allow_commit_mismatch=1).returncode == 0

    _write_servers_json(path, commit="f" * 40, dirty=1)
    dirty = _run_external_check(path, allow_commit_mismatch=1)
    assert dirty.returncode == 1 and "repo_dirty=1" in dirty.stderr

    record = _write_servers_json(path)
    record["repo_commit"] = "not-a-commit"
    path.write_text(json.dumps(record), encoding="utf-8")
    malformed = _run_external_check(path, allow_commit_mismatch=1)
    assert malformed.returncode == 1 and "is not a commit" in malformed.stderr


def test_a_client_outside_a_git_checkout_cannot_match_a_commit(tmp_path):
    """The launcher passes "unknown" when it cannot read its own commit; that never matches."""
    path = tmp_path / "servers.json"
    _write_servers_json(path)
    result = _run_external_check(path, client_commit="unknown")
    assert result.returncode == 1
    assert "this client is unknown" in result.stderr


# --------------------------------------------------------------------------- #
# (3) The episode-set check after the summary, run on its own
# --------------------------------------------------------------------------- #
def _episode_list(path: Path, episodes, *, bare=False) -> None:
    items = [{"scene_id": scene, "episode_id": episode} for scene, episode in episodes]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items if bare else {"episodes": items}), encoding="utf-8")


def _progress(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        if isinstance(row, dict):
            lines.append(json.dumps(row))
        else:
            scene, episode = row
            lines.append(json.dumps(_progress_row(scene, episode)))
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _progress_row(scene, episode) -> dict:
    return {"scene_id": scene, "episode_id": episode, "history_pose_source": "amb3r_vo_da3",
            "ppa_applied_calls": 3, "success": 1.0, "spl": 0.9, "os": 1.0, "ne": 0.5}


def _run_episode_set_check(workers: Path, lists: Path, shards: str):
    # The same positional order as the launcher: "$WORKERS_DIR" "$EPISODE_LISTS_DIR" "$SHARD_CSV".
    return subprocess.run([sys.executable, "-", str(workers), str(lists), shards],
                          input=_episode_set_check_source(), capture_output=True, text=True, timeout=60)


EPISODE_SET_CASES = [
    # name, recorded rows, expected exit, phrase
    ("exact", SHARD0_EPISODES, 0, None),
    ("exact_in_another_order", list(reversed(SHARD0_EPISODES)), 0, None),
    ("short", SHARD0_EPISODES[:2], 1, "shard 0: 1 listed episode(s) missing, 0 unlisted episode(s) present"),
    ("long", SHARD0_EPISODES + [("sceneC", 9)], 1, "shard 0: 0 listed episode(s) missing, 1 unlisted episode(s) present"),
    ("duplicate", SHARD0_EPISODES + [SHARD0_EPISODES[0]], 1, "shard 0: 1 duplicate episode row(s)"),
    ("swapped", SHARD0_EPISODES[:2] + [("sceneC", 9)], 1, "1 listed episode(s) missing, 1 unlisted episode(s) present"),
    ("row_without_ids", SHARD0_EPISODES + [{"success": 1.0}], 1, "has no scene_id/episode_id"),
    ("empty", [], 1, "3 listed episode(s) missing"),
]


@pytest.mark.parametrize(("name", "rows", "code", "phrase"), EPISODE_SET_CASES, ids=[c[0] for c in EPISODE_SET_CASES])
def test_episode_set_check_accepts_only_the_listed_episodes(tmp_path, name, rows, code, phrase):
    _episode_list(tmp_path / "lists" / "shard_00.json", SHARD0_EPISODES)
    _progress(tmp_path / "workers" / "shard_00" / "progress.json", rows)
    result = _run_episode_set_check(tmp_path / "workers", tmp_path / "lists", "0")
    assert result.returncode == code, result.stderr
    if phrase is not None:
        assert "episode set mismatch" in result.stderr
        assert phrase in result.stderr


def test_episode_set_check_compares_keys_the_way_the_client_does(tmp_path):
    """r2r_val_unseen.py:_load_episode_list reads ids with int(); a string id is the same episode."""
    _episode_list(tmp_path / "lists" / "shard_00.json", [(scene, str(episode)) for scene, episode in SHARD0_EPISODES])
    _progress(tmp_path / "workers" / "shard_00" / "progress.json", SHARD0_EPISODES)
    assert _run_episode_set_check(tmp_path / "workers", tmp_path / "lists", "0").returncode == 0


def test_episode_set_check_reads_a_bare_list_as_well(tmp_path):
    _episode_list(tmp_path / "lists" / "shard_00.json", SHARD0_EPISODES, bare=True)
    _progress(tmp_path / "workers" / "shard_00" / "progress.json", SHARD0_EPISODES[:1])
    result = _run_episode_set_check(tmp_path / "workers", tmp_path / "lists", "0")
    assert result.returncode == 1 and "2 listed episode(s) missing" in result.stderr


def test_episode_set_check_names_the_short_shard_among_several(tmp_path):
    """Pooled over shards, one short shard and one long one would cancel out in a count."""
    other = [("sceneD", 11), ("sceneD", 12)]
    _episode_list(tmp_path / "lists" / "shard_00.json", SHARD0_EPISODES)
    _episode_list(tmp_path / "lists" / "shard_03.json", other)
    _progress(tmp_path / "workers" / "shard_00" / "progress.json", SHARD0_EPISODES + [("sceneD", 13)])
    _progress(tmp_path / "workers" / "shard_03" / "progress.json", other[:1])
    result = _run_episode_set_check(tmp_path / "workers", tmp_path / "lists", "0,3")
    assert result.returncode == 1
    assert "shard 0: 0 listed episode(s) missing, 1 unlisted" in result.stderr
    assert "shard 3: 1 listed episode(s) missing, 0 unlisted" in result.stderr


def test_episode_set_check_fails_a_shard_that_wrote_nothing(tmp_path):
    # Only the exit status is pinned: the check has no message of its own for a missing
    # progress file and fails with a FileNotFoundError traceback, which the launcher
    # still turns into "the recorded episodes are not the listed ones".
    _episode_list(tmp_path / "lists" / "shard_00.json", SHARD0_EPISODES)
    _episode_list(tmp_path / "lists" / "shard_01.json", [("sceneE", 1)])
    _progress(tmp_path / "workers" / "shard_00" / "progress.json", SHARD0_EPISODES)
    result = _run_episode_set_check(tmp_path / "workers", tmp_path / "lists", "0,1")
    assert result.returncode != 0


# --------------------------------------------------------------------------- #
# The whole launcher against a fake tree
# --------------------------------------------------------------------------- #
# The client the launcher runs, standing in for r2r_val_unseen.py.  Each invocation
# takes the next step of $FAKE_PLAN: which rows to append to progress.json, what the
# fake servers should look like from now on, and the exit status.  Every argv is
# appended to $FAKE_PLAN.calls so a test can count restarts and read the flags.
STUB_CLIENT = r'''
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
plan_path = Path(os.environ["FAKE_PLAN"])
calls = plan_path.with_suffix(".calls")
count = len(calls.read_text().splitlines()) if calls.exists() else 0
with calls.open("a") as handle:
    handle.write(json.dumps(argv) + "\n")
plan = json.loads(plan_path.read_text())["client"]
step = plan[min(count, len(plan) - 1)]
output = Path(argv[argv.index("--output_path") + 1])
output.mkdir(parents=True, exist_ok=True)
listed = json.loads(Path(argv[argv.index("--episode_list") + 1]).read_text())["episodes"]
rows = step.get("rows", [])
if rows == "listed":
    rows = [[item["scene_id"], item["episode_id"]] for item in listed]
with (output / "progress.json").open("a") as handle:
    for scene, episode in rows:
        handle.write(json.dumps({
            "scene_id": scene, "episode_id": episode, "history_pose_source": "amb3r_vo_da3",
            "ppa_applied_calls": 3, "success": 1.0, "spl": 0.9, "os": 1.0, "ne": 0.5,
        }) + "\n")
if "rpc_state" in step:
    Path(os.environ["FAKE_RPC_STATE"]).write_text(json.dumps(step["rpc_state"]))
raise SystemExit(step.get("exit", 0))
'''

# vla_rpc.client as rpc_ready imports it.  It answers from $FAKE_RPC_STATE, keyed by
# port, so a test decides what each server advertises and whether it is healthy.
# connect() waits for the files in "wait_for" (the local stub servers' ready markers),
# because the launcher reads the model log right after the first successful probe.
FAKE_VLA_CLIENT = r'''
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace


def _state():
    return json.loads(Path(os.environ["FAKE_RPC_STATE"]).read_text())


class VLAClient:
    def __init__(self, server_addr, timeout_ms=0):
        self.port = server_addr.rsplit(":", 1)[1]

    def connect(self):
        deadline = time.time() + 30
        while time.time() < deadline and not all(Path(p).exists() for p in _state().get("wait_for", [])):
            time.sleep(0.05)

    def _server(self):
        return _state()["servers"][self.port]

    def get_server_info(self):
        server = self._server()
        return SimpleNamespace(supported_formats=list(server["formats"]),
                               model_version=server.get("model_version", "ppa-refine-v2"))

    def health_check(self):
        return bool(self._server()["healthy"])

    def close(self):
        pass
'''

# Local-mode servers: print the preflight evidence the launcher greps for, mark
# themselves ready, and wait to be stopped by the launcher's cleanup.
STUB_MODEL_SERVER = r'''
import os
import sys
import time
from pathlib import Path

print("Formal PPA online AMB3R runtime enabled", flush=True)
if "--ppa_bridge_off" in sys.argv:
    print("PPA bridge off (EXP-20 A1)", flush=True)
Path(os.environ["FAKE_READY_DIR"], "model").touch()
time.sleep(300)
'''

STUB_VO_SERVER = r'''
import os
import time
from pathlib import Path

Path(os.environ["FAKE_READY_DIR"], "vo").touch()
time.sleep(300)
'''

# Xvfb: the launcher only checks that the display's TCP port opens.
STUB_XVFB = r'''
import socket
import sys

display = int(sys.argv[1].lstrip(":"))
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 6000 + display))
server.listen(64)
while True:
    connection, _ = server.accept()
    connection.close()
'''


def _free_port(low: int = 0) -> int:
    for _ in range(200):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port > low:
            return port
    raise RuntimeError("no free port")


class _Listener:
    """A TCP port held open, standing in for the local end of the SSH tunnel."""

    def __init__(self, port: int):
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", port))
        self.server.listen(64)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            connection.close()

    def close(self):
        self.server.close()


def _executable(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _bash_expands_empty_arrays() -> bool:
    if BASH is None:
        return False
    probe = subprocess.run([BASH, "-c", 'set -u; a=(); : "${a[@]}"'], capture_output=True)
    return probe.returncode == 0


def _model_formats(*, instance=INSTANCE, slot=0, device="npu", timing=0):
    # What rpc_model_server.py advertises with --server_instance (and only the
    # protocol without it, which is what the certified CUDA servers run).
    return ["ppa-online-amb3r-v1", f"heatmapvln-instance:{instance}/slot{slot}",
            f"heatmapvln-device:{device}", f"heatmapvln-timing:{timing}"]


def _vo_formats(*, instance=INSTANCE, slot=0, device="npu", timing=0, seed=VO_SEED):
    # What rpc_amb3r_vo_server.py advertises with --server-instance.
    return ["json+jpeg", f"heatmapvln-instance:{instance}/slot{slot}", f"heatmapvln-device:{device}",
            f"heatmapvln-timing:{timing}", f"heatmapvln-vo-rng-seed:{seed}"]


class FakeTree:
    """Everything the launcher's preflight requires, under tmp_path, with stubs."""

    def __init__(self, tmp_path: Path):
        if BASH is None:
            pytest.skip("bash is not installed")
        self.tmp = tmp_path
        self.root = tmp_path / "root"
        self.repo = self.root / "HeatmapVLN"
        self.plan_dir = self.root / "evaluation_plans" / "internnav_native_r2r_val_unseen_8gpu_20260802"
        self.cohorts = self.plan_dir / "cohorts"
        self.output = tmp_path / "out"
        self.ready = tmp_path / "ready"
        self.plan = tmp_path / "plan.json"
        self.rpc_state = tmp_path / "rpc_state.json"
        self.model_port = _free_port()
        self.vo_port = _free_port()
        self.display = _free_port(low=6000) - 6000
        self.listeners: list[_Listener] = []

        def stub(relative: str, content: str | bytes = "stub\n") -> Path:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            (path.write_bytes if isinstance(content, bytes) else path.write_text)(content)
            return path

        stub("HeatmapVLN/configs/ppa_action_refine_v2_8gpu.yaml")
        stub("HeatmapVLN/scripts/evaluation/r2r_val_unseen.py", STUB_CLIENT)
        stub("HeatmapVLN/scripts/evaluation/rpc_model_server.py", STUB_MODEL_SERVER)
        stub("HeatmapVLN/scripts/amb3r_vo/rpc_amb3r_vo_server.py", STUB_VO_SERVER)
        stub("rpc/src/vla_rpc/__init__.py", "")
        stub("rpc/src/vla_rpc/client.py", FAKE_VLA_CLIENT)
        stub(str((self.plan_dir / "tools" / "merge_shards.py").relative_to(self.root)),
             "raise SystemExit('the merge tool must not run on a one-shard run')\n")
        stub("R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz", gzip.compress(b"{}"))
        stub("weights/ppa_refine_v2_best.pth")
        stub("amb3r/checkpoints/DA3NESTED-GIANT-LARGE/model.safetensors")
        stub("amb3r/slam/slam_config.yaml")
        (self.root / "InternNav_Model").mkdir()
        (tmp_path / "dataset" / "mp3d").mkdir(parents=True)
        for shard in range(8):
            episodes = SHARD0_EPISODES if shard == 0 else [(f"scene{shard}", 100 + shard)]
            _episode_list(self.cohorts / f"shard_0{shard}.json", episodes)
            (self.cohorts / f"dataset_shard_0{shard}.json.gz").write_bytes(gzip.compress(b"{}"))
        self.ready.mkdir()
        self.xvfb = _executable(tmp_path / "bin" / "Xvfb", STUB_XVFB)
        # An Xvfb that exits at once: the first thing after the whole preflight, so a
        # run that dies with "Xvfb slot 0 failed" got past every guard tested here.
        self.dead_xvfb = tmp_path / "bin" / "Xvfb-dead"
        self.dead_xvfb.write_text("#!/bin/sh\nexit 1\n")
        self.dead_xvfb.chmod(0o755)
        self.set_plan([{"rows": "listed"}])
        self.set_servers(model=["ppa-online-amb3r-v1"], vo=["json+jpeg"])

    def set_plan(self, steps) -> None:
        self.plan.write_text(json.dumps({"client": steps}))

    def rpc_state_for(self, *, model, vo, healthy=True, wait_for=(),
                      model_version=MODEL_VERSION) -> dict:
        return {
            "servers": {str(self.model_port): {"formats": model, "healthy": healthy,
                                               "model_version": model_version},
                        str(self.vo_port): {"formats": vo, "healthy": healthy}},
            "wait_for": [str(path) for path in wait_for],
        }

    def set_servers(self, **kwargs) -> None:
        self.rpc_state.write_text(json.dumps(self.rpc_state_for(**kwargs)))

    def client_calls(self) -> list[list[str]]:
        calls = self.plan.with_suffix(".calls")
        if not calls.exists():
            return []
        return [json.loads(line) for line in calls.read_text().splitlines()]

    def runtime_dirs(self, output: Path | None = None) -> list[Path]:
        runtime = (output or self.output) / "runtime"
        return sorted(runtime.iterdir()) if runtime.exists() else []

    # External mode ---------------------------------------------------------------
    def external(self, *, commit: str | None = None, timing=0, **writer) -> Path:
        """A copied server runtime directory, a git client checkout and open tunnel ports."""
        if GIT is None:
            pytest.skip("git is not installed")
        git = [GIT, "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run([GIT, "init", "-q", str(self.repo)], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-q", "-m", "fake client tree"], check=True)
        head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        server_dir = self.tmp / "server_runtime"
        (server_dir / "logs").mkdir(parents=True)
        (server_dir / "logs" / "model_0.log").write_text("Formal PPA online AMB3R runtime enabled\n")
        _write_servers_json(server_dir / "servers.json", slots=1, model_base=self.model_port,
                            vo_base=self.vo_port, commit=commit or head, timing=timing, **writer)
        self.listeners += [_Listener(self.model_port), _Listener(self.vo_port)]
        self.set_servers(model=_model_formats(timing=timing), vo=_vo_formats(timing=timing))
        return server_dir

    def close(self) -> None:
        for listener in self.listeners:
            listener.close()

    def env(self, **overrides) -> dict:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.tmp),
            "LC_ALL": "C",
            "PPA_EVAL_ROOT": str(self.root),
            "PPA_EVAL_SCENES_DIR": str(self.tmp / "dataset"),
            "PPA_EVAL_PYTHON": sys.executable,
            "PPA_EVAL_XVFB": str(self.xvfb),
            "PPA_EVAL_OUTPUT_ROOT": str(self.output),
            "PPA_EVAL_RUN_STAMP": "test",
            "PPA_EVAL_GPU_DEVICES": "0",
            "PPA_EVAL_SHARDS": "0",
            "PPA_EVAL_DISPLAY_BASE": str(self.display),
            "PPA_EVAL_MODEL_PORT_BASE": str(self.model_port),
            "PPA_EVAL_VO_PORT_BASE": str(self.vo_port),
            "PPA_EVAL_SERVER_STAGGER_S": "0",
            "FAKE_PLAN": str(self.plan),
            "FAKE_RPC_STATE": str(self.rpc_state),
            "FAKE_READY_DIR": str(self.ready),
        }
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        return env

    def run(self, timeout: float = 120, **overrides) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, str(LAUNCHER)], env=self.env(**overrides), capture_output=True,
                              text=True, timeout=timeout, stdin=subprocess.DEVNULL)


@pytest.fixture
def tree(tmp_path):
    fake = FakeTree(tmp_path)
    yield fake
    fake.close()


needs_empty_arrays = pytest.mark.skipif(
    not _bash_expands_empty_arrays(),
    reason="this bash rejects an empty array under set -u (bash < 4.4); the launcher targets bash 5",
)


# --------------------------------------------------------------------------- #
# (1) The capped-run guard
# --------------------------------------------------------------------------- #
CAPPED_RETRY_REFUSAL = (
    "PPA_EVAL_MAX_EPISODES_PER_SHARD with PPA_EVAL_SHARD_RETRIES=1 would run 2 more new episodes per restart"
)


def test_a_capped_run_that_would_retry_is_refused_before_anything_starts(tree):
    result = tree.run(PPA_EVAL_MAX_EPISODES_PER_SHARD=2, PPA_EVAL_SHARD_RETRIES=1)
    assert result.returncode == 2, result.stderr
    assert CAPPED_RETRY_REFUSAL in result.stderr
    # Refused in the preflight: no runtime directory, no Xvfb, no client.
    assert tree.runtime_dirs() == []
    assert tree.client_calls() == []


def test_a_capped_run_into_a_directory_with_rows_is_refused(tree):
    progress = tree.output / "workers" / "shard_00" / "progress.json"
    _progress(progress, SHARD0_EPISODES[:2])
    result = tree.run(PPA_EVAL_MAX_EPISODES_PER_SHARD=2)
    assert result.returncode == 2, result.stderr
    assert f"capped run into {progress}, which already has 2 episode(s)" in result.stderr
    assert "the cap would add 2 more" in result.stderr
    assert tree.runtime_dirs() == []


@pytest.mark.parametrize(
    ("overrides", "progress_rows", "progress_shard"),
    [
        # The documented canary: capped, fresh directory, no retries.
        ({"PPA_EVAL_MAX_EPISODES_PER_SHARD": 2}, None, 0),
        # The certified full run resumes into its own rows with retries; uncapped, so
        # --resume plus a fixed list finishes exactly the listed episodes.
        ({"PPA_EVAL_SHARD_RETRIES": 2}, SHARD0_EPISODES[:2], 0),
        # A crashed launch can leave an empty progress.json: no rows, nothing to add to.
        ({"PPA_EVAL_MAX_EPISODES_PER_SHARD": 2}, [], 0),
        # Rows of a shard this run does not touch are not this run's concern.
        ({"PPA_EVAL_MAX_EPISODES_PER_SHARD": 2}, SHARD0_EPISODES[:1], 5),
        # The explicit opt-in lets both through.
        ({"PPA_EVAL_MAX_EPISODES_PER_SHARD": 2, "PPA_EVAL_SHARD_RETRIES": 1,
          "PPA_EVAL_ALLOW_CAPPED_RESUME": 1}, SHARD0_EPISODES[:2], 0),
    ],
    ids=["fresh_canary", "uncapped_resume_with_retries", "empty_progress_file", "other_shard_rows", "opt_in"],
)
def test_runs_the_guard_must_not_refuse_get_past_the_preflight(tree, overrides, progress_rows, progress_shard):
    if progress_rows is not None:
        _progress(tree.output / "workers" / f"shard_0{progress_shard}" / "progress.json", progress_rows)
    result = tree.run(PPA_EVAL_XVFB=tree.dead_xvfb, **overrides)
    assert result.returncode == 2
    assert "Xvfb slot 0 failed" in result.stderr, result.stderr
    assert "capped run" not in result.stderr and "would run" not in result.stderr


def test_the_capped_resume_switch_takes_only_0_or_1(tree):
    result = tree.run(PPA_EVAL_ALLOW_CAPPED_RESUME="yes", PPA_EVAL_MAX_EPISODES_PER_SHARD=2)
    assert result.returncode == 2
    assert "PPA_EVAL_ALLOW_CAPPED_RESUME must be 0 or 1" in result.stderr
    assert tree.runtime_dirs() == []


def test_external_mode_names_the_missing_output_root_before_the_capped_guard(tree):
    """Pointing the operator at the default directory's rows would send them to the wrong place.

    Without PPA_EVAL_OUTPUT_ROOT the capped default is $ROOT/eval_runs/canary_seed42,
    the 4090's own canary directory.  The rows there must not produce the capped-resume
    message: the fix is to name an output root, not to clean that directory.
    """
    server_dir = tree.external()
    default = tree.root / "eval_runs" / "canary_seed42"
    _progress(default / "workers" / "shard_00" / "progress.json", SHARD0_EPISODES[:2])
    result = tree.run(PPA_EVAL_EXTERNAL_SERVERS=1, PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir,
                      PPA_EVAL_OUTPUT_ROOT=None, PPA_EVAL_MAX_EPISODES_PER_SHARD=2, PPA_EVAL_SHARD_RETRIES=1)
    assert result.returncode == 2
    assert "requires an explicit PPA_EVAL_OUTPUT_ROOT" in result.stderr
    assert "capped run" not in result.stderr and "would run" not in result.stderr
    assert tree.runtime_dirs(default) == []


def test_external_mode_with_an_explicit_root_still_gets_the_capped_guard(tree):
    server_dir = tree.external()
    progress = tree.output / "workers" / "shard_00" / "progress.json"
    _progress(progress, SHARD0_EPISODES[:1])
    result = tree.run(PPA_EVAL_EXTERNAL_SERVERS=1, PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir,
                      PPA_EVAL_MAX_EPISODES_PER_SHARD=2)
    assert result.returncode == 2
    assert f"capped run into {progress}, which already has 1 episode(s)" in result.stderr
    assert tree.runtime_dirs() == []


def test_an_opted_in_capped_restart_does_run_past_its_cap(tree):
    """Why the guard exists, shown on the launcher itself.

    With PPA_EVAL_ALLOW_CAPPED_RESUME=1 a one-episode canary whose client dies after
    recording its episode is restarted with the same --max_episodes 1, and --resume
    skips the recorded episode, so the restart records a second one.  progress.json
    ends with two rows and nothing in it says which one was the extra.
    """
    server_dir = tree.external()
    tree.set_plan([{"rows": [list(SHARD0_EPISODES[0])], "exit": 1}, {"rows": [list(SHARD0_EPISODES[1])]}])
    result = tree.run(timeout=180, PPA_EVAL_EXTERNAL_SERVERS=1, PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir,
                      PPA_EVAL_MAX_EPISODES_PER_SHARD=1, PPA_EVAL_SHARD_RETRIES=1,
                      PPA_EVAL_ALLOW_CAPPED_RESUME=1, PPA_EVAL_SERVER_START_TIMEOUT_S=0)
    assert result.returncode == 0, result.stderr
    calls = tree.client_calls()
    assert len(calls) == 2
    for argv in calls:
        assert argv[argv.index("--max_episodes") + 1] == "1"
        assert "--resume" in argv
    rows = (tree.output / "workers" / "shard_00" / "progress.json").read_text().splitlines()
    assert len(rows) == 2
    assert '"episodes": 2' in result.stdout


# --------------------------------------------------------------------------- #
# (2) The servers.json check, wired through the launcher
# --------------------------------------------------------------------------- #
EXTERNAL_CANARY = {"PPA_EVAL_EXTERNAL_SERVERS": 1, "PPA_EVAL_MAX_EPISODES_PER_SHARD": 1,
                   "PPA_EVAL_SERVER_START_TIMEOUT_S": 0}


def test_external_canary_against_the_recorded_servers_completes(tree):
    """The whole external path: servers.json check, identity probe, client, summary."""
    server_dir = tree.external()
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **EXTERNAL_CANARY)
    assert result.returncode == 0, result.stderr
    assert "[ppa-eval] COMPLETE" in result.stdout
    assert len(tree.client_calls()) == 1
    assert (tree.runtime_dirs()[0] / "external_servers.json").exists()


@pytest.mark.parametrize(
    ("writer", "run", "phrase"),
    [
        # The launcher reads its own commit with git and hands it to the check.
        ({"commit": "f" * 40}, {}, "set PPA_EVAL_ALLOW_COMMIT_MISMATCH=1"),
        # PPA_EVAL_TIMING reaches the check, so untimed servers cannot pass a timed run.
        ({"timing": 0}, {"PPA_EVAL_TIMING": 1}, "servers timing=0, this run sets 1"),
        ({}, {"PPA_EVAL_NUM_INFERENCE_STEPS": 10}, "servers num_inference_steps='', this run sets '10'"),
        ({}, {"PPA_EVAL_BRIDGE_OFF": 1}, "servers bridge_off=0, this run sets 1"),
        ({"dirty": 1}, {}, "repo_dirty=1"),
    ],
    ids=["commit", "timing", "num_inference_steps", "bridge_off", "dirty"],
)
def test_the_launcher_hands_the_check_the_runs_own_settings(tree, writer, run, phrase):
    """The heredoc tests above prove the check; these prove the launcher feeds it the right argv."""
    server_dir = tree.external(**writer)
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **{**EXTERNAL_CANARY, **run})
    assert result.returncode == 2
    assert "external servers.json does not match this run" in result.stderr
    assert phrase in result.stderr
    assert tree.client_calls() == []


def test_the_commit_escape_hatch_is_wired_through(tree):
    server_dir = tree.external(commit="f" * 40)
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, PPA_EVAL_ALLOW_COMMIT_MISMATCH=1, **EXTERNAL_CANARY)
    assert result.returncode == 0, result.stderr
    assert "[ppa-eval] COMPLETE" in result.stdout


@pytest.mark.parametrize(
    ("marker", "phrase"),
    [
        ("STOPPED", "that server set has already been stopped"),
        ("RETIRED", "at least one slot was retired"),
        ("logs/vo_0.log", "an external server log reports its NPU unusable"),
    ],
)
def test_a_stopped_or_poisoned_server_set_is_not_resumed_against(tree, marker, phrase):
    server_dir = tree.external()
    path = server_dir / marker
    path.write_text("NPU device unusable after a failed request: slot 0\n")
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **EXTERNAL_CANARY)
    assert result.returncode == 2
    assert phrase in result.stderr
    assert tree.client_calls() == []


# --------------------------------------------------------------------------- #
# (4) rpc_ready: 1 is "not ready", 3 is "not the recorded servers"
# --------------------------------------------------------------------------- #
def test_startup_refuses_servers_that_are_not_the_recorded_set(tree):
    """The 4090's own CUDA pair on the same default ports answers the protocol but not the identity.

    Start-up timeout 0 means a probe that returned "not ready" (1) would die with
    "RPC startup timeout"; only the "wrong servers" exit (3) gives this message.
    """
    server_dir = tree.external()
    tree.set_servers(model=["ppa-online-amb3r-v1"], vo=["json+jpeg"])
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **EXTERNAL_CANARY)
    assert result.returncode == 2
    assert "are not the ones in" in result.stderr
    assert "RPC startup timeout" not in result.stderr
    assert tree.client_calls() == []


@pytest.mark.parametrize(
    ("model", "vo"),
    [
        (_model_formats(instance="20261009_000000_1"), _vo_formats(instance="20261009_000000_1")),
        (_model_formats(device="cuda"), _vo_formats(device="cuda")),
        (_model_formats(), _vo_formats(timing=1)),
        (_model_formats(), _vo_formats(seed=VO_SEED + 1)),
        (_model_formats(slot=1), _vo_formats()),
        (["ppa-online-amb3r-v2", *_model_formats()[1:]], _vo_formats()),
    ],
    ids=["other_instance", "cuda_device", "vo_timing", "vo_seed", "other_slot", "other_protocol"],
)
def test_each_identity_field_is_demanded_at_startup(tree, model, vo):
    server_dir = tree.external()
    tree.set_servers(model=model, vo=vo)
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **EXTERNAL_CANARY)
    assert result.returncode == 2
    assert "are not the ones in" in result.stderr


def test_a_live_model_version_other_than_the_recorded_one_is_refused(tree):
    """The build, as the running server names it, must be the one the evidence describes.

    The instance token says "these are the processes servers.json was written from";
    model_version says "and they are serving the build it recorded".  A server set
    restarted from a different checkout on the same ports keeps answering, so without
    this the run would be attributed to the recorded commit.
    """
    server_dir = tree.external()
    tree.set_servers(model=_model_formats(), vo=_vo_formats(), model_version="ppa-refine-v1")
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **EXTERNAL_CANARY)
    assert result.returncode == 2
    assert "are not the ones in" in result.stderr
    assert tree.client_calls() == []


def test_startup_treats_an_unhealthy_server_as_not_ready(tree):
    server_dir = tree.external()
    tree.set_servers(model=_model_formats(), vo=_vo_formats(), healthy=False)
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, **EXTERNAL_CANARY)
    assert result.returncode == 2
    assert "RPC startup timeout at slot 0" in result.stderr
    assert "are not the ones in" not in result.stderr


def _restart_plan(tree, after_death: dict) -> None:
    tree.set_plan([{"rows": [list(SHARD0_EPISODES[0])], "exit": 1, "rpc_state": after_death}, {"rows": "listed"}])


def test_a_restart_gives_up_the_shard_when_other_servers_answer(tree):
    """Retrying cannot fix the wrong server set, so the remaining retries are not spent on it."""
    server_dir = tree.external()
    _restart_plan(tree, tree.rpc_state_for(model=["ppa-online-amb3r-v1"], vo=["json+jpeg"]))
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, PPA_EVAL_SHARD_RETRIES=2,
                      PPA_EVAL_ALLOW_CAPPED_RESUME=1, **EXTERNAL_CANARY)
    assert result.returncode == 2
    assert "the servers on these ports are not the recorded set; giving up the shard" in result.stderr
    assert "an evaluation slot failed" in result.stderr
    assert len(tree.client_calls()) == 1


def test_a_restart_waits_for_a_healthy_server_and_gives_up_at_the_deadline(tree):
    """An Ascend device error leaves the server LISTENing but NOT_SERVING; that is "not ready"."""
    server_dir = tree.external()
    _restart_plan(tree, tree.rpc_state_for(model=_model_formats(), vo=_vo_formats(), healthy=False))
    result = tree.run(PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir, PPA_EVAL_SHARD_RETRIES=2,
                      PPA_EVAL_ALLOW_CAPPED_RESUME=1, **EXTERNAL_CANARY)
    assert result.returncode == 2
    assert "no healthy RPC within 0s; giving up the shard" in result.stderr
    assert "not the recorded set" not in result.stderr
    assert len(tree.client_calls()) == 1


# --------------------------------------------------------------------------- #
# The certified local-server path is unaffected
# --------------------------------------------------------------------------- #
# PPA_EVAL_BRIDGE_OFF=1 and a cap keep the two arrays the launcher expands non-empty,
# so these run under macOS bash 3.2 too; neither changes the readiness probe.
LOCAL_CANARY = {"PPA_EVAL_MAX_EPISODES_PER_SHARD": 3, "PPA_EVAL_BRIDGE_OFF": 1}


def test_local_servers_without_an_identity_are_ready(tree):
    """The certified CUDA servers advertise only their protocol and are never asked who they are."""
    tree.set_servers(model=["ppa-online-amb3r-v1"], vo=["json+jpeg"],
                     wait_for=[tree.ready / "model", tree.ready / "vo"])
    result = tree.run(**LOCAL_CANARY)
    assert result.returncode == 0, result.stderr
    assert "[ppa-eval] COMPLETE" in result.stdout
    assert "slot=0 gpu=0 vo_gpu=0" in result.stdout
    assert len(tree.client_calls()) == 1


def test_local_servers_speaking_another_protocol_are_only_not_ready(tree):
    """Locally the wrong protocol is "keep waiting" (1), as before, never the external-only 3."""
    tree.set_servers(model=["ppa-online-amb3r-v2"], vo=["json+jpeg"],
                     wait_for=[tree.ready / "model", tree.ready / "vo"])
    result = tree.run(PPA_EVAL_SERVER_START_TIMEOUT_S=0, **LOCAL_CANARY)
    assert result.returncode == 2
    assert "RPC startup timeout at slot 0" in result.stderr
    assert "are not the ones in" not in result.stderr


# --------------------------------------------------------------------------- #
# (3) The episode-set check, wired through the launcher
# --------------------------------------------------------------------------- #
@needs_empty_arrays
def test_an_uncapped_pinned_run_that_recorded_the_list_completes(tree):
    server_dir = tree.external()
    tree.set_plan([{"rows": "listed"}])
    result = tree.run(PPA_EVAL_EXTERNAL_SERVERS=1, PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir,
                      PPA_EVAL_SERVER_START_TIMEOUT_S=0)
    assert result.returncode == 0, result.stderr
    assert '"episodes": 3' in result.stdout
    assert "[ppa-eval] COMPLETE" in result.stdout


@needs_empty_arrays
def test_an_uncapped_pinned_run_that_came_out_short_fails_after_printing_the_summary(tree):
    """The numbers stay visible, but the run must not end as COMPLETE."""
    server_dir = tree.external()
    tree.set_plan([{"rows": [list(episode) for episode in SHARD0_EPISODES[:2]]}])
    result = tree.run(PPA_EVAL_EXTERNAL_SERVERS=1, PPA_EVAL_EXTERNAL_SERVER_DIR=server_dir,
                      PPA_EVAL_SERVER_START_TIMEOUT_S=0)
    assert result.returncode == 2
    assert '"status": "passed"' in result.stdout and '"episodes": 2' in result.stdout
    assert "shard 0: 1 listed episode(s) missing" in result.stderr
    assert "the recorded episodes are not the listed ones" in result.stderr
    assert "[ppa-eval] COMPLETE" not in result.stdout
