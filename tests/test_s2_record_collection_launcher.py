"""EXP-23 collector: the train cohorts and the collection launcher (design §2 A).

scripts/tools/build_r2r_train_cohorts.py runs on a synthetic train.json.gz.
scripts/run_s2_dagger_collect_cuda.sh runs whole against a fake tree in tmp_path: stub
model and VO servers, a stub Xvfb, a fake ``vla_rpc`` client and a stub client that
writes progress rows and record files the way r2r_val_unseen.py --s2_record_dir does.
Nothing here needs torch, a GPU or the simulator; the launcher needs bash >= 4.4
(the deployment runs bash 5) and the claim liveness check reads /proc.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "tools" / "build_r2r_train_cohorts.py"
LAUNCHER = REPO / "scripts" / "run_s2_dagger_collect_cuda.sh"
BASH = shutil.which("bash")


# ── the cohort builder ─────────────────────────────────────────────────────


def _train_json(path: Path, *, scenes=("sceneA", "sceneB", "sceneC"), routes_per_scene=4) -> list[dict]:
    episodes, episode_id = [], 0
    for scene_index, scene in enumerate(scenes):
        for route in range(routes_per_scene):
            trajectory_id = 10 * scene_index + route
            for _ in range(4 if (scene, route) == (scenes[-1], 3) else 3):
                episode_id += 1
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "trajectory_id": trajectory_id,
                        "scene_id": f"mp3d/{scene}/{scene}.glb",
                        "instruction": {"instruction_text": f"go {episode_id}", "instruction_tokens": [1, 2]},
                        "goals": [{"position": [1.0, 0.0, 2.0], "radius": 3.0}],
                        "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]],
                        "start_position": [0.0, 0.0, 0.0],
                        "start_rotation": [0.0, 0.0, 0.0, 1.0],
                        "info": {"geodesic_distance": 2.2},
                    }
                )
    payload = {"episodes": episodes, "instruction_vocab": {"word_list": ["go"], "num_vocab": 1}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return episodes


def _build(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(BUILDER), *args], capture_output=True, text=True)


def _shards(output: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((output / "manifest.json").read_text())
    lists = [json.loads((output / record["path"]).read_text()) for record in manifest["shards"]]
    datasets = []
    for record in manifest["shards"]:
        with gzip.open(output / record["dataset_path"], "rt", encoding="utf-8") as handle:
            datasets.append(json.load(handle))
    return manifest, lists, datasets


def test_shards_are_route_disjoint_exact_and_in_the_evaluation_format(tmp_path):
    dataset = tmp_path / "train" / "train.json.gz"
    episodes = _train_json(dataset)
    result = _build("--dataset", str(dataset), "--output-dir", str(tmp_path / "out"), "--num-shards", "3", "--seed", "7")
    assert result.returncode == 0, result.stderr
    manifest, lists, datasets = _shards(tmp_path / "out")
    assert manifest["schema"] == "heatmapvln-r2r-train-cohorts-v1" and manifest["split"] == "train"
    assert manifest["num_shards"] == 3 and manifest["total_episodes"] == len(episodes) == 37
    assert manifest["total_routes"] == 12 and manifest["route_disjoint"] is True
    keys = [(item["scene_id"], item["episode_id"]) for shard in lists for item in shard["episodes"]]
    assert sorted(keys) == sorted((e["scene_id"].split("/")[1], e["episode_id"]) for e in episodes)
    owners = {}
    for index, shard in enumerate(lists):
        assert shard["split"] == "train" and shard["shard_index"] == index
        for item in shard["episodes"]:
            assert owners.setdefault((item["scene_id"], item["trajectory_id"]), index) == index
    # The dataset shard is train.json.gz cut to the list, in the list's order, the rest kept.
    raw = {e["episode_id"]: e for e in episodes}
    for shard, data in zip(lists, datasets):
        assert data["instruction_vocab"] == {"word_list": ["go"], "num_vocab": 1}
        assert [e["episode_id"] for e in data["episodes"]] == [item["episode_id"] for item in shard["episodes"]]
        assert all(e == raw[e["episode_id"]] for e in data["episodes"])
        scenes = [item["scene_id"] for item in shard["episodes"]]
        assert scenes == sorted(scenes)  # grouped by scene, as Habitat iterates
    sizes = [len(shard["episodes"]) for shard in lists]
    assert max(sizes) - min(sizes) <= 4


def test_the_same_arguments_rebuild_identical_bytes_and_the_seed_matters(tmp_path):
    dataset = tmp_path / "train.json.gz"
    _train_json(dataset)
    for name, seed in (("a", "3"), ("b", "3"), ("c", "4")):
        result = _build("--dataset", str(dataset), "--output-dir", str(tmp_path / name), "--num-shards", "4",
                        "--seed", seed)
        assert result.returncode == 0, result.stderr
    files = sorted(path.name for path in (tmp_path / "a").iterdir())
    assert files == sorted(path.name for path in (tmp_path / "b").iterdir())
    assert all((tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes() for f in files)
    assert (tmp_path / "a" / "shard_000.json").read_bytes() != (tmp_path / "c" / "shard_000.json").read_bytes()


def test_instructions_per_route_scene_exclusion_route_cap_and_shard_size(tmp_path):
    dataset = tmp_path / "train.json.gz"
    _train_json(dataset)
    result = _build("--dataset", str(dataset), "--output-dir", str(tmp_path / "one"), "--episodes-per-route", "1",
                    "--exclude-scenes", "sceneB", "--max-routes", "5", "--shard-size", "2")
    assert result.returncode == 0, result.stderr
    manifest, lists, _ = _shards(tmp_path / "one")
    items = [item for shard in lists for item in shard["episodes"]]
    assert len(items) == 5 and len({(i["scene_id"], i["trajectory_id"]) for i in items}) == 5
    assert all(item["scene_id"] != "sceneB" for item in items)
    assert manifest["num_shards"] == 3 and manifest["exclude_scenes"] == ["sceneB"]
    assert manifest["episodes_per_route"] == 1 and manifest["max_routes"] == 5


@pytest.mark.parametrize(
    "extra, phrase",
    [
        (["--exclude-scenes", "sceneZ", "--num-shards", "2"], "does not have: ['sceneZ']"),
        (["--num-shards", "13"], "every shard needs a route"),
        (["--episodes-per-route", "-1", "--num-shards", "2"], "must be >= 0"),
    ],
)
def test_bad_selections_fail_and_write_nothing(tmp_path, extra, phrase):
    dataset = tmp_path / "train.json.gz"
    _train_json(dataset)
    result = _build("--dataset", str(dataset), "--output-dir", str(tmp_path / "out"), *extra)
    assert result.returncode == 2 and phrase in result.stderr
    assert not (tmp_path / "out").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["train.json.gz"]


def test_a_non_empty_output_directory_is_refused(tmp_path):
    dataset = tmp_path / "train.json.gz"
    _train_json(dataset)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("x")
    result = _build("--dataset", str(dataset), "--output-dir", str(tmp_path / "out"), "--num-shards", "2")
    assert result.returncode == 2 and "non-empty" in result.stderr
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["keep.txt"]


# ── the launcher against a fake tree ───────────────────────────────────────

# r2r_val_unseen.py's stand-in: logs its argv, then for every listed episode not yet in
# progress.json writes a record tar + .sha256 marker and a progress row naming it.
# $FAKE_FAIL_FIRST=N: the first invocation stops after N episodes and exits 1.
# $FAKE_NO_RECORD=1: rows without s2_record (a client that did not record).
STUB_CLIENT = r'''
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
calls = Path(os.environ["FAKE_CALLS"])
first = not calls.exists()
with calls.open("a") as handle:
    handle.write(json.dumps(argv) + "\n")
value = lambda flag: argv[argv.index(flag) + 1]
output, records = Path(value("--output_path")), Path(value("--s2_record_dir"))
output.mkdir(parents=True, exist_ok=True)
progress = output / "progress.json"
done = set()
if progress.exists():
    done = {(r["scene_id"], r["episode_id"]) for r in map(json.loads, progress.read_text().splitlines())}
limit = int(os.environ.get("FAKE_FAIL_FIRST", "-1")) if first else -1
written = 0
for item in json.loads(Path(value("--episode_list")).read_text())["episodes"]:
    key = (item["scene_id"], item["episode_id"])
    if key in done:
        continue
    if written == limit:
        raise SystemExit(1)
    row = {"scene_id": key[0], "episode_id": key[1], "success": 1.0, "spl": 0.5, "os": 1.0, "ne": 1.0,
           "ppa_applied_calls": 2}
    if os.environ.get("FAKE_NO_RECORD") != "1":
        tar = records / key[0] / f"{key[0]}_{key[1]:06d}.s2rec.tar"
        tar.parent.mkdir(parents=True, exist_ok=True)
        tar.write_bytes(b"tar")
        Path(str(tar) + ".sha256").write_text("x  name\n")
        row["s2_record"] = {"path": str(tar), "num_calls": 3, "num_steps": 9, "front_mismatch_calls": 0}
    with progress.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    written += 1
'''

FAKE_VLA_CLIENT = r'''
import os
import time
from pathlib import Path
from types import SimpleNamespace


class VLAClient:
    def __init__(self, server_addr, timeout_ms=0):
        self.port = int(server_addr.rsplit(":", 1)[1])

    def connect(self):
        deadline = time.time() + 30
        ready = Path(os.environ["FAKE_READY_DIR"])
        while time.time() < deadline and not ((ready / "model").exists() and (ready / "vo").exists()):
            time.sleep(0.05)

    def get_server_info(self):
        formats = ["ppa-online-amb3r-v1"] if self.port == int(os.environ["FAKE_MODEL_PORT"]) else ["json+jpeg"]
        return SimpleNamespace(supported_formats=formats, model_version="fake")

    def health_check(self):
        return True

    def close(self):
        pass
'''

STUB_MODEL_SERVER = r'''
import json
import os
import sys
import time
from pathlib import Path

print("Formal PPA online AMB3R runtime enabled", flush=True)
Path(os.environ["FAKE_READY_DIR"], "model_argv.json").write_text(json.dumps(sys.argv[1:]))
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

STUB_XVFB = r'''
import socket
import sys

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 6000 + int(sys.argv[1].lstrip(":"))))
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


def _bash_ok() -> bool:
    if BASH is None or not Path("/proc/self/cmdline").exists():
        return False
    probe = subprocess.run([BASH, "-c", 'set -u; a=(); : "${a[@]}"; mapfile -t b < /dev/null'], capture_output=True)
    return probe.returncode == 0


needs_bash5 = pytest.mark.skipif(not _bash_ok(), reason="the launcher targets bash >= 4.4 on Linux (/proc)")


class Tree:
    def __init__(self, tmp_path: Path, num_shards: int = 2):
        self.tmp = tmp_path
        self.root = tmp_path / "root"
        self.output = tmp_path / "collect"
        self.ready = tmp_path / "ready"
        self.calls = tmp_path / "client_calls.jsonl"
        self.model_port, self.vo_port = _free_port(), _free_port()
        self.display = _free_port(low=6000) - 6000

        def stub(relative: str, content: str = "stub\n") -> Path:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return path

        stub("HeatmapVLN/configs/ppa_action_refine_v2_8gpu.yaml")
        stub("HeatmapVLN/scripts/evaluation/r2r_val_unseen.py", STUB_CLIENT)
        stub("HeatmapVLN/scripts/evaluation/rpc_model_server.py", STUB_MODEL_SERVER)
        stub("HeatmapVLN/scripts/amb3r_vo/rpc_amb3r_vo_server.py", STUB_VO_SERVER)
        stub("rpc/src/vla_rpc/__init__.py", "")
        stub("rpc/src/vla_rpc/client.py", FAKE_VLA_CLIENT)
        stub("weights/ppa_refine_v2_best.pth")
        stub("amb3r/checkpoints/DA3NESTED-GIANT-LARGE/model.safetensors")
        stub("amb3r/slam/slam_config.yaml")
        (self.root / "InternNav_Model").mkdir()
        (tmp_path / "dataset" / "mp3d").mkdir(parents=True)
        self.ready.mkdir()
        self.xvfb = tmp_path / "bin" / "Xvfb"
        self.xvfb.parent.mkdir(parents=True)
        self.xvfb.write_text(f"#!{sys.executable}\n{STUB_XVFB}")
        self.xvfb.chmod(0o755)
        dataset = tmp_path / "train.json.gz"
        _train_json(dataset, scenes=("sceneA", "sceneB"), routes_per_scene=2)
        self.cohort = tmp_path / "cohort"
        result = _build("--dataset", str(dataset), "--output-dir", str(self.cohort), "--num-shards", str(num_shards))
        assert result.returncode == 0, result.stderr

    def env(self, **overrides) -> dict:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.tmp),
            "LC_ALL": "C",
            "PPA_EVAL_ROOT": str(self.root),
            "PPA_EVAL_SCENES_DIR": str(self.tmp / "dataset"),
            "PPA_EVAL_PYTHON": sys.executable,
            "PPA_EVAL_XVFB": str(self.xvfb),
            "PPA_EVAL_RUN_STAMP": "test",
            "PPA_EVAL_GPU_DEVICES": "0",
            "PPA_EVAL_DISPLAY_BASE": str(self.display),
            "PPA_EVAL_MODEL_PORT_BASE": str(self.model_port),
            "PPA_EVAL_VO_PORT_BASE": str(self.vo_port),
            "PPA_EVAL_SERVER_STAGGER_S": "0",
            "PPA_EVAL_SHARD_RETRIES": "0",
            "PPA_COLLECT_RETRY_PAUSE_S": "0",
            "PPA_COLLECT_COHORT_DIR": str(self.cohort),
            "PPA_COLLECT_OUTPUT_ROOT": str(self.output),
            "FAKE_CALLS": str(self.calls),
            "FAKE_READY_DIR": str(self.ready),
            "FAKE_MODEL_PORT": str(self.model_port),
        }
        for key, value in overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        return env

    def run(self, timeout: float = 120, **overrides) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, str(LAUNCHER)], env=self.env(**overrides), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)

    def client_calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def summary(self, result) -> dict:
        lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        return json.loads(lines[-1])


@pytest.fixture
def tree(tmp_path):
    if not _bash_ok():
        pytest.skip("the launcher targets bash >= 4.4 on Linux (/proc)")
    return Tree(tmp_path)


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


@needs_bash5
def test_every_shard_runs_with_the_collection_flags_and_is_done(tree):
    result = tree.run()
    assert result.returncode == 0, result.stdout + result.stderr
    calls = tree.client_calls()
    assert len(calls) == 2
    for tag, argv in zip(("000", "001"), sorted(calls, key=lambda a: _flag(a, "--episode_list"))):
        assert _flag(argv, "--dataset_split") == "train"
        assert _flag(argv, "--data_path") == str(tree.cohort / f"dataset_shard_{tag}.json.gz")
        assert _flag(argv, "--episode_list") == str(tree.cohort / f"shard_{tag}.json")
        assert _flag(argv, "--s2_record_dir") == str(tree.output / "records")
        assert _flag(argv, "--output_path") == str(tree.output / "workers" / f"shard_{tag}")
        assert _flag(argv, "--max_steps_per_episode") == "500"
        assert _flag(argv, "--auto_stop_distance") == "0"
        assert _flag(argv, "--num_history") == "8"
        assert _flag(argv, "--history_pose_source") == "amb3r_vo_da3"
        assert "--resume" in argv and "--rpc_require_deterministic_sampling" in argv
        assert "val_unseen" not in " ".join(argv)
        assert (tree.output / "workers" / f"shard_{tag}" / "DONE").exists()
    summary = tree.summary(result)
    assert summary["status"] == "complete" and summary["shards_done"] == 2
    assert summary["episodes"] == 12 and summary["records"] == 12 and summary["system2_calls"] == 36
    assert not list((tree.output / "workers").glob("*.claim"))
    assert "--system2_lora_adapter" not in json.loads((tree.ready / "model_argv.json").read_text())

    again = tree.run()
    assert again.returncode == 0 and len(tree.client_calls()) == 2  # DONE shards are not rerun


@needs_bash5
def test_a_shard_claimed_by_a_live_launcher_is_left_to_it(tree):
    holder = subprocess.Popen([BASH, "-c", "exec -a run_s2_dagger_collect_cuda.sh sleep 60"])
    try:
        claim = tree.output / "workers" / "shard_001.claim"
        claim.mkdir(parents=True)
        (claim / "owner").write_text(f"{holder.pid} other\n")
        result = tree.run()
    finally:
        holder.kill()
        holder.wait()
    assert result.returncode != 0
    summary = tree.summary(result)
    assert summary["status"] == "incomplete" and summary["shards_not_done_elsewhere_or_unstarted"] == ["001"]
    assert [_flag(a, "--episode_list") for a in tree.client_calls()] == [str(tree.cohort / "shard_000.json")]
    assert claim.is_dir()  # not ours to release


@needs_bash5
def test_a_stale_claim_is_taken_over(tree):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    claim = tree.output / "workers" / "shard_000.claim"
    claim.mkdir(parents=True)
    (claim / "owner").write_text(f"{gone.pid} crashed\n")
    result = tree.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tree.output / "workers" / "shard_000" / "DONE").exists() and not claim.exists()


@needs_bash5
def test_a_client_that_dies_is_restarted_and_resumes(tree):
    result = tree.run(PPA_EVAL_SHARD_RETRIES=1, PPA_COLLECT_SHARDS="0", FAKE_FAIL_FIRST=2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(tree.client_calls()) == 2
    rows = (tree.output / "workers" / "shard_000" / "progress.json").read_text().splitlines()
    listed = json.loads((tree.cohort / "shard_000.json").read_text())["episodes"]
    assert len(rows) == len(listed)


@needs_bash5
def test_a_shard_without_committed_records_is_not_done(tree):
    result = tree.run(PPA_COLLECT_SHARDS="1", FAKE_NO_RECORD=1)
    assert result.returncode != 0
    assert not (tree.output / "workers" / "shard_001" / "DONE").exists()
    assert tree.summary(result)["shards_failed"] == ["001"]
    assert "no s2_record in its progress row" in (tree.output / "runtime" / "test" / "logs" / "client_shard_001.log").read_text()


@needs_bash5
def test_a_local_adapter_reaches_the_model_server(tree):
    adapter = tree.tmp / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    result = tree.run(PPA_EVAL_SYSTEM2_LORA=adapter, PPA_COLLECT_SHARDS="0")
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads((tree.ready / "model_argv.json").read_text())
    assert _flag(argv, "--system2_lora_adapter") == str(adapter)


@needs_bash5
@pytest.mark.parametrize(
    "overrides, phrase",
    [
        ({"PPA_COLLECT_COHORT_DIR": None}, "PPA_COLLECT_COHORT_DIR is required"),
        ({"PPA_COLLECT_OUTPUT_ROOT": None}, "PPA_COLLECT_OUTPUT_ROOT is required"),
        ({"PPA_COLLECT_SHARDS": "0,x"}, "takes 'all' or indices"),
        ({"PPA_COLLECT_SHARDS": "2"}, "outside the cohort's 2 shards"),
        ({"PPA_EVAL_EXTERNAL_SERVERS": "1"}, "requires PPA_EVAL_EXTERNAL_SERVER_DIR"),
        ({"PPA_EVAL_GPU_DEVICES": "0,0", "PPA_EVAL_VO_GPU_DEVICES": "0,0"}, "GPU IDs must be unique"),
    ],
)
def test_preflight_refusals(tree, overrides, phrase):
    result = tree.run(**overrides)
    assert result.returncode == 2 and phrase in result.stderr
    assert not tree.client_calls()


@needs_bash5
def test_a_manifest_that_is_not_a_train_cohort_is_refused(tree):
    manifest = json.loads((tree.cohort / "manifest.json").read_text())
    manifest["split"] = "val_unseen"
    (tree.cohort / "manifest.json").write_text(json.dumps(manifest))
    result = tree.run()
    assert result.returncode == 2 and "not a build_r2r_train_cohorts.py train manifest" in result.stderr
