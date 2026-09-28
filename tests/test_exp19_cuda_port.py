"""EXP-19 on the RTX 4090 box: the rebuilt candidate table, the run-vs-log comparison and the ffprobe-less probe.

numpy / pandas / PIL (the synthetic tree); no torch, no simulator.
"""
from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pytest

from scripts.exp19 import build_records as br
from scripts.exp19 import compare_run_to_log as cmp
from scripts.exp19 import rebuild_cases as rc
from scripts.exp19 import synthetic_traces

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# rebuild_cases.py
# --------------------------------------------------------------------------- #
def _dataset(path: Path, drop=()) -> Path:
    episodes = []
    for _cat, _rank, key, _s42, _s1337 in rc.CANDIDATE_TABLE:
        if key in drop:
            continue
        scene, episode_id = rc.split_key(key)
        episodes.append({"episode_id": episode_id, "scene_id": f"mp3d/{scene}/{scene}.glb",
                         "info": {"geodesic_distance": 5.0 + episode_id / 1000},
                         "instruction": {"instruction_text": f"walk to {key}"}})
    episodes.append({"episode_id": 1, "scene_id": "mp3d/Other/Other.glb", "info": {}, "instruction": {}})
    with gzip.open(path, "wt") as fh:
        json.dump({"episodes": episodes}, fh)
    return path


def test_candidate_table_is_the_ledgers():
    """Five categories x ranks 0-2, no episode twice; rank 0 are the C500 main cases (ledger run record 3)."""
    by_cat = {}
    for cat, rank, key, _s42, _s1337 in rc.CANDIDATE_TABLE:
        by_cat.setdefault(cat, []).append((rank, key))
    assert list(by_cat) == ["T1", "T2", "T3", "F1", "F2"]
    assert all([r for r, _ in v] == [0, 1, 2] for v in by_cat.values())
    keys = [k for _c, _r, k, _a, _b in rc.CANDIDATE_TABLE]
    assert len(set(keys)) == 15
    assert [v[0][1] for v in by_cat.values()] == ["X7HyMhZNoso_0601", "EU6Fwq7SyZv_0346", "zsNo4HB9uLZ_0713",
                                                  "zsNo4HB9uLZ_0163", "x8F5xyUWy9e_0950"]


def test_rebuild_writes_what_build_records_reads(tmp_path):
    ds = _dataset(tmp_path / "val_unseen.json.gz")
    out = tmp_path / "cases"
    assert rc.main(["--dataset", str(ds), "--out-dir", str(out), "--num-gpus", "2"]) == 0
    cands = br.load_candidates(out / "candidates.json")
    assert cands["ordered"]["T1"] == ["X7HyMhZNoso_0601", "zsNo4HB9uLZ_1754", "TbHJrupSAjP_0472"]
    assert sorted(cands["members"]) == sorted(k for _c, _r, k, _a, _b in rc.CANDIDATE_TABLE)
    data = json.loads((out / "candidates.json").read_text())
    assert data["reconstructed"]["tool"] == "scripts/exp19/rebuild_cases.py"
    t1 = data["categories"]["T1"]["candidates"][0]
    assert (t1["scene_id"], t1["episode_id"], t1["steps_42"]) == ("X7HyMhZNoso", 601, 64)
    assert t1["instruction"] == "walk to X7HyMhZNoso_0601" and t1["geodesic_m"] == pytest.approx(5.601)
    # default lists: the five rank-0 cases; LPT puts the 500-step F2 alone on one GPU
    lists = [json.loads((out / "episode_lists" / f"gpu{j}.json").read_text()) for j in range(2)]
    keys = [[f"{e['scene_id']}_{e['episode_id']:04d}" for e in lst["episodes"]] for lst in lists]
    assert keys == [["x8F5xyUWy9e_0950"],
                    ["X7HyMhZNoso_0601", "EU6Fwq7SyZv_0346", "zsNo4HB9uLZ_0713", "zsNo4HB9uLZ_0163"]]
    assert all(set(e) == {"scene_id", "episode_id"} for lst in lists for e in lst["episodes"])
    assert lists[0]["dataset_sha256"] == rc.sha256_file(ds)


def test_rebuild_fallback_lists_and_refusals(tmp_path):
    ds = _dataset(tmp_path / "val_unseen.json.gz")
    out = tmp_path / "cases"
    assert rc.main(["--dataset", str(ds), "--out-dir", str(out), "--num-gpus", "2"]) == 0
    before = (out / "candidates.json").read_text()
    assert rc.main(["--dataset", str(ds), "--out-dir", str(out), "--num-gpus", "1", "--lists-only",
                    "--episodes", "zsNo4HB9uLZ_1754", "--lists-dir", str(out / "fb")]) == 0
    assert json.loads((out / "fb" / "gpu0.json").read_text())["episodes"] == [
        {"scene_id": "zsNo4HB9uLZ", "episode_id": 1754}]
    assert (out / "candidates.json").read_text() == before  # still the main run's lists
    assert json.loads((out / "episode_lists" / "gpu1.json").read_text())["episodes"][0]["episode_id"] == 601
    bad = tmp_path / "bad"
    assert rc.main(["--dataset", str(ds), "--out-dir", str(bad), "--episodes", "Other_0001"]) == 1
    assert rc.main(["--dataset", str(ds), "--out-dir", str(bad), "--num-gpus", "3", "--episodes",
                    "X7HyMhZNoso_0601", "EU6Fwq7SyZv_0346"]) == 1
    missing = _dataset(tmp_path / "missing.json.gz", drop=("QUCTc6BB5sX_1420",))
    assert rc.main(["--dataset", str(missing), "--out-dir", str(bad)]) == 1
    assert not bad.exists()


def test_rebuild_checks_scene_meshes(tmp_path):
    ds = _dataset(tmp_path / "val_unseen.json.gz")
    scenes = tmp_path / "scenes"
    for _c, _r, key, _a, _b in rc.CANDIDATE_TABLE:
        scene, _ = rc.split_key(key)
        (scenes / "mp3d" / scene).mkdir(parents=True, exist_ok=True)
        (scenes / "mp3d" / scene / f"{scene}.glb").write_bytes(b"x")
    assert rc.main(["--dataset", str(ds), "--out-dir", str(tmp_path / "ok"), "--scenes-dir", str(scenes)]) == 0
    (scenes / "mp3d" / "pLe4wQe7qrG" / "pLe4wQe7qrG.glb").unlink()
    assert rc.main(["--dataset", str(ds), "--out-dir", str(tmp_path / "no"), "--scenes-dir", str(scenes)]) == 1


# --------------------------------------------------------------------------- #
# compare_run_to_log.py on the synthetic tree
# --------------------------------------------------------------------------- #
def _client_log_from_refs(root: Path, path: Path) -> tuple:
    """A deployed client's log (the lines select_cases.parse_client_log reads) + progress rows, from the
    synthetic tree's eval-log references: episode A matches its rerun, B diverges at one call."""
    lines, progress = [], []
    refs = sorted((root / "cases" / "eval_log_reference").glob("*.json"))
    for n, ref_path in enumerate(refs, start=1):
        ref = json.loads(ref_path.read_text())
        lines.append(f"[{n}/{len(refs)}] Episode {ref['scene_id']}_{ref['episode_id']}: {'x' * 10}")
        for c in ref["calls"]:
            lines.append(f"step_id: {c['step']}, RPC kind={c['kind']}, VLM output: {c['vlm_output']}")
            lines.append(f"[debug] actions=[{', '.join(str(a) for a in c['actions'])}]")
        f = ref["final"]
        lines.append(f"=> success: {f['success']:.4f}, spl: {f['spl']:.4f}, os: {f['os']:.4f}, ne: {f['ne']:.4f}, "
                     f"vlm_calls: {f['vlm_calls']}, trajectory_calls: {f['trajectory_calls']}")
        progress.append({"scene_id": ref["scene_id"], "episode_id": ref["episode_id"], "steps": f["steps"]})
    path.write_text("\n".join(lines) + "\n")
    prog = path.with_suffix(".progress.json")
    prog.write_text("".join(json.dumps(r) + "\n" for r in progress))
    return path, prog


def test_compare_run_to_log_on_the_synthetic_tree(tmp_path):
    root = tmp_path / "exp19"
    assert synthetic_traces.main(["--root", str(root)]) == 0
    log, prog = _client_log_from_refs(root, tmp_path / "client.log")
    out = tmp_path / "cmp.json"
    run = root / "runs" / "synth"
    assert cmp.main(["--run-dir", str(run), "--client-log", str(log), "--progress", str(prog)]) == 1  # no DONE yet
    (run / "DONE").write_text(json.dumps({"status": "complete"}))
    assert cmp.main(["--run-dir", str(run), "--client-log", str(log), "--progress", str(prog), "--out", str(out),
                     "--require-call0", "--require-neutral"]) == 0
    rec = json.loads(out.read_text())
    by = {e["ep_key"]: e for e in rec["episodes"]}
    assert rec["n_compared"] == 2 and rec["all_call0_identical"] and rec["all_trace_neutral"]
    assert by["SynthSceneA_0007"]["trace_neutrality"]["n_trajectory_calls"] > 0
    assert by["SynthSceneA_0007"]["calls"]["all_identical"] and by["SynthSceneA_0007"]["outcome_comparison"]["same"]
    assert by["SynthSceneA_0007"]["outcome_comparison"]["steps_equal"]
    b = by["SynthSceneB_0042"]
    assert b["calls"]["first_divergent_call"] == 9 and b["first_divergence"]["rerun"]["call_index"] == 9
    assert not rec["all_identical"]
    assert cmp.main(["--run-dir", str(run), "--client-log", str(log), "--progress", str(prog),
                     "--require-identical"]) == 1


def test_compare_run_to_log_gate_fails_on_a_changed_call0(tmp_path):
    root = tmp_path / "exp19"
    assert synthetic_traces.main(["--root", str(root)]) == 0
    log, prog = _client_log_from_refs(root, tmp_path / "client.log")
    text = log.read_text().splitlines()
    i = next(n for n, line in enumerate(text) if line.startswith("step_id: "))
    text[i] = text[i] + " (changed)"
    log.write_text("\n".join(text) + "\n")
    run = root / "runs" / "synth"
    (run / "DONE").write_text(json.dumps({"status": "complete"}))
    assert cmp.main(["--run-dir", str(run), "--client-log", str(log), "--progress", str(prog),
                     "--require-call0"]) == 1
    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n")
    assert cmp.main(["--run-dir", str(run), "--client-log", str(empty)]) == 1


# --------------------------------------------------------------------------- #
# animate_v2: probing without ffprobe (imageio-ffmpeg ships ffmpeg only)
# --------------------------------------------------------------------------- #
# stderr of `ffmpeg -hide_banner -i <file> -map 0:v:0 -c copy -f null -` (ffmpeg 8.0.1) for an H.264 MP4 and a
# GIF of the animations' formats; ffprobe on the same files gave the values asserted below.
MP4_STDERR = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'a.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:00:03.00, start: 0.000000, bitrate: 174 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1426 [SAR 1:1 DAR 960:713], 170 kb/s, 12 fps, 12 tbr, 12288 tbn (default)
      Metadata:
        handler_name    : VideoHandler
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
Output #0, null, to 'pipe:':
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1426 [SAR 1:1 DAR 960:713], q=2-31, 170 kb/s, 12 fps, 12 tbr, 12288 tbn (default)
frame=   12 fps=0.0 q=-1.0 size=N/A time=00:00:01.00 bitrate=N/A speed=N/A\rframe=   36 fps=0.0 q=-1.0 Lsize=N/A time=00:00:02.91 bitrate=N/A speed= 520x
"""
GIF_STDERR = """Input #0, gif, from 'a.gif':
  Duration: 00:00:03.00, start: 0.000000, bitrate: 1209 kb/s
  Stream #0:0: Video: gif, bgra, 900x668 [SAR 64:64 DAR 225:167], 6.25 fps, 6 tbr, 100 tbn
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
Output #0, null, to 'pipe:':
  Stream #0:0: Video: gif, bgra, 900x668 [SAR 64:64 DAR 225:167], q=2-31, 6.25 fps, 6 tbr, 100 tbn
frame=   18 fps=0.0 q=-1.0 Lsize=N/A time=00:00:02.83 bitrate=N/A speed= 300x
"""


def test_probe_without_ffprobe_matches_ffprobe():
    an = pytest.importorskip("scripts.exp19.figures.animate_v2")
    assert an.parse_ffmpeg_probe(MP4_STDERR) == {"codec": "h264", "pix_fmt": "yuv420p", "width": 1920, "height": 1426,
                                                 "frame_rate": "12/1", "duration_s": 3.0, "frames": "36"}
    assert an.parse_ffmpeg_probe(GIF_STDERR) == {"codec": "gif", "pix_fmt": "bgra", "width": 900, "height": 668,
                                                 "frame_rate": "6/1", "duration_s": 3.0, "frames": "18"}
    assert an.parse_ffmpeg_probe("no video here") == {}


def test_probe_uses_ffmpeg_when_ffprobe_is_missing(tmp_path, monkeypatch):
    an = pytest.importorskip("scripts.exp19.figures.animate_v2")
    video = tmp_path / "a.mp4"
    video.write_bytes(b"\0" * 10)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["stdin"] = cmd, kw.get("stdin")
        return subprocess.CompletedProcess(cmd, 0, stdout=MP4_STDERR)

    monkeypatch.setattr(an.subprocess, "run", fake_run)
    info = an.probe(video, str(tmp_path / "bin" / "ffmpeg-linux-x86_64-v7.0.2"))
    assert seen["cmd"][1:] == ["-hide_banner", "-i", str(video), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"]
    assert seen["stdin"] == subprocess.DEVNULL
    assert (info["width"], info["height"], info["frames"], info["bytes"]) == (1920, 1426, "36", 10)
    assert "no ffprobe" in info["probe_tool"]


# --------------------------------------------------------------------------- #
# the two CUDA launchers parse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["scripts/exp19/run_rerun_cuda.sh", "scripts/exp19/run_post_cuda.sh"])
def test_cuda_launchers_parse(script):
    assert subprocess.run(["bash", "-n", str(REPO / script)]).returncode == 0


# --------------------------------------------------------------------------- #
# merge_runs.py (the main-case fallback reads two reruns as one)
# --------------------------------------------------------------------------- #
def _split_runs(root: Path) -> None:
    """runs/A and runs/B, one synthetic episode each, with DONE files as run_rerun_cuda.sh writes them."""
    import shutil

    src = root / "runs" / "synth" / "gpu0"
    for name, keep in (("A", "SynthSceneA_0007"), ("B", "SynthSceneB_0042")):
        dst = root / "runs" / name / "gpu0"
        shutil.copytree(src, dst)
        for sub in ("steps", "trace"):
            for d in (dst / sub).iterdir():
                if d.is_dir() and d.name != keep:
                    shutil.rmtree(d)
        done = {"schema": "exp19-run-done-v1", "run": name, "git_sha": "abc", "src": "/src",
                "platform": {"accelerator": "cuda"}, "code_sha256": {"x": "1"}, "protocol_seed": 42,
                "trace_diagnostics": True, "source_fingerprint": {"start": "f", "end": "f", "unchanged": True},
                "started_utc": f"2026-09-28T0{len(name)}:00:00Z", "finished_utc": "2026-09-28T09:00:00Z",
                "status": "complete", "trace_error_lines": 0,
                "ranks": [{"rank": 0, "gpu": 4, "client_exit_code": 0, "episodes": [{"ep_key": keep, "complete": True}]}]}
        (root / "runs" / name / "DONE").write_text(json.dumps(done))


def test_merge_runs_reads_two_reruns_as_one(tmp_path):
    from scripts.exp19 import merge_runs as mr

    root = tmp_path / "exp19"
    assert synthetic_traces.main(["--root", str(root)]) == 0
    _split_runs(root)
    assert mr.main(["--exp-root", str(root), "--out", "AB", "A", "B"]) == 0
    merged = root / "runs" / "AB"
    assert sorted(br.discover_run(merged)) == ["SynthSceneA_0007", "SynthSceneB_0042"]
    assert os_readlink(merged / "gpu1") == "../B/gpu0"
    done = json.loads((merged / "DONE").read_text())
    assert done["status"] == "complete" and [m["run"] for m in done["merged_from"]] == ["A", "B"]
    assert [(r["rank"], r["source_run"]) for r in done["ranks"]] == [(0, "A"), (1, "B")]
    assert mr.main(["--exp-root", str(root), "--out", "AB", "A", "B"]) == 1  # never overwritten


def test_merge_runs_refuses_runs_of_different_code_or_overlapping_episodes(tmp_path):
    from scripts.exp19 import merge_runs as mr

    root = tmp_path / "exp19"
    assert synthetic_traces.main(["--root", str(root)]) == 0
    _split_runs(root)
    done_b = root / "runs" / "B" / "DONE"
    d = json.loads(done_b.read_text())
    done_b.write_text(json.dumps(dict(d, git_sha="other")))
    assert mr.main(["--exp-root", str(root), "--out", "X", "A", "B"]) == 1
    done_b.write_text(json.dumps(dict(d, status="incomplete")))
    assert mr.main(["--exp-root", str(root), "--out", "X", "A", "B"]) == 1
    d["ranks"][0]["episodes"][0]["ep_key"] = "SynthSceneA_0007"
    done_b.write_text(json.dumps(d))
    assert mr.main(["--exp-root", str(root), "--out", "X", "A", "B"]) == 1
    assert not (root / "runs" / "X").exists()


def os_readlink(path: Path) -> str:
    import os

    return os.readlink(path)


def test_compare_run_to_log_trace_neutrality_gate(tmp_path):
    root = tmp_path / "exp19"
    assert synthetic_traces.main(["--root", str(root)]) == 0
    log, prog = _client_log_from_refs(root, tmp_path / "client.log")
    run = root / "runs" / "synth"
    (run / "DONE").write_text(json.dumps({"status": "complete"}))
    path = next(p for p in sorted(run.glob("gpu0/trace/*/call_*.json"))
                if json.loads(p.read_text())["response"].get("kind") == "trajectory")
    c = json.loads(path.read_text())
    path.write_text(json.dumps(dict(c, actions_match=False)))
    assert cmp.main(["--run-dir", str(run), "--client-log", str(log), "--progress", str(prog),
                     "--require-call0"]) == 0
    assert cmp.main(["--run-dir", str(run), "--client-log", str(log), "--progress", str(prog),
                     "--require-neutral"]) == 1
